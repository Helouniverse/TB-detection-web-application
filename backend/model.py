import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
import math
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class SinePositionalEncoding2D(nn.Module):
    def __init__(self, d_model, height, width):
        super().__init__()
        assert d_model % 2 == 0
        pe = torch.zeros(d_model, height, width)
        d_model_half = d_model // 2
        
        div_term = torch.exp(torch.arange(0., d_model_half, 2) * -(math.log(10000.0) / d_model_half))
        
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        
        pe[0:d_model_half:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model_half:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model_half::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model_half+1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        
        self.register_buffer('pe', pe.unsqueeze(0)) # shape (1, d_model, H, W)

    def forward(self, x):
        return self.pe[:, :, :x.size(2), :x.size(3)]

class STN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 6) # Affine matrix 2x3
        )
        # Initialize to identity
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.conv_net(x)
        xs = F.adaptive_avg_pool2d(xs, (1, 1)).view(xs.size(0), -1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        return theta

class SPE(nn.Module):
    def __init__(self, channels, max_height=256, max_width=256):
        super().__init__()
        self.pe_generator = SinePositionalEncoding2D(channels, max_height, max_width)
        self.stn = STN(channels)

    def forward(self, f_i):
        B, C, H, W = f_i.shape
        # Base PE
        pe = self.pe_generator(f_i).expand(B, -1, -1, -1).to(f_i.device)
        
        half_w = W // 2
        # P_right: Right side is index half_w onwards. 
        # But if total W is odd, wait... W // 2 for left, and right. Let's split exactly matching P_left and P_right halves for flip.
        
        p_left = pe[:, :, :, :half_w]
        p_right = pe[:, :, :, W-half_w:] # ensure shapes match exactly
        
        # Predict Affine Theta
        theta = self.stn(f_i)
        
        # Affine Transform P_right to left geometry (or rather transform P_right itself)
        grid = F.affine_grid(theta, p_right.size(), align_corners=False)
        p_right_trans = F.grid_sample(p_right, grid, align_corners=False)
        
        # Horizontal flip
        p_right_flipped = torch.flip(p_right_trans, dims=[3])
        
        # Middle column if W is odd
        if W % 2 != 0:
            p_mid = pe[:, :, :, half_w:half_w+1]
            p_sym = torch.cat([p_right_flipped, p_mid, p_right], dim=3)
        else:
            p_sym = torch.cat([p_right_flipped, p_right], dim=3)
        
        f_recalib = f_i + p_sym
        return f_recalib

class SymAttention(nn.Module):
    def __init__(self, channels, num_heads=8, num_points=4):
        super().__init__()
        self.M = num_heads
        self.K = num_points
        self.C = channels
        
        # Predict \Delta p_x, \Delta p_y
        self.offset_conv = nn.Conv2d(channels, self.M * self.K * 2, kernel_size=3, padding=1)
        
        # Predict Attention Weights
        self.attn_conv = nn.Conv2d(channels, self.M * self.K, kernel_size=3, padding=1)
        
        # Predict Features (Value matrix)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1)
        
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 2, channels, kernel_size=1)
        )

    def forward(self, f_recalib, f_orig):
        B, C, H, W = f_recalib.shape
        
        # Predict Offsets
        offsets = self.offset_conv(f_recalib) # B, M*K*2, H, W
        offsets = offsets.view(B, self.M, self.K, 2, H, W)
        
        # Predict Attention
        attn = self.attn_conv(f_recalib) # B, M*K, H, W
        attn = attn.view(B, self.M, self.K, H, W)
        attn = F.softmax(attn, dim=2)
        
        # Values
        value = self.value_conv(f_recalib) # B, C, H, W
        
        # Create base coordinate grid [0, 1] mapped to [-1, 1] for grid_sample
        # Symmetrical property: we want to sample from (W - x) location
        gy, gx = torch.meshgrid(torch.arange(H, device=f_recalib.device), 
                                torch.arange(W, device=f_recalib.device), indexing='ij')
                                
        # Symmetrical Search: x_sym = W - 1 - gx
        gx_sym = (W - 1) - gx
        
        # Reshape for broadcasting
        gy = gy.view(1, 1, 1, H, W).float()
        gx_sym = gx_sym.view(1, 1, 1, H, W).float()
        
        # Add offsets: offsets are in raw pixel space loosely
        grid_x = gx_sym + offsets[:, :, :, 0, :, :]
        grid_y = gy + offsets[:, :, :, 1, :, :]
        
        # Normalize to [-1, 1]
        grid_x = (grid_x / (W - 1)) * 2 - 1
        grid_y = (grid_y / (H - 1)) * 2 - 1
        
        # grid shape needed for grid_sample: (B*M, K*H, W, 2)
        grid = torch.stack((grid_x, grid_y), dim=-1)
        grid = grid.view(B*self.M, self.K*H, W, 2)
        
        value_split = value.view(B*self.M, C//self.M, H, W)
        
        sampled_value = F.grid_sample(value_split, grid, align_corners=True) # (B*M, C/M, K*H, W)
        sampled_value = sampled_value.view(B, self.M, C//self.M, self.K, H, W)
        
        attn = attn.view(B, self.M, 1, self.K, H, W)
        
        out = (sampled_value * attn).sum(dim=3) # Output: B, M, C/M, H, W
        out = out.view(B, C, H, W)
        
        out = self.proj(out) + f_orig
        out = self.mlp(out) + out
        return out

class SymFormer(nn.Module):
    def __init__(self, num_classes_test=3, num_classes_det=2):
        super().__init__()
        # Backbone ResNet50
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = create_feature_extractor(resnet, return_nodes={
            'layer1': 'feat1', # usually stride 4
            'layer2': 'feat2', # stride 8
            'layer3': 'feat3', # stride 16
            'layer4': 'feat4'  # stride 32
        })
        
        # FPN
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=256
        )
        
        # SAS (Symmetric Abnormality Search) for each FPN layer
        # Can share weights or have separate. Paper: "share the same weights"
        self.spe = SPE(256)
        self.sym_attn = SymAttention(256)
        
        # CXR Classification Head (applied to top level F4 enhanced)
        self.cls_head = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes_test)
        )
        
        # Detection Heads (RetinaNet Style)
        # Shared across FPN scales
        self.det_cls_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
        )
        # Anchor boxes = 9 per location usually. Let's keep it simple: 1 anchor per location for dummy,
        # or use torchvision's built-in RetinaNet.
        # Since we're doing a simplified one-stage detection head logic without torchvision's full RetinaNet:
        # We will output a heatmap pseudo-probability for detection to keep the pipeline similar to before
        # unless full RetinaNet anchor processing is built.
        # Let's output a 1-channel pseudo-heatmap per FPN level for bbox proxy
        self.det_heatmap_out = nn.Conv2d(256, 1, kernel_size=3, padding=1)

    def forward(self, x):
        B, C, H, W = x.shape
        features = self.backbone(x)
        
        # Keys match in_channels_list order
        features = [features['feat1'], features['feat2'], features['feat3'], features['feat4']]
        fpn_dict = {
            'feat1': features[0],
            'feat2': features[1],
            'feat3': features[2],
            'feat4': features[3]
        }
        
        fpn_features = self.fpn(fpn_dict)
        enhanced_fpn = {}
        
        # SAS Processing
        for k, feat in fpn_features.items():
            f_recalib = self.spe(feat)
            f_enhanced = self.sym_attn(f_recalib, feat)
            enhanced_fpn[k] = f_enhanced
            
        # Top-level Classification (F4)
        top_feat = enhanced_fpn['feat4']
        img_cls_logits = self.cls_head(top_feat)
        
        # Detection pseudo-heatmaps (simplification of RetinaNet anchors for the pipeline)
        det_maps = []
        for k in ['feat1', 'feat2', 'feat3', 'feat4']:
            det_feat = self.det_cls_conv(enhanced_fpn[k])
            det_heatmap = self.det_heatmap_out(det_feat) # (B, 1, H', W')
            # upscale to original
            det_maps.append(F.interpolate(det_heatmap, size=(H, W), mode='bilinear', align_corners=False))
            
        # Average or Max over FPN levels for final pseudo-heatmap
        final_det_map = torch.stack(det_maps, dim=0).mean(dim=0).squeeze(1) # (B, H, W)
        
        return img_cls_logits, final_det_map

if __name__ == '__main__':
    model = SymFormer()
    x = torch.randn(2, 3, 512, 512)
    cls, det = model(x)
    print(f"Cls output: {cls.shape}")
    print(f"Det output: {det.shape}")
