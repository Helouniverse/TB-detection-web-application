import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import argparse
import json
import datetime
import shutil
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch import optim
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import sys
import subprocess

from dataset import Tbx11kDataset
from model import SymFormer

class SymFormerLoss(torch.nn.Module):
    def __init__(self, lambda_det=1.0):
        super().__init__()
        self.lambda_det = lambda_det
        self.cls_criterion = torch.nn.CrossEntropyLoss()
        self.det_criterion = torch.nn.BCEWithLogitsLoss()
        
    def forward(self, logits, targets, det_maps, gt_masks):
        # 1. Classification Loss (Cross Entropy)
        loss_cls = self.cls_criterion(logits, targets)
        
        # 2. Detection Loss
        tb_indices = (targets >= 2).nonzero(as_tuple=True)[0]
        
        loss_det = torch.tensor(0.0, device=logits.device)
        if len(tb_indices) > 0:
            det_maps_tb = det_maps[tb_indices] # (N, H, W)
            gt_masks_tb = gt_masks[tb_indices].squeeze(1) # (N, H, W)
            loss_det = self.det_criterion(det_maps_tb, gt_masks_tb)
            
        total_loss = loss_cls + (self.lambda_det * loss_det)
        return total_loss, loss_cls, loss_det


matplotlib.use('Agg') # For saving plots headlessly

def calculate_cls_metrics(preds, labels):
    """
    Calculates 4-class overall Accuracy, and Binary (TB vs Non-TB) Sensitivity, Specificity, and Precision.
    """
    # Overall 4-class accuracy
    overall_acc = (preds == labels).sum().item() / max(len(preds), 1)

    # Binary mapping: TB (2, 3) -> 1, Non-TB (0, 1) -> 0
    bin_preds = np.where(preds >= 2, 1, 0)
    bin_labels = np.where(labels >= 2, 1, 0)

    TP = ((bin_preds == 1) & (bin_labels == 1)).sum().item()
    TN = ((bin_preds == 0) & (bin_labels == 0)).sum().item()
    FP = ((bin_preds == 1) & (bin_labels == 0)).sum().item()
    FN = ((bin_preds == 0) & (bin_labels == 1)).sum().item()
    
    eps = 1e-6
    sensitivity = TP / (TP + FN + eps)
    specificity = TN / (TN + FP + eps)
    precision = TP / (TP + FP + eps)
    
    return overall_acc, sensitivity, specificity, precision

def calculate_det_iou(pred_maps, gt_masks, labels, threshold=0.5):
    """
    Calculates Mean Mask IoU and AP50 Object Detection equivalent for TB cases (labels >= 2).
    pred_maps: attention maps upscaled to original mask size
    gt_masks: ground truth bounding box binary masks
    labels: classification labels (4 classes)
    """
    tb_indices = (labels >= 2).nonzero(as_tuple=True)[0]
    if len(tb_indices) == 0:
        return 0.0, 0.0 # No TB cases to evaluate detection
        
    ious = []
    ap50_hits = 0
    
    for idx in tb_indices:
        pred = (pred_maps[idx] > threshold).float()
        gt = gt_masks[idx].float()
        
        intersection = (pred * gt).sum().item()
        union = pred.sum().item() + gt.sum().item() - intersection
        
        iou = intersection / (union + 1e-6) if union > 0 else 0.0
        ious.append(iou)
        if iou >= 0.5:
            ap50_hits += 1
            
    mean_iou = sum(ious) / len(ious)
    ap50 = ap50_hits / len(ious)
    return mean_iou, ap50

def evaluate_epoch(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    epoch_cls_loss = 0.0
    epoch_attn_loss = 0.0
    
    all_preds = []
    all_labels = []
    
    total_det_iou = 0.0
    total_det_ap50 = 0.0
    batches_with_tb = 0
    
    with torch.no_grad():
        for images, labels, masks in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            
            logits, attn_maps = model(images)
            loss, lcls, lattn = criterion(logits, labels, attn_maps, masks)
            
            epoch_loss += loss.item() * images.size(0)
            epoch_cls_loss += lcls.item() * images.size(0)
            epoch_attn_loss += lattn.item() * images.size(0)
            
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Extract detection metrics for TB images
            if (labels >= 2).sum() > 0:
                # Apply sigmoid to det_maps for IoU calc
                det_probs = torch.sigmoid(attn_maps) 
                # Up/down sample if size mismatch
                if det_probs.shape[1:] != masks.shape[2:]:
                    det_probs = F.interpolate(det_probs.unsqueeze(1), size=(masks.shape[2], masks.shape[3]), 
                                                mode='bilinear', align_corners=False).squeeze(1)
                iou, ap50 = calculate_det_iou(det_probs, masks.squeeze(1), labels)
                total_det_iou += iou
                total_det_ap50 += ap50
                batches_with_tb += 1
                
    avg_loss = epoch_loss / len(dataloader.dataset)
    avg_cls_loss = epoch_cls_loss / len(dataloader.dataset)
    avg_attn_loss = epoch_attn_loss / len(dataloader.dataset)
    
    acc, sens, spec, prec = calculate_cls_metrics(np.array(all_preds), np.array(all_labels))
    
    mean_iou = total_det_iou / batches_with_tb if batches_with_tb > 0 else 0.0
    mean_ap50 = total_det_ap50 / batches_with_tb if batches_with_tb > 0 else 0.0
    
    return avg_loss, avg_cls_loss, avg_attn_loss, acc, sens, spec, prec, mean_iou, mean_ap50

def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    
    # 0. Setup Experiment Directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"symformer_lr{config['lr']}_bs{config['batch_size']}_{timestamp}"
    
    # Resolve paths reliably relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config['data_dir'] = os.path.abspath(os.path.join(script_dir, config['data_dir']))
    config['save_dir'] = os.path.abspath(os.path.join(script_dir, config['save_dir']))
    
    exp_dir = os.path.join(config['save_dir'], exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save config to exp folder
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
        
    print(f"Using device: {device}")
    print(f"Experiment Directory automatically created: {exp_dir}")
    
    train_transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.RandomRotation(15), 
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    full_train_dataset = Tbx11kDataset(
        csv_file=os.path.join(config['data_dir'], 'data.csv'),
        img_dir=os.path.join(config['data_dir'], 'images'),
        split='train',
        transform=train_transform,
        mask_size=(config['img_size'], config['img_size'])
    )
    
    val_size = int(0.2 * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    train_subset, val_subset = random_split(full_train_dataset, [train_size, val_size])
    val_subset.dataset.transform = val_test_transform
    
    test_dataset = Tbx11kDataset(
        csv_file=os.path.join(config['data_dir'], 'data.csv'),
        img_dir=os.path.join(config['data_dir'], 'images'),
        split='val',
        transform=val_test_transform,
        mask_size=(config['img_size'], config['img_size'])
    )
    
    train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    
    # Subset of only TB images for Phase 1
    tb_indices_train = []
    for idx in train_subset.indices:
        # Check target column ('tb' meaning Active or Latent)
        if full_train_dataset.grouped_df.iloc[idx]['target'] == 'tb':
            tb_indices_train.append(idx)
            
    tb_train_subset = torch.utils.data.Subset(full_train_dataset, tb_indices_train)
    tb_train_loader = DataLoader(tb_train_subset, batch_size=config['batch_size'], shuffle=True, num_workers=4)

    val_loader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    
    print("Building Model... (SymFormer)")
    model = SymFormer(num_classes_test=4)
    model.to(device)
    
    criterion = SymFormerLoss(lambda_det=config.get('lambda_det', 1.0))
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    
    # Track split histories
    history = {
        'train_loss': [], 'val_loss': [], 
        'train_cls': [], 'val_cls': [],
        'train_attn': [], 'val_attn': [],
        'val_acc': [], 'val_sens': [], 'val_spec': [],
        'val_iou': [], 'val_ap50': []
    }
    
    best_val_loss = float('inf')
    best_model_path = os.path.join(exp_dir, 'best_model.pth')
    
    epochs_stage1 = config['epochs'] // 2
    
    print(f"Starting Two-Stage Training for {config['epochs']} epochs...")
    for epoch in range(config['epochs']):
        
        if epoch == 0:
            print("\n--- Phase 1: Lesion Detection Learning (TB Only) ---")
            current_loader = tb_train_loader
            current_optimizer = optimizer
        elif epoch == epochs_stage1:
            print("\n--- Phase 2: Classifier Fine-Tuning (Full Dataset & Frozen Backbone) ---")
            current_loader = train_loader
            # Freeze backbone and det heads
            for param in model.backbone.parameters():
                param.requires_grad = False
            for param in model.fpn.parameters():
                param.requires_grad = False
            for param in model.spe.parameters():
                param.requires_grad = False
            for param in model.sym_attn.parameters():
                param.requires_grad = False
            for param in model.det_cls_conv.parameters():
                param.requires_grad = False
            for param in model.det_heatmap_out.parameters():
                param.requires_grad = False
            # Optimizer for classifier only
            current_optimizer = optim.AdamW(model.cls_head.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
            
        model.train()
        epoch_loss, epoch_cls, epoch_attn = 0.0, 0.0, 0.0
        
        pbar = tqdm(current_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [TRAIN]")
        for images, labels, masks in pbar:
            images = images.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            
            current_optimizer.zero_grad()
            logits, attn_maps = model(images)
            loss, lcls, lattn = criterion(logits, labels, attn_maps, masks)
            
            loss.backward()
            current_optimizer.step()
            
            epoch_loss += loss.item() * images.size(0)
            epoch_cls += lcls.item() * images.size(0)
            epoch_attn += lattn.item() * images.size(0)
            
            pbar.set_postfix({'Total': f"{loss.item():.3f}", 'Cls': f"{lcls.item():.3f}", 'Det': f"{lattn.item():.3f}"})
            
        train_len = len(current_loader.dataset)
        train_loss = epoch_loss / train_len
        train_cls_loss = epoch_cls / train_len
        train_attn_loss = epoch_attn / train_len
        
        val_loss, val_cls, val_attn, val_acc, val_sens, val_spec, val_prec, val_iou, val_ap50 = evaluate_epoch(model, val_loader, criterion, device)
        
        print(f"[Epoch {epoch+1} Valid] Total Loss: {val_loss:.4f} | Cls: {val_cls:.4f} | Det: {val_attn:.4f}")
        print(f"  --> Classification : Acc: {val_acc*100:.2f}% | Sens: {val_sens*100:.2f}% | Spec: {val_spec*100:.2f}%")
        print(f"  --> Detection      : Mean IoU: {val_iou:.4f} | AP50: {val_ap50*100:.2f}%")
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_cls'].append(train_cls_loss)
        history['val_cls'].append(val_cls)
        history['train_attn'].append(train_attn_loss)
        history['val_attn'].append(val_attn)
        history['val_acc'].append(val_acc)
        history['val_sens'].append(val_sens)
        history['val_spec'].append(val_spec)
        history['val_iou'].append(val_iou)
        history['val_ap50'].append(val_ap50)
        
        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"--> Saved better model (Val Loss: {val_loss:.4f}) to {best_model_path}")
            
    # 4. Plotting 4-grid Visualization
    print("Generating Training Curves...")
    epochs_range = range(1, config['epochs'] + 1)
    
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Total Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Total Loss')
    plt.plot(epochs_range, history['val_loss'], label='Val Total Loss')
    plt.title('Overall Hybrid Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    
    # Plot 2: Separated Losses (Cls vs Det)
    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, history['train_cls'], label='Train Classification Loss', linestyle='-', color='dodgerblue')
    plt.plot(epochs_range, history['val_cls'], label='Val Classification Loss', linestyle='--', color='deepskyblue')
    plt.plot(epochs_range, history['train_attn'], label='Train Detection Loss', linestyle='-', color='crimson')
    plt.plot(epochs_range, history['val_attn'], label='Val Detection Loss', linestyle='--', color='lightcoral')
    plt.title('Loss Breakdown (Classification vs Object Detection)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    
    # Plot 3: Classification Metrics
    plt.subplot(2, 2, 3)
    plt.plot(epochs_range, np.array(history['val_acc'])*100, label='Accuracy')
    plt.plot(epochs_range, np.array(history['val_sens'])*100, label='Sensitivity (TB Recall)')
    plt.plot(epochs_range, np.array(history['val_spec'])*100, label='Specificity')
    plt.title('Classification Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Percentage (%)')
    plt.legend()
    plt.grid()
    
    # Plot 4: Detection Metrics
    plt.subplot(2, 2, 4)
    plt.plot(epochs_range, history['val_iou'], label='Mean Mask IoU')
    plt.plot(epochs_range, history['val_ap50'], label='AP50 Equivalent (IoU > 0.5)', linestyle='--')
    plt.title('Object Detection Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Score (0 to 1)')
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    graph_path = os.path.join(exp_dir, 'training_curves.png')
    plt.savefig(graph_path)
    print(f"Graph saved to {graph_path}")
    
    # 5. Final Test Phase
    print("\n--- Final Test on Unseen Split ---")
    model.load_state_dict(torch.load(best_model_path)) 
    
    test_loss, test_cls, test_attn, test_acc, test_sens, test_spec, test_prec, test_iou, test_ap50 = evaluate_epoch(model, test_loader, criterion, device)
    
    test_results = {
        'test_loss_total': test_loss, 'test_loss_cls': test_cls, 'test_loss_attn': test_attn,
        'accuracy': test_acc, 'sensitivity': test_sens, 'specificity': test_spec,
        'precision': test_prec, 'mean_iou': test_iou, 'ap50': test_ap50
    }
    
    with open(os.path.join(exp_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=4)
        
    print(f"Test Loss:        Overall {test_loss:.4f} | Cls {test_cls:.4f} | Det {test_attn:.4f}")
    print(f"Test Accuracy:    {test_acc*100:.2f}%")
    print(f"Test Sensitivity: {test_sens*100:.2f}%")
    print(f"Test Specificity: {test_spec*100:.2f}%")
    print(f"Test Precision:   {test_prec*100:.2f}%")
    print(f"Test Mean IoU:    {test_iou:.4f}")
    print(f"Test AP50 Eq.:    {test_ap50*100:.2f}%")
    print("----------------------------------\n")
    print(f"Training Pipeline complete! All results mathematically logged to: {exp_dir}")
    
    # 6. Automatic Inference Visualization Trigger
    print("\n--- Generating Visual Inference Samples ---")
    inference_cmd = [
        sys.executable, "inference.py", 
        "--exp_dir", exp_dir,
        "--data_dir", os.path.abspath(config['data_dir']),
        "--img_size", str(config['img_size'])
    ]
    subprocess.run(inference_cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    print(f"Inference complete! Visualizations available inside {exp_dir}/inference_visuals/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train SymFormer Model")
    default_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
    parser.add_argument('--config', type=str, default=default_config_path, help='Path to hyperparameter config JSON')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config_params = json.load(f)
        
    train(config_params)
