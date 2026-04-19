# TB-Detection Web Application

A web-based clinical dashboard for Tuberculosis detection utilizing the SymFormer PyTorch model. This application features a FastAPI backend that runs inference on a symmetric transformer architecture and a pure Vanilla JS/HTML/CSS frontend.

## Features
- **FastAPI Backend**: Efficient image tensor preprocessing, threshold mapping, bounding box generation, and dynamic heatmap Alpha blending.
- **Vanilla Frontend**: A dark-mode, responsive graphical user interface managing drag-and-drop file uploads. It visually displays bounding boxes, multi-class predictions, and heatmap overlays without relying on any external JS or CSS frameworks.
- **SymFormer Inference**: Capable of assessing PA/AP chest radiographs for 4 specific target classes: *Healthy*, *Sick (Non-TB)*, *Active TB*, and *Latent TB*.

## Repository Structure
```text
.
├── backend/
│   ├── app.py          # FastAPI application & image processing logic
│   └── model.py        # SymFormer PyTorch network definition
├── frontend/
│   └── index.html      # Self-contained frontend user interface
├── Dockerfile          # Configuration for containerized deployment
├── railway.toml        # Railway.app build/deploy configurations 
└── requirements.txt    # Frozen Python pip dependencies
```

## Running Locally

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Helouniverse/TB-detection-web-application.git
   cd TB-detection-web-application
   ```

2. **Add Model Weights**
   Ensure your PyTorch weight file (e.g., `best_model.pth`) is securely downloaded. 
   *(Note: Model weight `.pth` files are ignored by git to keep this repository lightweight).*

3. **Install Dependencies**
   It's strongly recommended to use a virtual environment.
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   # We use PyTorch CPU wheels to minimize overhead for web serving
   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
   ```

4. **Launch Application**
   ```bash
   # By default, the app looks for `./best_model.pth`. Set the env var to override this.
   export MODEL_PATH="/path/to/your/best_model.pth"
   
   python -m uvicorn backend.app:app --reload
   ```
   Open `http://localhost:8000` in your web browser.

## Cloud Deployment (Railway)
This repository is fully pre-configured for automated CI/CD deployment on [Railway.app](https://railway.app). 
1. Create a new Railway project deployed from your GitHub fork.
2. In Railway, attach a **Volume** to your service mounted to the path `/app/models`.
3. Manually upload your `.pth` weights file into that volume using the Railway dashboard.
4. Add the Environment Variable `MODEL_PATH=/app/models/best_model.pth` in your Railway settings. 
The Railway builder will map `railway.toml` into your `Dockerfile` and boot the server seamlessly.
