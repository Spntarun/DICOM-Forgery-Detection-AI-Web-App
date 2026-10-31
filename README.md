# DICOM Forgery Detection AI Web App

This standalone Flask application detects splicing and copy-move manipulate patterns in Medical CT/DICOM images (converted to standard formats like PNG/JPEG). 
It relies on a cutting-edge Steganalysis Rich Model (SRM) combined with Error Level Analysis (ELA) and a ResNet50 backbone to capture minute metadata and noise forensic footprint inconsistencies.

## Setup Instructions

This app can be smoothly run entirely on local CPUs/GPUs since the AI model inference uses PyTorch conditionally initialized on `cuda` or `cpu`.

### 1. Requirements

Make sure you have Python 3.9+ installed.

1. Open your terminal or command prompt inside this (`webapp`) folder.
2. Ensure you have the model weights file (`forensic_srm_best.pth`) located in the `/models` directory.
3. Install the dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```
*(Note: If you run into PyTorch versioning issues depending on whether your GPU supports CUDA, please install the appropriate torch/torchvision libraries from [pytorch.org](https://pytorch.org/))*

### 2. Run the server

```bash
python app.py
```

### 3. Usage

1. Open your browser and navigate to the address shown in your server log (typically `http://127.0.0.1:5000`).
2. Register a new user account.
3. Log in.
4. Upload any suspected CT/Medical Image.
5. The backend will perform test-time augmentation inference, extract the ELA compression heatmaps, calculate the spatial SRM noise residuals, and provide the forgery verdict.
