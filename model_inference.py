import os, io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224
CLASSES = ['Real / Authentic', 'Forged / Manipulated']

def get_srm_kernels():
    kernels = []
    k1=np.array([[ 0, 0, 0],[ 0, 1,-1],[ 0, 0, 0]],np.float32)/1
    k2=np.array([[ 0, 0, 0],[ 0, 1, 0],[ 0,-1, 0]],np.float32)/1
    k3=np.array([[ 0, 0, 0],[-1, 2,-1],[ 0, 0, 0]],np.float32)/2
    k4=np.array([[ 0,-1, 0],[ 0, 2, 0],[ 0,-1, 0]],np.float32)/2
    k5=np.array([[-1, 2,-1],[ 2,-4, 2],[-1, 2,-1]],np.float32)/4
    k6=np.array([[ 0, 0, 0],[ 0, 1, 0],[0, -1, 0]],np.float32)
    k7=np.array([[ 0, 0, 0],[ 1,-2, 1],[ 0, 0, 0]],np.float32)/2
    k8=np.array([[ 0, 1, 0],[ 0,-2, 0],[ 0, 1, 0]],np.float32)/2
    k9=np.array([[-1, 2,-1],[ 0, 0, 0],[ 1,-2, 1]],np.float32)/4
    k10=np.array([[-1, 0, 1],[ 2, 0,-2],[-1, 0, 1]],np.float32)/4
    def make5(c):
        k=np.zeros((5,5),np.float32); k[2,2]=c
        k[1,2]=k[3,2]=k[2,1]=k[2,3]=-c/4
        k[0,2]=k[4,2]=k[2,0]=k[2,4]=-c/8
        return k/abs(k).sum()
    for c in [1,2,3]: kernels.append(make5(c))
    import math
    for ang in range(0,180,30):
        a=math.radians(ang)
        k=np.array([[ math.cos(a), math.sin(a), 0],
                    [-math.sin(a), math.cos(a), 0],
                    [ 0,          0,            0]],np.float32)
        kernels.append(k[:3,:3]/max(abs(k[:3,:3]).sum(),1e-6))
    base=[k1,k2,k3,k4,k5,k6,k7,k8,k9,k10]
    kernels.extend(base)
    while len(kernels)<30: kernels.append(kernels[len(kernels)%len(base)])
    return kernels[:30]

class SRMLayer(nn.Module):
    def __init__(self):
        super().__init__()
        ks=get_srm_kernels()
        w=np.stack([k.reshape(1,*k.shape) if k.shape==(3,3) else k[:3,:3].reshape(1,3,3) for k in ks])
        self.register_buffer('weight', torch.from_numpy(w))
    def forward(self, x):
        return F.conv2d(x, self.weight, padding=1)

def compute_ela(pil, quality=85, scale=25):
    buf=io.BytesIO(); pil.save(buf,'JPEG',quality=quality); buf.seek(0)
    comp=Image.open(buf).convert('RGB')
    diff=np.abs(np.array(pil,np.float32)-np.array(comp,np.float32))*scale
    return Image.fromarray(np.clip(diff,0,255).astype(np.uint8))

def compute_noise_map_for_viz(pil):
    """Generates a high-pass noise map for the frontend visualization."""
    gray = pil.convert('L')
    arr = np.array(gray).astype(np.float32)
    # Simple Laplacian kernel to stand-in as a noise visualization
    kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=np.float32)
    import scipy.signal
    noise = scipy.signal.convolve2d(arr, kernel, mode='same', boundary='symm')
    noise = np.abs(noise) * 5.0
    return Image.fromarray(np.clip(noise, 0, 255).astype(np.uint8))

class ForensicSRM(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        res=models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.rgb_feat=nn.Sequential(*list(res.children())[:-1])
        
        self.srm=SRMLayer()
        self.srm_cnn=nn.Sequential(
            nn.Conv2d(30,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1),nn.BatchNorm2d(256),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(256,512,3,padding=1),nn.BatchNorm2d(512),nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.ela_cnn=nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1),nn.BatchNorm2d(256),nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        fused=2048+512+256
        self.head=nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused,512),nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(512,64),nn.ReLU(),
            nn.Linear(64,2),
        )
    def forward(self,rgb,gray,ela):
        fa=self.rgb_feat(rgb).flatten(1)
        with torch.no_grad(): noise=self.srm(gray)
        noise=noise.detach()
        fb=self.srm_cnn(noise).flatten(1)
        fc=self.ela_cnn(ela).flatten(1)
        return self.head(torch.cat([fa,fb,fc],1))

# Global model loader
_model_instance = None
to_gray = transforms.Grayscale(num_output_channels=1)
norm_gray = transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)),transforms.ToTensor()])
norm_rgb = transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

def load_model():
    global _model_instance
    if _model_instance is None:
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'forensic_srm_best.pth')
        _model_instance = ForensicSRM(dropout=0.5).to(DEVICE)
        
        # Determine weights_only handling depending on PyTorch version
        try:
            ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
        except TypeError:
            ckpt = torch.load(model_path, map_location=DEVICE)
            
        _model_instance.load_state_dict(ckpt['model_state'])
        _model_instance.eval()
        print("Model loaded successfully into memory.")
    return _model_instance

def predict_single_image(pil_img):
    """
    Runs TTAx5 on the loaded model and returns prediction classification along with confidence.
    """
    model = load_model()
    
    ttas = [
        lambda p: p,
        lambda p: p.transpose(Image.FLIP_LEFT_RIGHT),
        lambda p: p.rotate(10, resample=Image.BICUBIC),
        lambda p: p.rotate(-10, resample=Image.BICUBIC),
        lambda p: ImageEnhance.Brightness(p).enhance(1.1),
    ]
    
    ensemble_logits = 0
    with torch.no_grad():
        for t_fn in ttas:
            pil = t_fn(pil_img)
            
            rgb = norm_rgb(pil).unsqueeze(0).to(DEVICE)
            gray = norm_gray(to_gray(pil)).unsqueeze(0).to(DEVICE)
            ela = compute_ela(pil, quality=85, scale=25)
            ela_t = norm_gray(transforms.Resize((IMG_SIZE,IMG_SIZE))(ela).convert('L')).unsqueeze(0).to(DEVICE)
            
            out = model(rgb, gray, ela_t)
            probs = F.softmax(out, dim=1)
            ensemble_logits += probs
            
        ensemble_logits /= len(ttas)
        confidence, pred_idx = torch.max(ensemble_logits, dim=1)
        
    class_name = CLASSES[pred_idx.item()]
    conf = float(confidence.item()) * 100.0
    
    # 0 = Real, 1 = Forged
    is_forged = (pred_idx.item() == 1)
    
    return is_forged, class_name, float(conf)
