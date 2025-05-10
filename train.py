import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models import mobilenetv2
from datasets import FaceDataset
import pickle
from torchvision.models import mobilenet_v2
from utils.tddfa_util import _parse_param, similar_transform
import numpy as np


class CombinedLoss(nn.Module):
    def __init__(self, eye_weight=5.0):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.eye_weight = eye_weight
        # Chỉ số landmarks vùng mắt (theo 68-point convention)
        self.eye_indices = list(range(36, 48))  # 36-41: mắt trái, 42-47: mắt phải

    def forward(self, pred_params, gt_params, pred_landmarks, gt_landmarks):
        # MSE loss cho tham số 3DMM
        param_loss = self.mse_loss(pred_params, gt_params)
        
        # Landmark loss
        landmark_loss = torch.mean((pred_landmarks - gt_landmarks) ** 2)
        
        # Tăng trọng số cho vùng mắt
        eye_loss = torch.mean((pred_landmarks[:, self.eye_indices] - gt_landmarks[:, self.eye_indices]) ** 2)
        
        return param_loss + landmark_loss + self.eye_weight * eye_loss

transform = Compose([
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Dataset (đã định nghĩa ở trên)
dataset = FaceDataset(data_dir="/kaggle/input/facenet/300W_LP", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Load MobileNet V2
model = mobilenetv2(widen_factor=1.0, num_classes=62, size=120, mode='small')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Load pre-trained weights (ImageNet)
pretrained_model = mobilenet_v2(pretrained=True)
pretrained_dict = pretrained_model.state_dict()
model_dict = model.state_dict()
# Lọc các trọng số không khớp
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

# Định nghĩa hàm loss
criterion = CombinedLoss(eye_weight=5.0)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

# Load TDDFA để dự đoán landmarks từ tham số 3DMM
from TDDFA import TDDFA
cfg = yaml.load(open("configs/mb1_120x120.yml"), Loader=yaml.SafeLoader)
tddfa = TDDFA(**cfg)

# Tinh chỉnh
model.train()
num_epochs = 20
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, gt_params, gt_landmarks in dataloader:
        images, gt_params, gt_landmarks = images.to(device), gt_params.to(device), gt_landmarks.to(device)
        
        # Forward pass
        pred_params = model(images)
        
        # Dự đoán landmarks từ tham số 3DMM
        pred_landmarks = []
        for param in pred_params:
            R, offset, alpha_shp, alpha_exp = _parse_param(param.cpu().numpy())
            pts3d = R @ (tddfa.bfm.u_base + tddfa.bfm.w_shp_base @ alpha_shp + tddfa.bfm.w_exp_base @ alpha_exp). \
                reshape(3, -1, order='F') + offset
            pts3d = similar_transform(pts3d, [0, 0, 120, 120], tddfa.size)  # Giả sử roi_box
            # Chuyển pts3d thành landmarks 2D (chiếu orthographic)
            pred_landmarks.append(pts3d[:2, tddfa.bfm.kpt_ind].T)  # Lấy 68 landmarks
        pred_landmarks = torch.tensor(np.array(pred_landmarks), dtype=torch.float32).to(device)
        
        # Tính loss
        loss = criterion(pred_params, gt_params, pred_landmarks, gt_landmarks)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    scheduler.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.4f}")

# Lưu checkpoint
torch.save(model.state_dict(), "weights/mb2_120x120.pth")

# Cập nhật param_mean_std_62d_120x120.pkl (nếu cần)
all_params = []
for _, params, _ in dataset:
    all_params.append(params.numpy())
all_params = np.array(all_params)
param_mean = all_params.mean(axis=0)
param_std = all_params.std(axis=0)
with open('configs/param_mean_std_62d_120x120.pkl', 'wb') as f:
    pickle.dump({'mean': param_mean, 'std': param_std}, f)