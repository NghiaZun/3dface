import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize

class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None, include_flip=True):
        """
        Args:
            root_dir (str): Đường dẫn đến thư mục gốc chứa các tập con (300W_LP, AFW, HELEN, IBUG, LFPW).
            transform (callable, optional): Transform áp dụng cho ảnh.
            include_flip (bool, optional): Bao gồm các tập Flip (mặc định True).
        """
        self.root_dir = root_dir
        self.transform = transform
        self.include_flip = include_flip
        self.images = []
        self.params = []
        self.landmarks = []
        self.samples = []
        # Danh sách các tập con
        subdatasets = ['AFW', 'HELEN', 'IBUG', 'LFPW']
        
        for subdataset in subdatasets:
            subdir = os.path.join(root_dir, subdataset)
            if os.path.isdir(subdir):
                self._load_data_from_subdir(subdir)
                self.samples.append(os.path.join(data_dir, fname))

            print(f"[DEBUG] Loaded {len(self.samples)} samples from {data_dir}")

            
            # Kiểm tra và tải tập Flip nếu include_flip=True
            if include_flip:
                flip_subdir = os.path.join(root_dir, f"{subdataset}_Flip")
                if os.path.isdir(flip_subdir):
                    self._load_data_from_subdir(flip_subdir)

    def _load_data_from_subdir(self, subdir):
        """Hàm phụ để tải dữ liệu từ một thư mục con."""
        for file in os.listdir(subdir):
            if file.endswith('.jpg'):
                img_path = os.path.join(subdir, file)
                param_path = os.path.join(subdir, file.replace('.jpg', '_param.npy'))
                landmark_path = os.path.join(subdir, file.replace('.jpg', '_landmarks.npy'))
                if os.path.exists(param_path) and os.path.exists(landmark_path):
                    self.images.append(img_path)
                    self.params.append(param_path)
                    self.landmarks.append(landmark_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img = cv2.imread(self.images[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (120, 120))  # Resize to 120x120

        # Load 3DMM parameters and landmarks
        param = np.load(self.params[idx])  # Vector 62 chiều
        landmark = np.load(self.landmarks[idx])  # Ma trận (68, 2) hoặc (68, 3)

        # Apply transform
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(param, dtype=torch.float32), torch.tensor(landmark, dtype=torch.float32)

# Thiết lập transform
transform = Compose([
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Khởi tạo dataset
dataset = FaceDataset(root_dir="facenet/300W_LP", transform=transform, include_flip=True)

# Tạo DataLoader (tối ưu cho GPU)
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=8,
    pin_memory=True
)
