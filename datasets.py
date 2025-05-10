import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize

class FaceDataset(Dataset):
    def __init__(self, data_dir, transform=None, include_flip=True):
        """
        Args:
            root_dir (str): Đường dẫn đến thư mục gốc chứa các tập con (AFW, HELEN, IBUG, LFPW).
            transform (callable, optional): Transform áp dụng cho ảnh.
            include_flip (bool, optional): Bao gồm các tập Flip (mặc định True).
        """
        self.root_dir = root_dir
        self.transform = transform
        self.include_flip = include_flip
        self.images = []
        self.params = []
        self.landmarks = []

        # Danh sách các tập con cần load
        subdatasets = ['AFW', 'HELEN', 'IBUG', 'LFPW']

        for subdataset in subdatasets:
            subdir = os.path.join(root_dir, subdataset)
            if os.path.isdir(subdir):
                self._load_data_from_subdir(subdir)
            else:
                print(f"[WARNING] Subdir not found: {subdir}")
            
            if include_flip:
                flip_subdir = os.path.join(root_dir, f"{subdataset}_Flip")
                if os.path.isdir(flip_subdir):
                    self._load_data_from_subdir(flip_subdir)
                else:
                    print(f"[INFO] Flip subdir not found: {flip_subdir}")

        print(f"[INFO] Loaded {len(self.images)} samples from {root_dir}")

        # Nếu không có dữ liệu, raise lỗi rõ ràng
        if len(self.images) == 0:
            raise ValueError(f"No valid samples found in {root_dir}. Please check the folder structure and content.")

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
                else:
                    print(f"[WARNING] Missing param or landmark for {img_path}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img = cv2.imread(self.images[idx])
        if img is None:
            raise ValueError(f"Failed to load image: {self.images[idx]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (120, 120))

        # Load 3DMM parameters and landmarks
        param = np.load(self.params[idx])  # (62,)
        landmark = np.load(self.landmarks[idx])  # (68, 2)

        # Apply transform
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(param, dtype=torch.float32), torch.tensor(landmark, dtype=torch.float32)

