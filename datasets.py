import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize

class FaceDataset(Dataset):
    def __init__(self, root_dir, landmark_root, transform=None, include_flip=True):
        self.root_dir = root_dir
        self.landmark_root = landmark_root
        self.transform = transform
        self.include_flip = include_flip
        self.images = []
        self.params = []
        self.landmarks = []

        subdatasets = ['AFW', 'HELEN', 'IBUG', 'LFPW']
        for subdataset in subdatasets:
            subdir = os.path.join(root_dir, subdataset)
            landmark_subdir = os.path.join(landmark_root, subdataset)
            if os.path.isdir(subdir) and os.path.isdir(landmark_subdir):
                self._load_data_from_subdir(subdir, landmark_subdir)

            if include_flip:
                flip_subdir = os.path.join(root_dir, f"{subdataset}_Flip")
                flip_landmark_subdir = os.path.join(landmark_root, f"{subdataset}_Flip")
                if os.path.isdir(flip_subdir) and os.path.isdir(flip_landmark_subdir):
                    self._load_data_from_subdir(flip_subdir, flip_landmark_subdir)

            # Nếu không có dữ liệu, raise lỗi rõ ràng
            if len(self.images) == 0:
                raise ValueError(f"No valid samples found in {data_dir}. Please check the folder structure and content.")

    def _load_data_from_subdir(self, image_subdir, landmark_subdir):
        for file in os.listdir(image_subdir):
            if file.endswith('.jpg'):
                img_path = os.path.join(image_subdir, file)
                param_path = os.path.join(image_subdir, file.replace('.jpg', '.mat'))
                landmark_path = os.path.join(landmark_subdir, file.replace('.jpg', '_pts.mat'))

                if os.path.exists(param_path) and os.path.exists(landmark_path):
                    self.images.append(img_path)
                    self.params.append(param_path)
                    self.landmarks.append(landmark_path)
                else:
                    print(f"[WARNING] Missing param or landmark for {file}")


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
        param = np.load(self.params[idx], allow_pickle = True)  # (62,)
        landmark = np.load(self.landmarks[idx])  # (68, 2)

        # Apply transform
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(param, dtype=torch.float32), torch.tensor(landmark, dtype=torch.float32)

