import torch
import numpy as np
import os
from PIL import Image
from facenet_pytorch import MTCNN
import cv2
import yaml
from TDDFA import TDDFA
from utils.serialization import ser_to_obj

IMG_FOLDER = "DATA/AFLW2000"
NAME_SAVE = "facenet"
PATH_SAVE = os.path.join("output", NAME_SAVE)
os.makedirs(PATH_SAVE, exist_ok=True)
FAIL_FILE = os.path.join(PATH_SAVE, "facenet_failures.txt")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, device=device, keep_all=True)

cfg = yaml.load(open("configs/mb5_120x120.yml"), Loader=yaml.SafeLoader)
tddfa = TDDFA(**cfg)

with open(FAIL_FILE, "w") as fail_file:
    for filename in os.listdir(IMG_FOLDER):
        if filename.endswith(".jpg"):
            img_path = os.path.join(IMG_FOLDER, filename)
            print(f"Processing {img_path}...")

            try:
                img = Image.open(img_path)
                img_cv2 = cv2.imread(img_path)
                img_np = np.array(img)

                boxes, probs, points = mtcnn.detect(img, landmarks=True)
                if boxes is None:
                    print(f"No face detected in {img_path} with Facenet")
                    fail_file.write(f"{img_path}\n")
                    continue

                param_lst, roi_box_lst = tddfa(img_np, boxes)
                ver_3d = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)

                wfp_obj = os.path.join(PATH_SAVE, f"{filename.split('.')[0]}_facenet.obj")
                ser_to_obj(img_cv2, ver_3d, tddfa.tri, height=img_cv2.shape[0], wfp=wfp_obj)

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

print("Facenet processing completed.")
