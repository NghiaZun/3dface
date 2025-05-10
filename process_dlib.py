import numpy as np
import os
import dlib
import cv2
import yaml
from TDDFA import TDDFA
from utils.serialization import ser_to_obj

IMG_FOLDER = "DATA/AFLW2000"
NAME_SAVE = "dlib"
PATH_SAVE = os.path.join("output2", NAME_SAVE)
os.makedirs(PATH_SAVE, exist_ok=True)
FAIL_FILE = os.path.join(PATH_SAVE, "dlib_failures.txt")

dlib_detector = dlib.get_frontal_face_detector()

cfg = yaml.load(open("configs/mb5_120x120.yml"), Loader=yaml.SafeLoader)
tddfa = TDDFA(**cfg)

with open(FAIL_FILE, "w") as fail_file:
    for filename in os.listdir(IMG_FOLDER):
        if filename.endswith(".jpg"):
            img_path = os.path.join(IMG_FOLDER, filename)
            print(f"Processing {img_path}...")

            try:
                img = cv2.imread(img_path)
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

                dlib_faces = dlib_detector(img_gray, 0)
                boxes = [[face.left(), face.top(), face.right(), face.bottom()] for face in dlib_faces]

                if len(boxes) == 0:
                    print(f"No face detected in {img_path} with Dlib")
                    fail_file.write(f"{img_path}\n")
                    continue

                param_lst, roi_box_lst = tddfa(img, boxes)
                ver_3d = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)

                wfp_obj = os.path.join(PATH_SAVE, f"{filename.split('.')[0]}_dlib.obj")
                ser_to_obj(img, ver_3d, tddfa.tri, height=img.shape[0], wfp=wfp_obj)

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

print("Dlib processing completed.")
