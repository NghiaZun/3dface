import torch
import numpy as np
import os
from PIL import Image
from facenet_pytorch import MTCNN
import dlib
import cv2
import yaml
from TDDFA import TDDFA
from utils.render import render
from utils.uv import uv_tex
from utils.serialization import ser_to_obj
from utils.functions import draw_landmarks

# Path to the folder containing images
IMG_FOLDER = "DATA/AFLW2000"
NAME_SAVE = ""
PATH_SAVE = os.path.join("output", str(NAME_SAVE))
os.makedirs(PATH_SAVE, exist_ok=True)

# Files to save images that cannot be detected
FACENET_FAIL_FILE = os.path.join(PATH_SAVE, "facenet_failures.txt")
DLIB_FAIL_FILE = os.path.join(PATH_SAVE, "dlib_failures.txt")

# Determine if an NVIDIA GPU is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Initialize Facenet
mtcnn = MTCNN(
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device,
    keep_all=True
)

# Initialize Dlib
dlib_detector = dlib.get_frontal_face_detector()

# Load TDDFA configuration
cfg = yaml.load(open("configs/mb1_120x120.yml"), Loader=yaml.SafeLoader)
tddfa = TDDFA(**cfg)

# Open files to record failures
with open(FACENET_FAIL_FILE, "w") as facenet_fail, open(DLIB_FAIL_FILE, "w") as dlib_fail:
    # Process all JPG files in the folder
    for filename in os.listdir(IMG_FOLDER):
        if filename.endswith(".jpg"):
            img_path = os.path.join(IMG_FOLDER, filename)
            print(f"Processing {img_path}...")

            try:
                # Load image
                img = Image.open(img_path)
                img2 = cv2.imread(img_path)
                img_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
                img_np = np.array(img)

                # Step 1: Get boxes and probability
                # Facenet
                check = True
                fn_boxes, fn_probs, fn_points = mtcnn.detect(img, landmarks=True)
                if fn_boxes is None:
                    print(f"No face detected in {img_path} with Facenet")
                    facenet_fail.write(f"{img_path}\n")
                    check = False

                # Dlib
                dlib_faces = dlib_detector(img_gray, 0)
                dlib_boxes = []
                for face in dlib_faces:
                    x1 = face.left()
                    y1 = face.top()
                    x2 = face.right()
                    y2 = face.bottom()
                    dlib_boxes.append([x1, y1, x2, y2])

                if len(dlib_boxes) == 0:
                    print(f"No face detected in {img_path} with Dlib")
                    dlib_fail.write(f"{img_path}\n")
                    check = False
                    
                if not check:
                    continue

                # Step 2: TDDFA
                fn_param_lst, fn_roi_box_lst = tddfa(img_np, fn_boxes)
                dlib_param_lst, dlib_roi_box_lst = tddfa(img2, dlib_boxes)

                # Landmark visualization
                # ver_landmark = tddfa.recon_vers(fn_param_lst, fn_roi_box_lst, dense_flag=False)
                # path_imglandmark = os.path.join(PATH_SAVE, f"{filename}_facenet_landmarks.jpg")
                # draw_landmarks(img2, ver_landmark, show_flag=False, dense_flag=False, wfp=path_imglandmark)

                # ver_landmark = tddfa.recon_vers(dlib_param_lst, dlib_roi_box_lst, dense_flag=False)
                # path_imglandmark = os.path.join(PATH_SAVE, f"{filename}_dlib_landmarks.jpg")
                # draw_landmarks(img2, ver_landmark, show_flag=False, dense_flag=False, wfp=path_imglandmark)

                # Dense face reconstruction
                fn_ver_3d = tddfa.recon_vers(fn_param_lst, fn_roi_box_lst, dense_flag=True)
                dlib_ver_3d = tddfa.recon_vers(dlib_param_lst, dlib_roi_box_lst, dense_flag=True)

                # Obj
                fn_wfp_obj = os.path.join(PATH_SAVE, f"{filename.split('.')[0]}_facenet.obj")
                ser_to_obj(img2, fn_ver_3d, tddfa.tri, height=img2.shape[0], wfp=fn_wfp_obj)

                dlib_wfp_obj = os.path.join(PATH_SAVE, f"{filename.split('.')[0]}_dlib.obj")
                ser_to_obj(img2, dlib_ver_3d, tddfa.tri, height=img2.shape[0], wfp=dlib_wfp_obj)

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

print("Processing completed.")