import os
import numpy as np
from scipy.io import loadmat

# Đọc các điểm từ tệp .obj
def read_obj_vertices(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    vertices = []
    for line in lines:
        if line.startswith('v '):  # Tìm các dòng bắt đầu bằng 'v', đại diện cho các vertices
            vertex = list(map(float, line.strip().split()[1:4]))  # Lấy tọa độ (x, y, z), bỏ qua màu sắc và kích thước hộp
            vertices.append(vertex)
    
    return np.array(vertices)

# Đọc các điểm đặc trưng từ tệp .mat
def read_mat_landmarks(file_path):
    mat_data = loadmat(file_path)
    landmarks = mat_data['pt3d_68'].T  # Đọc dữ liệu từ tệp .mat (chú ý .T để chuyển vị trí shape (68, 3))
    
    if landmarks.shape[1] == 3:
        return landmarks
    else:
        raise ValueError(f"Expected landmarks with shape (n, 3), but got {landmarks.shape}")

# Tính bounding box (width, height) từ các điểm đặc trưng
def get_bounding_box_landmarks(landmarks):
    x_min, y_min = np.min(landmarks, axis=0)[:2]
    x_max, y_max = np.max(landmarks, axis=0)[:2]
    width = x_max - x_min  # Chiều rộng
    height = y_max - y_min  # Chiều cao
    return width, height

# Tính NME sử dụng bounding box chuẩn hóa
def calculate_nme_using_bbox(mapped_points, mat_landmarks):
    width, height = get_bounding_box_landmarks(mat_landmarks)
    normalization_factor = max(width, height)  # Dùng chiều rộng hoặc chiều cao làm chuẩn hóa
    
    diff = np.linalg.norm(mapped_points - mat_landmarks, axis=1)
    nme = np.mean(diff) / normalization_factor  # NME chuẩn hóa
    return nme

# So sánh các điểm từ obj và mat
def compare_obj_to_mat(obj_file_path, mat_file_path):
    obj_vertices = read_obj_vertices(obj_file_path)
    mat_landmarks = read_mat_landmarks(mat_file_path)
    
    print(f"Number of vertices in .obj: {obj_vertices.shape[0]}")
    print(f"Number of landmarks in .mat: {mat_landmarks.shape[0]}")
    
    # Ánh xạ các điểm
    mapped_points = []
    for lm in mat_landmarks:
        distances = np.linalg.norm(obj_vertices - lm, axis=1)
        closest_idx = np.argmin(distances)
        mapped_points.append(obj_vertices[closest_idx])
    
    mapped_points = np.array(mapped_points)
    
    # Tính NME chuẩn hóa
    nme = calculate_nme_using_bbox(mapped_points, mat_landmarks)
        
    return nme

# Duyệt qua tất cả các tệp trong thư mục chứa các tệp .obj và .mat
def process_folder(obj_folder, mat_folder):
    nme_list = []
    
    # Lấy danh sách các tệp .obj trong thư mục
    obj_files = [f for f in os.listdir(obj_folder) if f.endswith('.obj')]
    
    for obj_file in obj_files:
        # Tìm tệp .mat tương ứng
        mat_file = obj_file.replace('_dlib.obj', '.mat')
        mat_file_path = os.path.join(mat_folder, mat_file)
        obj_file_path = os.path.join(obj_folder, obj_file)
        
        # Kiểm tra nếu tệp .mat tồn tại
        if os.path.exists(mat_file_path):
            print(f"Processing: {obj_file} and {mat_file}")
            nme = compare_obj_to_mat(obj_file_path, mat_file_path)
            nme_list.append(nme)
        else:
            print(f"Warning: No corresponding .mat file for {obj_file}")
    
    # Tính NME trung bình
    if nme_list:
        avg_nme = np.mean(nme_list)
        print(f"Average NME: {avg_nme:.4f}")
    else:
        print("No valid NME calculated.")

# Ví dụ sử dụng
obj_folder = 'output2/dlib'  # Thư mục chứa các tệp .obj
mat_folder = 'DATA/AFLW2000'  # Thư mục chứa các tệp .mat

process_folder(obj_folder, mat_folder)
