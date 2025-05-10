import scipy.io

# Đọc dữ liệu từ file .mat
mat_file_path = 'FITUS-PatternRecognition-FinalProject/DATA/AFLW2000/image00002.mat'
mat_data = scipy.io.loadmat(mat_file_path)

# In ra các keys có trong file .mat để kiểm tra cấu trúc
print("Cấu trúc dữ liệu trong file .mat:")
for key in mat_data.keys():
    print(key)

print(mat_data['pt3d_68'])
