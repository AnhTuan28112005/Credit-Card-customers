 # In file: src/preprocessing.py

import numpy as np
import os

def find_mode(array):
    """Tìm giá trị xuất hiện nhiều nhất (mode) trong một mảng NumPy."""
    values, counts = np.unique(array, return_counts=True)
    return values[np.argmax(counts)]

def clean_and_separate_target(data, target_col_name):
    """Làm sạch và tách biến mục tiêu (y) ra khỏi dữ liệu."""
    # Làm sạch cột mục tiêu (loại bỏ khoảng trắng, dấu ngoặc kép)
    chars_to_strip = ' "'
    cleaned_target_data = np.char.strip(data[target_col_name].astype('U'), chars_to_strip)
    
    # Mã hóa biến mục tiêu: 'Attrited Customer' -> 1, 'Existing Customer' -> 0
    y = np.where(cleaned_target_data == 'Attrited Customer', 1, 0)
    
    # Lấy danh sách tên các cột đặc trưng (loại bỏ cột mục tiêu)
    feature_names = [name for name in data.dtype.names if name != target_col_name]
    
    return data[feature_names], y

def handle_missing_values(data, columns_to_process):
    """Xử lý giá trị 'Unknown' bằng cách thay thế bằng mode."""
    # Tạo một bản sao để không thay đổi dữ liệu gốc
    data_copy = data.copy()
    for col_name in columns_to_process:
        # Làm sạch cột trước khi tìm mode
        cleaned_col = np.char.strip(data_copy[col_name].astype('U'), ' "')
        
        # Tìm mode của cột (loại trừ 'Unknown')
        mode_val = find_mode(cleaned_col[cleaned_col != 'Unknown'])
        print(f" - Cột '{col_name}': Thay thế 'Unknown' bằng mode là '{mode_val}'.")
        
        # Thay thế các giá trị 'Unknown'
        data_copy[col_name][cleaned_col == 'Unknown'] = mode_val
        
    return data_copy

def encode_categorical_features(data, categorical_cols):
    """Mã hóa các cột categorical bằng Label Encoding (2 giá trị) và One-Hot Encoding (>2 giá trị)."""
    encoded_features = []
    
    for col_name in categorical_cols:
        cleaned_col = np.char.strip(data[col_name].astype('U'), ' "')
        unique_vals = np.unique(cleaned_col)
        
        if len(unique_vals) == 2: # Label Encoding
            # Mã hóa giá trị đầu tiên là 0, thứ hai là 1
            encoded_col = np.where(cleaned_col == unique_vals[0], 0, 1).reshape(-1, 1)
            encoded_features.append(encoded_col)
            print(f" - Cột '{col_name}': Mã hóa Label (0/1).")
        
        else: # One-Hot Encoding
            num_samples = len(cleaned_col)
            num_classes = len(unique_vals)
            one_hot_matrix = np.zeros((num_samples, num_classes), dtype=int)
            
            for i, category in enumerate(unique_vals):
                one_hot_matrix[cleaned_col == category, i] = 1
            
            # Bỏ cột đầu tiên để tránh đa cộng tuyến (dummy variable trap)
            encoded_features.append(one_hot_matrix[:, 1:])
            print(f" - Cột '{col_name}': Mã hóa One-Hot với {num_classes-1} cột mới.")
            
    return np.hstack(encoded_features)

def scale_numerical_features(data, numerical_cols):
    """Chuẩn hóa các cột số bằng Z-score Standardization."""
    numerical_data = np.array([data[col] for col in numerical_cols]).T.astype(float)
    
    mean = np.mean(numerical_data, axis=0)
    std = np.std(numerical_data, axis=0)
    
    # Tránh chia cho 0 nếu có cột nào đó có std = 0
    std[std == 0] = 1 
    
    scaled_data = (numerical_data - mean) / std
    return scaled_data

def train_test_split(X, y, test_size=0.2, random_state=42):
    """Tự viết hàm train_test_split bằng NumPy."""
    np.random.seed(random_state)
    n_samples = X.shape[0]
    shuffled_indices = np.random.permutation(n_samples)
    
    test_set_size = int(n_samples * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]





def save_processed_data(X_train, X_test, y_train, y_test, save_path):
    """
    Lưu dữ liệu đã xử lý vào file .npz
    """
    # 1. Lấy đường dẫn thư mục từ đường dẫn file đầy đủ
    # Ví dụ: '../data/processed/bank_churn_processed.npz' -> '../data/processed'
    directory = os.path.dirname(save_path)
    
    # 2. Kiểm tra nếu thư mục chưa tồn tại thì tạo mới
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Đã tạo thư mục mới: {directory}")
        
    # 3. Lưu file
    np.savez_compressed(
        save_path, 
        X_train=X_train, 
        X_test=X_test, 
        y_train=y_train, 
        y_test=y_test
    )
    print(f"Đã lưu dữ liệu thành công vào: {save_path}")



def load_processed_data(file_path):
    """
    Tải dữ liệu đã xử lý từ file nén .npz.
    Hàm này đọc file được tạo bởi save_processed_data.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Kiểm tra file có tồn tại không
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {file_path}")
        
    # Tải file .npz
    # np.load hoạt động giống như mở một từ điển (dictionary)
    data = np.load(file_path)
    
    # Lấy các mảng ra dựa theo tên key chúng ta đã đặt lúc lưu
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    print(f"Đã tải dữ liệu thành công từ: {file_path}")
    return X_train, X_test, y_train, y_test