 # In file: src/models.py

import numpy as np

# ==========================================
# PHẦN 1: CÁC HÀM ĐÁNH GIÁ (METRICS)
# ==========================================

def accuracy_score(y_true, y_pred):
    """Tính tỷ lệ dự đoán đúng."""
    correct = np.sum(y_true == y_pred)
    return correct / len(y_true)

def confusion_matrix(y_true, y_pred):
    """
    Tính ma trận nhầm lẫn thủ công.
    Trả về: TP, TN, FP, FN
    """
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TP, TN, FP, FN

def precision_score(y_true, y_pred):
    """Precision = TP / (TP + FP)"""
    TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
    if (TP + FP) == 0: return 0
    return TP / (TP + FP)

def recall_score(y_true, y_pred):
    """Recall = TP / (TP + FN)"""
    TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
    if (TP + FN) == 0: return 0
    return TP / (TP + FN)

def f1_score(y_true, y_pred):
    """F1 = 2 * (Precision * Recall) / (Precision + Recall)"""
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    if (p + r) == 0: return 0
    return 2 * (p * r) / (p + r)

# ==========================================
# PHẦN 2: MÔ HÌNH LOGISTIC REGRESSION
# ==========================================

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []

    def _sigmoid(self, z):
        """Hàm kích hoạt Sigmoid: 1 / (1 + e^-z)"""
        # np.clip để tránh tràn số (overflow) khi z quá lớn hoặc quá nhỏ
        z = np.clip(z, -250, 250) 
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y = y.flatten() # Đảm bảo y là mảng 1 chiều
        
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.losses = []
        
        # 1. TÍNH TOÁN CLASS WEIGHTS (TỰ ĐỘNG)
        # Đếm số lượng mẫu của từng lớp
        n_class_0 = np.sum(y == 0)
        n_class_1 = np.sum(y == 1)
        
        # Công thức heuristic chuẩn: Total / (2 * Count_Class)
        # Lớp nào ít mẫu hơn sẽ có trọng số lớn hơn
        weight_0 = n_samples / (2 * n_class_0)
        weight_1 = n_samples / (2 * n_class_1)
        
        print(f"-> Đã kích hoạt Class Weighting:")
        print(f"   - Trọng số Class 0 (Ở lại): {weight_0:.4f}")
        print(f"   - Trọng số Class 1 (Rời đi): {weight_1:.4f} (Gấp {weight_1/weight_0:.1f} lần)")

        # Tạo một vector trọng số cho từng mẫu dữ liệu
        # Nếu y[i] == 1 thì sample_weights[i] = weight_1, ngược lại là weight_0
        sample_weights = np.where(y == 1, weight_1, weight_0)

        # 2. VÒNG LẶP HUẤN LUYỆN
        for i in range(self.n_iterations):
            # --- Forward ---
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            # --- Tính Loss (Có trọng số) ---
            epsilon = 1e-15
            # Nhân thêm trọng số vào công thức Cross Entropy
            loss = -1/n_samples * np.sum(
                sample_weights * (y * np.log(y_predicted + epsilon) + (1-y) * np.log(1 - y_predicted + epsilon))
            )
            self.losses.append(loss)

            # --- Backward (Gradient Descent Có trọng số) ---
            # Đạo hàm cũng thay đổi: error * weight
            error = y_predicted - y
            weighted_error = error * sample_weights # <--- ĐIỂM QUAN TRỌNG NHẤT
            
            dw = (1 / n_samples) * np.dot(X.T, weighted_error)
            db = (1 / n_samples) * np.sum(weighted_error)

            # --- Update ---
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if i % 500 == 0:
                print(f"Iter {i}: Loss {loss:.4f}")

    def predict_proba(self, X):
        """Trả về xác suất (0.0 đến 1.0)"""
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        y_predicted_cls = self.predict_proba(X)
        # >>> SỬA Ở ĐÂY: Thêm np.array() bao bên ngoài
        return np.array([1 if i > threshold else 0 for i in y_predicted_cls])
# ==========================================
# PHẦN 3: K-NEAREST NEIGHBORS (KNN) (Bonus)
# ==========================================
# KNN đơn giản, không cần huấn luyện (lazy learning)

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        """KNN chỉ cần lưu dữ liệu train lại."""
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # 1. Tính khoảng cách Euclidean từ điểm x đến tất cả điểm trong X_train
        # Dùng Vectorization của NumPy để tính nhanh
        distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
        
        # 2. Lấy k chỉ số (index) có khoảng cách gần nhất
        k_indices = np.argsort(distances)[:self.k]
        
        # 3. Lấy nhãn của k điểm láng giềng đó
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # 4. Bầu chọn (Majority Vote)
        # Tìm nhãn xuất hiện nhiều nhất
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common
    


   

def k_fold_cross_validation(model_class, X, y, k=5, **model_params):
    """
    Thực hiện K-Fold Cross Validation hoàn toàn bằng NumPy.
    
    Parameters:
    - model_class: Class của mô hình (VD: LogisticRegression)
    - X, y: Dữ liệu (nên là dữ liệu Train gốc trước khi split)
    - k: Số lượng folds (mặc định 5)
    - **model_params: Các tham số khởi tạo cho model (VD: learning_rate=0.1)
    
    Returns:
    - list: Danh sách accuracy của từng fold
    """
    n_samples = len(X)
    fold_size = n_samples // k
    
    # Xáo trộn index để đảm bảo ngẫu nhiên
    indices = np.arange(n_samples)
    np.random.seed(42) # Cố định seed để tái lập kết quả
    np.random.shuffle(indices)
    
    scores = []
    
    print(f"Bắt đầu {k}-Fold Cross Validation...")
    
    for i in range(k):
        # 1. Xác định chỉ số cho Validation và Training
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < k - 1 else n_samples
        
        val_indices = indices[start_idx:end_idx]
        # np.setdiff1d: Lấy các index có trong indices nhưng không có trong val_indices
        train_indices = np.setdiff1d(indices, val_indices)
        
        # 2. Tạo dữ liệu Train/Val cho fold này
        X_train_fold = X[train_indices]
        y_train_fold = y[train_indices]
        X_val_fold = X[val_indices]
        y_val_fold = y[val_indices]
        
        # 3. Khởi tạo và Huấn luyện mô hình mới
        # **model_params dùng để truyền tham số linh hoạt (ví dụ learning_rate)
        model = model_class(**model_params) 
        model.fit(X_train_fold, y_train_fold)
        
        # 4. Dự đoán và Đánh giá
        y_pred = model.predict(X_val_fold)
        score = accuracy_score(y_val_fold, np.array(y_pred))
        scores.append(score)
        
        print(f" - Fold {i+1}/{k}: Accuracy = {score:.4f}")
        
    mean_score = np.mean(scores)
    print(f"==> Trung bình Accuracy: {mean_score:.4f}")
    
    return scores
