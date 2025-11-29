# In file: src/exploration.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Đặt style cho biểu đồ đẹp hơn
sns.set_theme(style="whitegrid")

def load_dataset(filepath):
    """
    Tải dữ liệu từ file CSV bằng NumPy.
    Sử dụng structured array để xử lý các kiểu dữ liệu khác nhau.
    """
    # dtype=None -> NumPy tự nhận diện kiểu dữ liệu mỗi cột
    # names=True -> Dùng dòng đầu tiên làm tên cột
    # delimiter=',' -> Dữ liệu cách nhau bằng dấu phẩy
    # encoding='utf-8' -> Đảm bảo đọc được các ký tự
    try:
        data = np.genfromtxt(filepath, dtype=None, names=True, delimiter=',', encoding='utf-8')
        print(f"Tải dữ liệu thành công từ {filepath}")
        return data
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu: {e}")
        return None

def summarize_data(data):
    """In ra thông tin tổng quan của dữ liệu: số dòng, số cột và tên các cột."""
    if data is None:
        return
    
    num_rows = data.shape[0]
    column_names = data.dtype.names
    num_cols = len(column_names)
    
    print("\n--- Thông tin Tổng quan ---")
    print(f"Số dòng: {num_rows}")
    print(f"Số cột: {num_cols}")
    print("Tên các cột:")
    for name in column_names:
        print(f" - {name} (kiểu: {data.dtype[name]})")
    print("---------------------------\n")

def analyze_categorical_column(data, column_name, plot_type='bar'):
    """
    Phân tích một cột dạng categorical.
    
    Parameters:
    - data (np.structuredarray): Bộ dữ liệu.
    - column_name (str): Tên cột cần phân tích.
    - plot_type (str): Loại biểu đồ ('bar' hoặc 'pie'). Mặc định là 'bar'.
    """
    if data is None:
        return
        
    print(f"--- Phân tích cột Categorical: '{column_name}' ---")
    
    # Lấy dữ liệu của cột
    column_data = data[column_name]
    
    # Tìm các giá trị duy nhất và đếm tần suất (logic này dùng chung cho cả 2 biểu đồ)
    unique_values, counts = np.unique(column_data, return_counts=True)
    
    # In báo cáo dạng text
    print("Tần suất và Tỷ lệ:")
    total_count = len(column_data)
    for value, count in zip(unique_values, counts):
        percentage = (count / total_count) * 100
        print(f" - '{value}': {count} lần ({percentage:.2f}%)")
        
    # --- PHẦN VẼ BIỂU ĐỒ ĐÃ ĐƯỢC NÂNG CẤP ---
    plt.figure(figsize=(10, 6))
    
    if plot_type == 'bar':
        sns.countplot(x=column_data, order=unique_values) # order để sắp xếp nhất quán
        plt.title(f"Phân phối của cột '{column_name}' (Biểu đồ cột)")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
    elif plot_type == 'pie':
        # autopct='%1.1f%%' là định dạng để hiển thị % trên biểu đồ
        plt.pie(counts, labels=unique_values, autopct='%1.1f%%', startangle=90, shadow=True)
        plt.title(f"Tỷ lệ của cột '{column_name}' (Biểu đồ tròn)")
        plt.axis('equal')  # Đảm bảo biểu đồ tròn không bị méo
    
    else:
        print(f"Lỗi: Kiểu biểu đồ '{plot_type}' không được hỗ trợ. Vui lòng chọn 'bar' hoặc 'pie'.")
        return

    plt.show()

def analyze_numerical_column(data, column_name):
    """Phân tích một cột dạng số: tính toán thống kê và vẽ histogram."""
    if data is None:
        return
        
    print(f"--- Phân tích cột Numerical: '{column_name}' ---")
    
    # Lấy dữ liệu của cột
    column_data = data[column_name]
    
    # Tính toán các giá trị thống kê
    print(f" - Mean: {np.mean(column_data):.2f}")
    print(f" - Median: {np.median(column_data):.2f}")
    print(f" - Std Dev: {np.std(column_data):.2f}")
    print(f" - Min: {np.min(column_data)}")
    print(f" - Max: {np.max(column_data)}")
    
    # Vẽ biểu đồ histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(column_data, kde=True) # kde=True để vẽ đường cong mật độ
    plt.title(f"Phân phối của cột '{column_name}'")
    plt.xlabel(column_name)
    plt.ylabel("Tần suất")
    plt.show()


def generate_missing_value_report(data):
    """
    Kiểm tra toàn bộ dataset và tạo ra một báo cáo về dữ liệu thiếu.
    - Đối với cột số: Kiểm tra giá trị NaN.
    - Đối với cột chữ: Kiểm tra các chuỗi rỗng ('') hoặc các giá trị đại diện khác.
    """
    if data is None:
        return

    print("--- Báo cáo Dữ liệu thiếu ---")
    found_missing = False
    
    # Danh sách các giá trị được coi là thiếu trong các cột chữ
    string_placeholders = ['Unknown', '', 'NA', 'N/A', '?']

    # Lặp qua tên của từng cột
    for col_name in data.dtype.names:
        column_data = data[col_name]
        dtype = column_data.dtype
        
        missing_count = 0
        
        # 1. Nếu cột là kiểu số (float, int)
        # np.issubdtype giúp kiểm tra kiểu dữ liệu một cách tổng quát
        if np.issubdtype(dtype, np.number):
            # np.isnan chỉ hoạt động trên float, nên ta có thể chuyển đổi an toàn
            missing_count = np.sum(np.isnan(column_data.astype(float)))
        
        # 2. Nếu cột là kiểu chữ (string) hoặc object
        elif np.issubdtype(dtype, np.character) or dtype == 'O':
            # Kiểm tra từng placeholder trong danh sách
            for placeholder in string_placeholders:
                missing_count += np.sum(column_data == placeholder)
        
        # In kết quả nếu tìm thấy giá trị thiếu
        if missing_count > 0:
            total_rows = len(column_data)
            percentage = (missing_count / total_rows) * 100
            print(f" - Cột '{col_name}': {missing_count} giá trị thiếu ({percentage:.2f}%)")
            found_missing = True

    if not found_missing:
        print("Tuyệt vời! Không tìm thấy giá trị thiếu nào trong dataset.")
        
    print("-----------------------------\n")



def analyze_categorical_vs_target(data, feature_col, target_col):
    """
    Phân tích mối quan hệ giữa một biến categorical và biến mục tiêu (categorical).
    Phiên bản này vẽ biểu đồ kép:
    1. Biểu đồ đếm số lượng để cung cấp bối cảnh.
    2. Biểu đồ tỷ lệ rời đi để cung cấp insight.
    """
    print(f"--- Tương quan: '{feature_col}' vs '{target_col}' ---")
    
    target_positive_value = 'Attrited Customer'
    
    # --- BƯỚC 1: CHUẨN HÓA DỮ LIỆU ---
    chars_to_strip = ' "'
    cleaned_feature_col_data = np.char.strip(data[feature_col].astype('U'), chars_to_strip)
    cleaned_target_col_data = np.char.strip(data[target_col].astype('U'), chars_to_strip)
    feature_categories = np.unique(cleaned_feature_col_data)
    
    # --- BƯỚC 2: TÍNH TOÁN TỶ LỆ RỜI ĐI (giữ nguyên) ---
    churn_rates = []
    for category in feature_categories:
        category_mask = (cleaned_feature_col_data == category)
        target_data_in_category = cleaned_target_col_data[category_mask]
        churn_count = np.sum(target_data_in_category == target_positive_value)
        total_in_category = len(target_data_in_category)
        rate = (churn_count / total_in_category) * 100 if total_in_category > 0 else 0
        churn_rates.append(rate)
        print(f" - Nhóm '{category}': {total_in_category} khách hàng, Tỷ lệ rời đi = {rate:.2f}%")

    # --- BƯỚC 3: VẼ BIỂU ĐỒ KÉP (ĐÃ NÂNG CẤP) ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Biểu đồ 1 (bên trái): Phân phối tổng thể (Bối cảnh)
    sns.countplot(x=cleaned_feature_col_data, ax=axes[0], order=feature_categories)
    axes[0].set_title(f"Phân phối tổng thể của '{feature_col}'")
    axes[0].set_xlabel(feature_col)
    axes[0].set_ylabel("Số lượng khách hàng (Count)")
    axes[0].tick_params(axis='x', rotation=45)

    # Biểu đồ 2 (bên phải): Tỷ lệ rời đi (Insight)
    sns.barplot(x=feature_categories, y=churn_rates, ax=axes[1])
    axes[1].set_title(f"Tỷ lệ rời đi theo '{feature_col}'")
    axes[1].set_xlabel(feature_col)
    axes[1].set_ylabel(f"Tỷ lệ {target_positive_value} (%)")
    axes[1].tick_params(axis='x', rotation=45)
    
    # Thêm tiêu đề chung cho cả figure
    fig.suptitle(f"Phân tích '{feature_col}' vs '{target_col}'", fontsize=16)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def analyze_numerical_vs_target(data, feature_col, target_col, plot_type='hist'):
    """
    Phân tích mối quan hệ giữa một biến numerical và biến mục tiêu (categorical).
    Phiên bản này có thể vẽ hist, box, hoặc cả hai ('both').
    """
    print(f"--- Tương quan: '{feature_col}' vs '{target_col}' ---")

    # --- BƯỚC 1: CHUẨN HÓA DỮ LIỆU ---
    feature_data = data[feature_col]
    chars_to_strip = ' "'
    cleaned_target_data = np.char.strip(data[target_col].astype('U'), chars_to_strip)
    
    # --- BƯỚC 2: VẼ BIỂU ĐỒ ---
    if plot_type == 'both':
        # Tạo một figure chứa 2 biểu đồ con (1 hàng, 2 cột)
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        # Biểu đồ 1: Histogram
        sns.histplot(x=feature_data, hue=cleaned_target_data, kde=True, common_norm=False, ax=axes[0])
        axes[0].set_title(f"Phân phối của '{feature_col}'")
        axes[0].set_xlabel(feature_col)
        axes[0].set_ylabel('Tần suất (Count)')

        # Biểu đồ 2: Boxplot
        sns.boxplot(x=cleaned_target_data, y=feature_data, ax=axes[1])
        axes[1].set_title(f"So sánh '{feature_col}'")
        axes[1].set_xlabel(target_col)
        axes[1].set_ylabel(feature_col)
        
        fig.suptitle(f"Phân tích '{feature_col}' theo '{target_col}'", fontsize=16)

    elif plot_type == 'hist':
        plt.figure(figsize=(12, 7))
        sns.histplot(x=feature_data, hue=cleaned_target_data, kde=True, common_norm=False)
        plt.title(f"Phân phối của '{feature_col}' theo '{target_col}'")
        plt.xlabel(feature_col)
        plt.ylabel('Tần suất (Count)')
    
    elif plot_type == 'box':
        plt.figure(figsize=(12, 7))
        sns.boxplot(x=cleaned_target_data, y=feature_data)
        plt.title(f"So sánh '{feature_col}' giữa các nhóm '{target_col}'")
        plt.xlabel(target_col)
        plt.ylabel(feature_col)
        
    else:
        print(f"Lỗi: Kiểu biểu đồ '{plot_type}' không được hỗ trợ.")
        return
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Điều chỉnh layout để title không bị đè
    plt.show()