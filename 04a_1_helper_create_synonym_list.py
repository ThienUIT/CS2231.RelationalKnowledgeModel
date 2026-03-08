import json
import os
import pandas as pd
from collections import defaultdict

# --- Cấu hình ---
INPUT_FILES = [
    'analysis/output_2024_merged.json',
    'analysis/output_2013_merged.json'
]

OUTPUT_CSV_FILE = 'entities_for_review.csv'

def extract_and_review_entities(input_files, output_filename):
    """
    Trích xuất tất cả các thực thể duy nhất từ nhiều file JSON,
    sắp xếp chúng và xuất ra file CSV để rà soát.
    """
    unique_entities = {}

    print("Bắt đầu đọc và tổng hợp thực thể từ các file nguồn...")
    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"Cảnh báo: Bỏ qua file không tồn tại: {file_path}")
            continue

        print(f"Đang xử lý file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for article_data in data:
            for entity in article_data.get('entities', []):
                entity_id = entity.get('id')
                if entity_id and entity_id not in unique_entities:
                    unique_entities[entity_id] = entity
    
    if not unique_entities:
        print("Lỗi: Không tìm thấy thực thể nào. Vui lòng kiểm tra lại file input.")
        return

    print(f"Đã tìm thấy tổng cộng {len(unique_entities)} thực thể duy nhất (unique IDs).")
    
    # --- Chuẩn bị dữ liệu để rà soát ---
    review_list = []
    for entity_id, entity_obj in unique_entities.items():
        properties = entity_obj.get('properties', {})
        # Làm sạch tên và cung cấp giá trị mặc định nếu thiếu
        name = str(properties.get('name', 'NO_NAME_FOUND')).strip().strip('.')
        label = entity_obj.get('label', 'NO_LABEL')
        
        review_list.append({
            'id': entity_id,
            'label': label,
            'name': name
        })

    # --- Sắp xếp để dễ rà soát ---
    # Sắp xếp theo tên (name) là ưu tiên chính, sau đó đến label
    # Điều này sẽ nhóm các thực thể có tên giống nhau lại gần nhau
    print("Đang sắp xếp danh sách thực thể...")
    sorted_list = sorted(review_list, key=lambda x: (x['name'].lower(), x['label']))

    # --- Xuất ra file CSV ---
    print(f"Đang xuất kết quả ra file '{output_filename}'...")
    df = pd.DataFrame(sorted_list)
    df.to_csv(output_filename, index=False, encoding='utf-8-sig')

    print("\n--- Hoàn thành! ---")
    print(f"Mở file '{output_filename}' bằng Excel hoặc Google Sheets để bắt đầu rà soát.")

# --- Main Execution ---
if __name__ == "__main__":
    extract_and_review_entities(INPUT_FILES, OUTPUT_CSV_FILE)