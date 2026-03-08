import os
import json
from glob import glob
from tqdm import tqdm

def validate_internal_extraction_file(file_path):
    """Kiểm tra một file trích xuất nội tại (output_json_...)."""
    issues = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 1. Kiểm tra cấu trúc tổng thể
        graph_data = data.get("graph", data)
        if not isinstance(graph_data, dict):
            issues.append("Cấu trúc JSON gốc không phải là một object.")
            return issues

        # 2. Kiểm tra sự tồn tại của 'entities' và 'relationships'
        if 'entities' not in graph_data:
            issues.append("Thiếu key 'entities' ở cấp cao nhất.")
        if 'relationships' not in graph_data:
            issues.append("Thiếu key 'relationships' ở cấp cao nhất.")

        # 3. Kiểm tra từng entity
        for i, entity in enumerate(graph_data.get('entities', [])):
            if not isinstance(entity, dict):
                issues.append(f"Entity ở vị trí {i} không phải là một object.")
                continue
            if 'id' not in entity or not entity['id']:
                issues.append(f"Entity ở vị trí {i} thiếu hoặc có 'id' rỗng.")
            if 'label' not in entity or not entity['label']:
                issues.append(f"Entity ở vị trí {i} (id: {entity.get('id')}) thiếu 'label'.")

        # 4. Kiểm tra từng relationship
        for i, rel in enumerate(graph_data.get('relationships', [])):
            if not isinstance(rel, dict):
                issues.append(f"Relationship ở vị trí {i} không phải là một object.")
                continue
            if 'source_id' not in rel or not rel['source_id']:
                issues.append(f"Relationship ở vị trí {i} thiếu hoặc có 'source_id' rỗng.")
            if 'target_id' not in rel or not rel['target_id']:
                issues.append(f"Relationship ở vị trí {i} thiếu hoặc có 'target_id' rỗng.")
            if 'relationship_type' not in rel or not rel['relationship_type']:
                issues.append(f"Relationship ở vị trí {i} thiếu 'relationship_type'.")

    except json.JSONDecodeError:
        issues.append("File không chứa JSON hợp lệ.")
    except Exception as e:
        issues.append(f"Lỗi không xác định: {e}")
        
    return issues

def validate_comparison_file(file_path):
    """Kiểm tra một file so sánh (comparisons_json/...)."""
    issues = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, dict):
            issues.append("Cấu trúc JSON không phải là một object.")
            return issues

        # 1. Kiểm tra source_id_2024
        if 'source_id_2024' not in data or not data['source_id_2024']:
            issues.append("Thiếu hoặc rỗng 'source_id_2024'")

        # 2. Kiểm tra sự tồn tại của target_id_2013 (giá trị có thể là null)
        if "target_id_2013" not in data:
            issues.append("Thiếu key 'target_id_2013'")
            
        # 3. Kiểm tra sự tồn tại của 'type' hoặc 'change_type'
        if "type" not in data and "change_type" not in data:
            issues.append("Thiếu cả key 'type' và 'change_type'")

    except json.JSONDecodeError:
        issues.append("File không chứa JSON hợp lệ.")
    except Exception as e:
        issues.append(f"Lỗi không xác định: {e}")
        
    return issues

def main():
    all_problematic_files = []

    # --- Kiểm tra các thư mục output_json ---
    internal_dirs = ['output_json_2013', 'output_json_2024']
    for directory in internal_dirs:
        if not os.path.isdir(directory):
            print(f"Cảnh báo: Thư mục '{directory}' không tồn tại, bỏ qua.")
            continue
        
        files = glob(os.path.join(directory, '*.json'))
        print(f"\n--- Bắt đầu kiểm tra {len(files)} file trong '{directory}' ---")
        for file_path in tqdm(files, desc=f"Kiểm tra {directory}"):
            issues = validate_internal_extraction_file(file_path)
            if issues:
                all_problematic_files.append({"file": file_path, "issues": issues})

    # --- Kiểm tra thư mục comparisons_json ---
    comp_dir = 'comparisons_json'
    if not os.path.isdir(comp_dir):
        print(f"Cảnh báo: Thư mục '{comp_dir}' không tồn tại, bỏ qua.")
    else:
        files = glob(os.path.join(comp_dir, '*.json'))
        print(f"\n--- Bắt đầu kiểm tra {len(files)} file trong '{comp_dir}' ---")
        for file_path in tqdm(files, desc=f"Kiểm tra {comp_dir}"):
            issues = validate_comparison_file(file_path)
            if issues:
                all_problematic_files.append({"file": file_path, "issues": issues})

    # --- In kết quả tổng hợp ---
    if not all_problematic_files:
        print("\n\n✅ Tuyệt vời! Tất cả các file JSON đều hợp lệ về mặt cấu trúc.")
    else:
        print(f"\n\n❌ Phát hiện {len(all_problematic_files)} file có vấn đề:")
        for item in all_problematic_files:
            print(f"\n- File: {item['file']}")
            for issue in item['issues']:
                print(f"  - {issue}")
                
    print("\n--- Hoàn thành kiểm tra ---")

if __name__ == "__main__":
    main()