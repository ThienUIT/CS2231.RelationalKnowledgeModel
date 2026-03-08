import os
import json
from glob import glob
from tqdm import tqdm

def merge_json_files(input_glob_pattern: str, output_file_path: str):
    """
    Tìm tất cả các file JSON theo một mẫu, đọc nội dung của chúng,
    và gộp tất cả vào một file JSON duy nhất chứa một mảng các object.
    
    Args:
        input_glob_pattern (str): Mẫu để tìm các file input, ví dụ: 'output_json_2013/*.json'
        output_file_path (str): Đường dẫn đến file JSON output.
    """
    json_files = glob(input_glob_pattern)
    
    if not json_files:
        print(f"Cảnh báo: Không tìm thấy file nào khớp với mẫu '{input_glob_pattern}'. Bỏ qua.")
        return

    print(f"\n--- Bắt đầu gộp {len(json_files)} file từ '{input_glob_pattern}' ---")
    
    merged_data = []
    
    for file_path in tqdm(json_files, desc=f"Đang xử lý {os.path.dirname(input_glob_pattern)}"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                merged_data.append(data)
        except json.JSONDecodeError:
            print(f"\nCảnh báo: File '{file_path}' không chứa JSON hợp lệ, đã bỏ qua.")
        except Exception as e:
            print(f"\nLỗi khi đọc file '{file_path}': {e}")
            
    # Đảm bảo thư mục output tồn tại
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Ghi danh sách đã gộp vào file output
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f_out:
            # indent=2 để file dễ đọc, có thể bỏ đi để file nhỏ hơn
            json.dump(merged_data, f_out, ensure_ascii=False, indent=2)
        print(f"--- Gộp thành công! Đã lưu kết quả vào '{output_file_path}' ---")
    except Exception as e:
        print(f"\nLỗi khi ghi file output '{output_file_path}': {e}")


def main():
    # Cấu hình các thư mục và file output
    tasks = [
        {"input": "output_json_2013/*.json", "output": "analysis/output_2013_merged.json"},
        {"input": "output_json_2024/*.json", "output": "analysis/output_2024_merged.json"},
        # Bạn cũng có thể gộp cả file so sánh nếu muốn
        {"input": "comparisons_json/*.json", "output": "analysis/comparisons_merged.json"}
    ]
    
    for task in tasks:
        merge_json_files(task["input"], task["output"])

if __name__ == "__main__":
    main()