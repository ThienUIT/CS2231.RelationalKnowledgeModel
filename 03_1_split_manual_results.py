import json
import os

def split_json_array_to_files(input_file_path, output_dir):
    """
    Đọc một file JSON chứa một danh sách các object,
    sau đó tách mỗi object thành một file JSON riêng biệt.
    Tên file output được đặt theo giá trị của key 'source_id_2024'.
    """
    
    # 1. Đảm bảo thư mục output tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Đã tạo thư mục: '{output_dir}'")
        
    # 2. Kiểm tra file input có tồn tại không
    if not os.path.exists(input_file_path):
        print(f"Lỗi: Không tìm thấy file input '{input_file_path}'")
        return

    # 3. Đọc và xử lý file JSON input
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data_array = json.load(f)
        
        # Kiểm tra xem file có phải là một danh sách không
        if not isinstance(data_array, list):
            print("Lỗi: Nội dung file input không phải là một danh sách (JSON array).")
            return
            
        print(f"Tìm thấy {len(data_array)} object trong file input. Bắt đầu quá trình tách file...")
        
        success_count = 0
        error_count = 0
        
        # 4. Lặp qua từng object và tạo file riêng
        for item in data_array:
            # Kiểm tra xem object có key 'source_id_2024' không
            if 'source_id_2024' in item:
                # Lấy tên file từ source_id
                file_name = f"{item['source_id_2024']}.json"
                output_file_path = os.path.join(output_dir, file_name)
                
                # Ghi object hiện tại vào file mới
                try:
                    with open(output_file_path, 'w', encoding='utf-8') as out_f:
                        json.dump(item, out_f, ensure_ascii=False, indent=2)
                    # print(f"Đã tạo file: {output_file_path}") # Bỏ comment nếu muốn xem chi tiết
                    success_count += 1
                except Exception as e:
                    print(f"Lỗi khi ghi file '{output_file_path}': {e}")
                    error_count += 1
            else:
                print(f"Cảnh báo: Bỏ qua một object vì thiếu key 'source_id_2024': {item}")
                error_count += 1
                
        print("\n--- Hoàn thành! ---")
        print(f"Tổng số file đã tạo thành công: {success_count}")
        if error_count > 0:
            print(f"Số object bị lỗi hoặc bỏ qua: {error_count}")

    except json.JSONDecodeError:
        print(f"Lỗi: File '{input_file_path}' không chứa nội dung JSON hợp lệ.")
    except Exception as e:
        print(f"Đã xảy ra lỗi không xác định: {e}")

# --- Cấu hình và Chạy ---
if __name__ == '__main__':
    # Đặt tên file input mà bạn đã lưu kết quả manual
    INPUT_JSON_FILE = 'all_remaining_comparision_output.json' 
    
    # Đặt tên thư mục output
    OUTPUT_DIRECTORY = 'comparisons_json_temp'
    
    split_json_array_to_files(INPUT_JSON_FILE, OUTPUT_DIRECTORY)