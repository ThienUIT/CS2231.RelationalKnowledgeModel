import os
import re
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from llm_callers import LLMClient

# --- Chuẩn bị dữ liệu ---
# Đọc toàn bộ nội dung luật cũ vào một biến. 
# Đảm bảo đã chạy script 01 và có file này.
try:
    with open('LuatDatDai2013_full.txt', 'r', encoding='utf-8') as f:
        luat_2013_full_text = f.read()
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file 'LuatDatDai2013_full.txt'.")
    print("Vui lòng chạy script 01 để tạo file này hoặc đảm bảo nó nằm trong cùng thư mục.")
    exit()

# --- Định nghĩa Prompt ---
def get_comparison_prompt(article_code_2024, article_content_2024, luat_2013_full_text):
    """Prompt 2: So sánh và liên kết."""
    return f"""
        Bạn là một chuyên gia pháp lý cao cấp, chuyên phân tích và so sánh các phiên bản luật.

        **NHIỆM VỤ:**
        Với "Điều luật nguồn" từ Luật Đất đai 2024, hãy tìm ra Điều luật tương ứng nhất trong "Văn bản tham chiếu" (Luật Đất đai 2013) và xác định bản chất của sự thay đổi.

        **QUY TẮC:**
        1.  **Tìm kiếm:** Đọc kỹ "Điều luật nguồn" và tìm trong "Văn bản tham chiếu" điều luật có nội dung, chủ đề tương đồng nhất.
        2.  **Xác định ID:** ID của điều luật 2024 là `dieu_{article_code_2024}_2024`. ID của điều luật 2013 tương ứng là `dieu_[số hiệu]_2013`.
        3.  **Xác định loại thay đổi:** Phân loại vào một trong các loại sau: `SỬA_ĐỔI_BỔ_SUNG`, `THAY_THẾ_HOÀN_TOÀN`, `GIỮ_NGUYÊN`, `ĐIỀU_LUẬT_MỚI`.
        4.  **Output:** Trả về kết quả dưới dạng một object JSON duy nhất. Nếu không tìm thấy, `target_id_2013` sẽ là `null`.

        ---
        **ĐIỀU LUẬT NGUỒN (Luật 2024):**
        *   **Mã Điều:** {article_code_2024}
        *   **Nội dung:** {article_content_2024}
        ---
        **VĂN BẢN THAM CHIẾU (Toàn bộ Luật 2013):**
        {luat_2013_full_text}
        ---
        **JSON OUTPUT:**
    """

# --- Hàm xử lý cho một file ---
def process_single_comparison(filename, input_dir, output_dir, luat_2013_full_text):
    """Hàm này xử lý so sánh cho một file duy nhất, được gọi bởi các luồng."""
    output_path = os.path.join(output_dir, filename.replace('.txt', '.json'))
    
    # === BỎ QUA NẾU ĐÃ XỬ LÝ ===
    if os.path.exists(output_path):
        print(f'{filename} đã được xử lý. Bỏ qua')
        return f"Bỏ qua (đã tồn tại): {filename}"
        
    input_path = os.path.join(input_dir, filename)
    
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    article_code_match = re.search(r'dieu_(\d+)_', filename)
    if not article_code_match:
        return f"Lỗi tên file: {filename}"
    article_code = article_code_match.group(1)

    prompt = get_comparison_prompt(article_code, content, luat_2013_full_text)
    
    try:
        response_text = LLMClient().generate(prompt)
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
        json_str = json_match.group(1) if json_match else response_text
        
        # Kiểm tra nội dung JSON hợp lệ trước khi lưu
        parsed_json = json.loads(json_str)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(parsed_json, f, ensure_ascii=False, indent=2)
        return f"Thành công: {filename}"
            
    except Exception as e:
        # Khi có lỗi (rate limit, JSON không hợp lệ...), hàm sẽ trả về thông báo lỗi
        # File output sẽ không được tạo, lần chạy sau sẽ tự động thử lại file này
        return f"Lỗi: {filename} - {e}"

# --- Hàm chính ---
def main(max_workers=3):
    input_dir = 'chunks_2024'
    output_dir = 'comparisons_json'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    file_list = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    
    print(f"\n--- Bắt đầu so sánh song song Luật 2024 vs 2013 ({len(file_list)} điều) ---")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_comparison, filename, input_dir, output_dir, luat_2013_full_text): filename for filename in file_list}
        
        for future in tqdm(as_completed(futures), total=len(file_list), desc="So sánh luật"):
            result = future.result()
            if "Lỗi" in result:
                # In ra lỗi để tiện theo dõi
                print(result)

    print("\n--- Hoàn thành trích xuất so sánh! ---")

if __name__ == "__main__":
    main(max_workers=10)