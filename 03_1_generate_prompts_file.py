import os
import re

def get_comparison_prompt_for_chat(article_code_2024, article_content_2024, output_file_name):
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
        **VĂN BẢN THAM CHIẾU (Toàn bộ Luật 2013) đã cung cấp trước đó**
        ---
        **JSON OUTPUT:**
    """

def main():
    input_dir = 'chunks_2024'
    output_dir = 'comparisons_json'
    prompts_file_path = 'all_remaining_comparison_prompts.txt'
    
    file_list = sorted([f for f in os.listdir(input_dir) if f.endswith('.txt')])
    
    prompts_to_generate = []
    for filename in file_list:
        output_filename = filename.replace('.txt', '.json')
        output_path = os.path.join(output_dir, output_filename)
        if not os.path.exists(output_path):
             prompts_to_generate.append((filename, output_filename))

    if not prompts_to_generate:
        print("Tất cả các file đã được xử lý. Không có prompt nào được tạo.")
        return

    print(f"Sẽ tạo prompt cho {len(prompts_to_generate)} điều luật còn lại.")

    with open(prompts_file_path, 'w', encoding='utf-8') as f:
        # Ghi prompt khởi tạo đầu tiên
        f.write("========== PROMPT 1 (NẠP NGỮ CẢNH) ==========\n")
        # Bạn cần có file LuatDatDai2013_full.txt
        with open('LuatDatDai2013_full.txt', 'r', encoding='utf-8') as law_file:
            luat_2013_text = law_file.read()
        f.write(get_initial_prompt(luat_2013_text))
        f.write("\n============================================\n\n")

        # Ghi các prompt so sánh tiếp theo
        for i, (filename, output_filename) in enumerate(prompts_to_generate, 1):
            input_path = os.path.join(input_dir, filename)
            with open(input_path, 'r', encoding='utf-8') as content_file:
                content = content_file.read()
            
            article_code_match = re.search(r'dieu_(\d+)_', filename)
            article_code = article_code_match.group(1) if article_code_match else "unknown"

            f.write(f"========== PROMPT {i+1} (So sánh cho {filename}) ==========\n")
            f.write(get_comparison_prompt_for_chat(article_code, content, output_filename))
            f.write("\n=======================================================\n\n")
            
    print(f"Đã tạo thành công file '{prompts_file_path}' chứa tất cả các prompt cần chạy.")

# Hàm get_initial_prompt cần được copy từ script trên
def get_initial_prompt(luat_2013_content):
    return f"""
Bắt đầu một phiên phân tích luật... (như trên)
"""
if __name__ == "__main__":
    main()