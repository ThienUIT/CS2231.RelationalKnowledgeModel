import os
import re
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from llm_callers import LLMClient 

def get_extraction_prompt(law_year, article_code, article_content):
    return f"""
Bạn là một trợ lý pháp lý chuyên nghiệp với nhiệm vụ xây dựng một Đồ thị Tri thức (Knowledge Graph) từ văn bản luật. Với nội dung của Điều luật được cung cấp dưới đây, hãy trích xuất tất cả các thực thể và mối quan hệ giữa chúng theo một cấu trúc JSON định sẵn.

**QUY TẮC TRÍCH XUẤT:**

1.  **Định danh (ID):** Tạo một ID duy nhất cho mỗi thực thể, theo quy tắc: `[loại]_[tên_không_dấu]`. Ví dụ: `chuthe_canhan`, `hanhvi_chuyennhuong`. Với ĐiềuLuật, ID là `dieu_{article_code}`.
2.  **Phân loại Thực thể:** Chỉ sử dụng các loại (labels) sau: `ĐiềuLuật`, `KháiNiệm`, `ChủThể`, `HànhViPhápLý`, `ChếTài`, `ĐiềuKiện`.
3.  **Quan hệ:** Mô tả các mối quan hệ tìm thấy dưới dạng một mảng các object. Mỗi object gồm `source_id`, `relationship_type`, và `target_id`.
4.  **Các loại quan hệ được phép:** Chỉ sử dụng các loại sau: `ĐỊNH_NGHĨA`, `QUY_ĐỊNH_VỀ`, `ÁP_DỤNG_CHO`, `DẪN_ĐẾN`, `YÊU_CẦU`.

---

**VĂN BẢN NGUỒN:**

*   **Tên Luật:** Luật Đất đai {law_year}
*   **Mã Điều:** {article_code}
*   **Nội dung:**
    {article_content}

---

**OUTPUT DƯỚI DẠNG JSON (Không thêm bất kỳ giải thích nào khác):**
"""

def process_single_file(filename, year, input_dir, output_dir):
    """Hàm này xử lý một file duy nhất, được gọi bởi các luồng."""
    output_path = os.path.join(output_dir, filename.replace('.txt', '.json'))
    
    if os.path.exists(output_path):
        return f"Bỏ qua: {filename}"
        
    input_path = os.path.join(input_dir, filename)
    
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    article_code_match = re.search(r'dieu_(\d+)_', filename)
    if not article_code_match:
        return f"Lỗi tên file: {filename}"
    article_code = article_code_match.group(1)

    prompt = get_extraction_prompt(year, article_code, content)
    
    try:
        response_text = LLMClient().generate(prompt)
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
        json_str = json_match.group(1) if json_match else response_text

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json.loads(json_str), f, ensure_ascii=False, indent=2)
        return f"Thành công: {filename}"
            
    except Exception as e:
        return f"Lỗi: {filename} - {e}"

def process_law_year_parallel(year, input_dir, output_dir, max_workers=5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    file_list = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    print(f"\n--- Bắt đầu trích xuất song song cho Luật {year} ({len(file_list)} điều) ---")

    # Sử dụng ThreadPoolExecutor để xử lý song song
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Tạo các task
        futures = {executor.submit(process_single_file, filename, year, input_dir, output_dir): filename for filename in file_list}
        
        # Dùng tqdm để theo dõi tiến trình
        for future in tqdm(as_completed(futures), total=len(file_list), desc=f"Luật {year}"):
            result = future.result()
            # Bạn có thể in kết quả nếu muốn, nhưng nó sẽ làm chậm thanh tiến trình
            print(result)
            pass

def main():
    process_law_year_parallel(2024, 'chunks_2024', 'output_json_2024', max_workers=5)
    process_law_year_parallel(2013, 'chunks_2013', 'output_json_2013', max_workers=5)
    print("\n--- Hoàn thành trích xuất thực thể! ---")

if __name__ == "__main__":
    main()