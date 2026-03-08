import re
import os
from PyPDF2 import PdfReader

def pdf_to_text(pdf_path):
    """Đọc toàn bộ nội dung từ file PDF."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def clean_text(text):
    """Loại bỏ các header/footer, số trang, và các chuỗi không cần thiết."""
    text = re.sub(r'about:blank\s*\d+/\d+', '', text)
    text = re.sub(r'\d+/\d+/\d+, \d+:\d+ [AP]M', '', text)
    # Thêm các quy tắc làm sạch khác nếu cần
    return text.strip()

def split_text_by_article(text, output_dir, law_year):
    """Tách văn bản thành các file nhỏ theo từng Điều luật."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    articles = re.split(r'(Điều \d+\.)', text)
    if len(articles) < 2:
        print(f"CẢNH BÁO: Không tìm thấy mẫu 'Điều X.' trong văn bản luật {law_year}. Vui lòng kiểm tra file PDF.")
        return
        
    articles = articles[1:]
    
    for i in range(0, len(articles), 2):
        article_title = articles[i].strip()
        article_content = articles[i+1].strip()
        
        match = re.search(r'\d+', article_title)
        if match:
            article_number = match.group(0)
            file_name = f"dieu_{article_number}_{law_year}.txt"
            file_path = os.path.join(output_dir, file_name)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"{article_title}\n\n{article_content}")
            # Bỏ print ở đây để không làm rối màn hình
            # print(f"Đã tạo file: {file_name}")

def main():
    law_files = {
        2024: 'LuatDatDai2024.pdf',
        2013: 'LuatDatDai2013.pdf'
    }

    for year, pdf_file in law_files.items():
        print(f"--- Bắt đầu xử lý {pdf_file} ---")
        if not os.path.exists(pdf_file):
            print(f"LỖI: Không tìm thấy file {pdf_file}. Bỏ qua...")
            continue

        text = pdf_to_text(pdf_file)
        cleaned_text = clean_text(text)
        
        full_text_filename = f"LuatDatDai{year}_full.txt"
        with open(full_text_filename, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        print(f"Đã tạo file văn bản đầy đủ: {full_text_filename}")
        
    print("\n--- Hoàn thành tiền xử lý! ---")

if __name__ == "__main__":
    main()