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
    """Loại bỏ các header/footer không cần thiết."""
    text = re.sub(r'about:blank\s*\d+/\d+', '', text)
    text = re.sub(r'\d+/\d+/\d+, \d+:\d+ [AP]M', '', text)
    return text.strip()

def split_text_by_article(text, output_dir, law_year):
    """Tách văn bản thành các file nhỏ theo từng Điều luật."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Regex để tìm "Điều X." hoặc "Điều XY." hoặc "Điều XYZ."
    articles = re.split(r'(Điều \d+\.)', text)
    
    # Bỏ phần tử đầu tiên (thường là phần giới thiệu trước Điều 1)
    articles = articles[1:]
    
    for i in range(0, len(articles), 2):
        article_title = articles[i].strip()
        article_content = articles[i+1].strip()
        
        # Lấy số hiệu điều luật từ tiêu đề
        match = re.search(r'\d+', article_title)
        if match:
            article_number = match.group(0)
            file_name = f"dieu_{article_number}_{law_year}.txt"
            file_path = os.path.join(output_dir, file_name)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"{article_title}\n\n{article_content}")
            print(f"Đã tạo file: {file_name}")

def main():
    # Xử lý Luật Đất đai 2024
    print("--- Bắt đầu xử lý Luật Đất đai 2024 ---")
    text_2024 = pdf_to_text('LuatDatDai2024.pdf')
    cleaned_text_2024 = clean_text(text_2024)
    split_text_by_article(cleaned_text_2024, 'chunks_2024', 2024)
    
    # Xử lý Luật Đất đai 2013
    print("\n--- Bắt đầu xử lý Luật Đất đai 2013 ---")
    text_2013 = pdf_to_text('LuatDatDai2013.pdf')
    cleaned_text_2013 = clean_text(text_2013)
    split_text_by_article(cleaned_text_2013, 'chunks_2013', 2013)
    print("\n--- Hoàn thành tiền xử lý! ---")

if __name__ == "__main__":
    main()