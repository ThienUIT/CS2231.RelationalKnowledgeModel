import os
import json
from glob import glob
import pandas as pd
from tqdm import tqdm
import re
from unidecode import unidecode

def normalize_string_id(s: str) -> str:
    if not isinstance(s, str): return ""
    s = unidecode(s.strip())
    s = re.sub(r'[^a-zA-Z0-9]+', '_', s)
    return s.lower().strip('_')

def main():
    output_dir = 'final_csv'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    law_contents_for_vectordb = []
    
    # Chỉ đọc từ các file JSON gốc
    json_dirs = ['output_json_2013', 'output_json_2024']
    
    for json_dir in json_dirs:
        json_files = glob(os.path.join(json_dir, '*.json'))
        text_dir = json_dir.replace('output_json', 'chunks')
        
        print(f"\nĐang xử lý {len(json_files)} file từ '{json_dir}'...")

        for file_path in tqdm(json_files, desc=f"Xử lý {json_dir}"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                graph_data = data.get("graph", data)
                
                # Tìm nút :DieuLuat chính trong file
                main_law_node = None
                for entity in graph_data.get('entities', []):
                    if 'DieuLuat' in entity.get('label', '') or 'Dieuluat' in entity.get('label', ''):
                        main_law_node = entity
                        break # Chỉ lấy nút đầu tiên tìm thấy

                # Tìm file text tương ứng
                text_filename = os.path.basename(file_path).replace('.json', '.txt')
                text_path = os.path.join(text_dir, text_filename)

                if main_law_node and main_law_node.get('id') and os.path.exists(text_path):
                    with open(text_path, 'r', encoding='utf-8') as f_text:
                        full_content = f_text.read()
                    
                    if full_content.strip(): # Chỉ thêm nếu nội dung không rỗng
                        law_contents_for_vectordb.append({
                            'id': normalize_string_id(main_law_node['id']),
                            'content': full_content
                        })

            except Exception as e:
                print(f"\nLỗi khi xử lý file {file_path}: {e}")

    # Tạo DataFrame và lưu file CSV
    if not law_contents_for_vectordb:
        print("\nLỗi: Không tạo được dữ liệu nào cho Vector DB.")
        return

    vectordb_df = pd.DataFrame(law_contents_for_vectordb)
    output_path = os.path.join(output_dir, 'data_for_vectordb.csv')
    vectordb_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\n--- Hoàn thành! ---")
    print(f"Đã tạo {len(vectordb_df)} bản ghi trong '{output_path}'")

if __name__ == "__main__":
    main()