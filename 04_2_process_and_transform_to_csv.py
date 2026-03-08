import os
import json
from glob import glob
import pandas as pd
from tqdm import tqdm
import re
from unidecode import unidecode

# ==============================================================================
# --- CÁC HÀM CHUẨN HÓA ---
# ==============================================================================
def normalize_string(s: str, case='snake') -> str:
    if not isinstance(s, str) or not s: return ""
    s = unidecode(s.strip())
    s = re.sub(r'[^a-zA-Z0-9]+', '_', s)
    s = s.strip('_')
    if case == 'snake': return s.lower()
    if case == 'pascal': return "".join(word.capitalize() for word in s.split('_') if word)
    if case == 'upper': return s.upper()
    return s

# ==============================================================================
# --- HÀM CHÍNH ---
# ==============================================================================
def main():
    output_dir = 'final_csv'
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    all_nodes_info = {}
    all_relationships_raw = []

    # === BƯỚC 1: ĐỌC VÀ CHUẨN HÓA DỮ LIỆU THÔ TỪ JSON ===
    
    # Xử lý file nội tại (output_json_2013 và output_json_2024)
    json_files_internal = glob('output_json_2013/*.json') + glob('output_json_2024/*.json')
    print(f"Bắt đầu xử lý {len(json_files_internal)} file trích xuất nội tại...")
    
    for file_path in tqdm(json_files_internal, desc="Đang đọc và chuẩn hóa file nội tại"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
            graph_data = data.get("graph", data)
            
            # Đọc nội dung text gốc để gán cho :DieuLuat
            text_dir = os.path.dirname(file_path).replace('output_json', 'chunks')
            text_filename = os.path.basename(file_path).replace('.json', '.txt')
            text_path = os.path.join(text_dir, text_filename)
            full_content = ""
            if os.path.exists(text_path):
                with open(text_path, 'r', encoding='utf-8') as f_text: full_content = f_text.read()

            for entity in graph_data.get('entities', []):
                node_id = entity.get('id')
                if node_id:
                    normalized_id = normalize_string(node_id, 'snake')
                    properties = {normalize_string(k, 'snake'): v for k, v in entity.items() if k not in ['id', 'label']}
                    
                    # Gán nội dung đầy đủ nếu là DieuLuat
                    label = normalize_string(entity.get('label'), 'pascal')
                    if label == 'Dieuluat':
                        properties['noi_dung'] = full_content
                        
                    all_nodes_info[normalized_id] = {'label': label, 'properties': properties}
            
            for rel in graph_data.get('relationships', []):
                if rel.get('source_id') and rel.get('target_id'):
                    all_relationships_raw.append({
                        'source_id': normalize_string(rel['source_id'], 'snake'),
                        'target_id': normalize_string(rel['target_id'], 'snake'),
                        'relationship_type': normalize_string(rel.get('relationship_type'), 'upper')
                    })
        except Exception as e: print(f"\nLỗi khi đọc file nội tại {file_path}: {e}")

    # Xử lý file so sánh
    comparison_files = glob('comparisons_json/*.json')
    print(f"Bắt đầu xử lý {len(comparison_files)} file so sánh...")
    for file_path in tqdm(comparison_files, desc="Đang đọc và chuẩn hóa file so sánh"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
            if data.get('target_id_2013') is not None and data.get('source_id_2024'):
                 all_relationships_raw.append({
                    'source_id': normalize_string(data['source_id_2024'], 'snake'),
                    'target_id': normalize_string(data['target_id_2013'], 'snake'),
                    'relationship_type': 'THAY_THE_CHO',
                    'properties': {'change_type': normalize_string(data.get('type') or data.get('change_type'), 'snake')}
                })
        except Exception as e: print(f"\nLỗi khi đọc file so sánh {file_path}: {e}")

    # === BƯỚC 2: TẠO FILE CSV SẠCH CHO NODES ===
    print("\nĐang tạo file CSV cho các nút...")
    all_possible_node_ids = set(all_nodes_info.keys())
    for rel in all_relationships_raw:
        if rel.get('source_id'): all_possible_node_ids.add(rel.get('source_id'))
        if rel.get('target_id'): all_possible_node_ids.add(rel.get('target_id'))

    final_nodes_list = []
    for node_id in sorted(list(all_possible_node_ids)):
        if not node_id: continue
        
        node_record = {'id:ID': node_id}
        
        if node_id in all_nodes_info:
            node_record[':LABEL'] = all_nodes_info[node_id].get('label', 'Unknown')
            node_record.update(all_nodes_info[node_id].get('properties', {}))
        else: # Tạo node "cụt"
            label = "Unknown"
            if "dieu" in node_id: label = "Dieuluat"
            node_record[':LABEL'] = label
            node_record['name'] = node_id.replace('_', ' ')
        
        final_nodes_list.append(node_record)
        
    nodes_df = pd.DataFrame(final_nodes_list)
    nodes_df.to_csv(os.path.join(output_dir, 'nodes_final.csv'), index=False, encoding='utf-8-sig')

    # === BƯỚC 3: TẠO FILE CSV SẠCH CHO RELATIONSHIPS ===
    print("Đang tạo file CSV cho các quan hệ...")
    final_relationships_list = []
    valid_node_ids_set = set(nodes_df['id:ID'])
    for rel in all_relationships_raw:
        start_id, end_id = rel.get('source_id'), rel.get('target_id')
        
        if start_id in valid_node_ids_set and end_id in valid_node_ids_set and rel.get('relationship_type'):
            rel_record = {
                ':START_ID': start_id,
                ':END_ID': end_id,
                ':TYPE': rel.get('relationship_type')
            }
            properties = rel.get('properties', {})
            if isinstance(properties, dict):
                rel_record.update(properties)
            final_relationships_list.append(rel_record)
            
    relationships_df = pd.DataFrame(final_relationships_list)
    relationships_df.to_csv(os.path.join(output_dir, 'relationships_final.csv'), index=False, encoding='utf-8-sig')
    
    print(f"\n--- HOÀN THÀNH ---")
    print(f"Đã tạo {len(nodes_df)} nút trong file: {os.path.join(output_dir, 'nodes_final.csv')}")
    print(f"Đã tạo {len(relationships_df)} quan hệ trong file: {os.path.join(output_dir, 'relationships_final.csv')}")

if __name__ == "__main__":
    main()