import json
import pandas as pd
import os

# --- Cấu hình ---
INPUT_JSON_FILE = 'analysis/comparisons_merged.json' 
OUTPUT_CSV_FILE = 'result_final/graph_edges_comparison.csv'

def create_comparison_edges(input_file, output_file):
    """
    Đọc file JSON chứa dữ liệu so sánh đã hợp nhất,
    chuyển đổi nó thành file CSV chứa các cạnh (edges) theo chuẩn Neo4j.
    """
    if not os.path.exists(input_file):
        print(f"Lỗi: Không tìm thấy file đầu vào '{input_file}'.")
        return

    print(f"Bắt đầu xử lý file so sánh: '{input_file}'...")

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            comparisons = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Lỗi: File JSON không hợp lệ. Vui lòng kiểm tra lại. Lỗi: {e}")
        return
    except Exception as e:
        print(f"Lỗi không xác định khi đọc file: {e}")
        return

    edge_list = []
    new_articles_count = 0
    
    for item in comparisons:
        source_id = item.get('source_id_2024')
        target_id = item.get('target_id_2013')
        rel_type = item.get('change_type')

        # --- Logic Lọc Quan Trọng ---
        # Chỉ tạo cạnh nếu có cả điểm đầu và điểm cuối.
        # Bỏ qua các điều luật mới vì chúng không có liên kết ngược.
        if source_id and target_id and rel_type:
            # Chuẩn hóa tên loại quan hệ: viết hoa và dùng gạch dưới
            # Ví dụ: "sua_doi_bo_sung" -> "SUA_DOI_BO_SUNG"
            standardized_rel_type = rel_type.upper().replace(" ", "_")

            edge_list.append({
                ':START_ID': source_id,
                ':END_ID': target_id,
                ':TYPE': standardized_rel_type
            })
        elif rel_type == 'dieu_luat_moi':
            new_articles_count += 1

    if not edge_list:
        print("Cảnh báo: Không tạo ra được mối quan hệ nào. File CSV đầu ra sẽ trống.")
        # Vẫn tạo file trống với header để đảm bảo quy trình không bị lỗi
        df_edges = pd.DataFrame(columns=[':START_ID', ':END_ID', ':TYPE'])
    else:
        df_edges = pd.DataFrame(edge_list)
        # Sắp xếp lại cột để đảm bảo đúng thứ tự cho Neo4j
        df_edges = df_edges[[':START_ID', ':END_ID', ':TYPE']]

    # Xuất ra file CSV
    df_edges.to_csv(output_file, index=False, encoding='utf-8')

    print("\n--- Hoàn thành! ---")
    print(f"Đã xử lý {len(comparisons)} mục so sánh.")
    print(f"Đã tạo thành công file '{output_file}' với {len(df_edges)} mối quan hệ liên kết.")
    print(f"Đã bỏ qua {new_articles_count} điều luật mới (không có liên kết).")


# --- Main Execution ---
if __name__ == "__main__":
    create_comparison_edges(INPUT_JSON_FILE, OUTPUT_CSV_FILE)