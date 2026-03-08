import pandas as pd
import os

# --- Cấu hình ---
FINAL_NODE_FILE = 'result_final/nodes_final.csv'
FINAL_EDGE_FILE = 'result_final/relationships_final.csv'

def validate_files(node_file, edge_file):
    """
    Kiểm tra tính toàn vẹn giữa file node và file edge.
    Tìm ra các mối quan hệ đang tham chiếu đến các node không tồn tại.
    """
    if not os.path.exists(node_file) or not os.path.exists(edge_file):
        print(f"Lỗi: Không tìm thấy file '{node_file}' hoặc '{edge_file}'.")
        return

    print(f"--- Bắt đầu kiểm tra tính toàn vẹn cho '{node_file}' và '{edge_file}' ---")

    try:
        # 1. Đọc tất cả các ID nút duy nhất vào một Set để tra cứu nhanh
        print(f"Đang đọc tất cả các ID từ '{node_file}'...")
        df_nodes = pd.read_csv(node_file)
        # Sử dụng .astype(str) để đảm bảo tất cả ID đều là chuỗi, tránh lỗi so sánh
        node_ids = set(df_nodes['nodeId:ID'].astype(str))
        print(f"Đã tìm thấy {len(node_ids)} ID nút duy nhất.")

        # 2. Đọc file cạnh và kiểm tra từng dòng
        print(f"Đang kiểm tra các mối quan hệ trong '{edge_file}'...")
        df_edges = pd.read_csv(edge_file)
        
        missing_start_nodes = []
        missing_end_nodes = []
        
        # iterrows() sẽ lặp qua từng dòng của DataFrame
        for index, row in df_edges.iterrows():
            start_id = str(row[':START_ID'])
            end_id = str(row[':END_ID'])

            # Kiểm tra xem start_id có trong tập hợp node_ids không
            if start_id not in node_ids:
                missing_start_nodes.append({
                    'row_index': index + 2, # +2 vì index bắt đầu từ 0 và có 1 dòng header
                    'missing_id': start_id,
                    'relationship': f"{start_id}-[{row[':TYPE']}]->{end_id}"
                })
            
            # Kiểm tra xem end_id có trong tập hợp node_ids không
            if end_id not in node_ids:
                 missing_end_nodes.append({
                    'row_index': index + 2,
                    'missing_id': end_id,
                    'relationship': f"{start_id}-[{row[':TYPE']}]->{end_id}"
                })

        # 3. In kết quả
        print("\n--- Kết quả kiểm tra ---")
        if not missing_start_nodes and not missing_end_nodes:
            print("Tuyệt vời! Tất cả các mối quan hệ đều hợp lệ. Dữ liệu có vẻ tốt.")
        else:
            print(f"Phát hiện vấn đề về tính toàn vẹn:")
            if missing_start_nodes:
                print(f"\nTìm thấy {len(missing_start_nodes)} mối quan hệ có ':START_ID' không tồn tại trong file node:")
                for error in missing_start_nodes:
                    print(f"  - Dòng {error['row_index']} trong '{edge_file}': Node '{error['missing_id']}' bị thiếu. (Quan hệ: {error['relationship']})")
            
            if missing_end_nodes:
                print(f"\nTìm thấy {len(missing_end_nodes)} mối quan hệ có ':END_ID' không tồn tại trong file node:")
                for error in missing_end_nodes:
                    print(f"  - Dòng {error['row_index']} trong '{edge_file}': Node '{error['missing_id']}' bị thiếu. (Quan hệ: {error['relationship']})")

    except Exception as e:
        print(f"Đã xảy ra lỗi trong quá trình kiểm tra: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    validate_files(FINAL_NODE_FILE, FINAL_EDGE_FILE)