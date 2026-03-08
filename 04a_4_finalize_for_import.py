# -*- coding: utf-8 -*-
"""
Gộp và Chuẩn Hóa Dữ Liệu Cuối Cùng cho Neo4j Bulk Import.

Mục đích:
1.  Đọc và gộp nhiều file CSV chứa thông tin về các nút (nodes).
2.  Thực hiện bước quan trọng nhất: **Loại bỏ các nút bị trùng lặp** dựa trên ID duy nhất ('nodeId:ID'),
    đảm bảo mỗi thực thể chỉ tồn tại một lần duy nhất trong đồ thị.
3.  Đọc và gộp nhiều file CSV chứa thông tin về các cạnh (relationships) thành một file duy nhất.
4.  Xuất ra 2 file CSV cuối cùng: `nodes_final.csv` và `relationships_final.csv`,
    sẵn sàng để sử dụng với lệnh `neo4j-admin database import full`.

Cách sử dụng:
1.  Đặt script này vào thư mục gốc của dự án.
2.  Đảm bảo các file CSV đầu vào (được định nghĩa trong `NODE_FILES` và `EDGE_FILES`)
    đã tồn tại trong cùng thư mục.
3.  Chạy script từ terminal: `python 05_finalize_for_import.py`
"""

import pandas as pd
import os
import sys

# --- CẤU HÌNH ---
# Liệt kê tất cả các file node đầu vào. Thứ tự quan trọng: file cuối cùng trong danh sách
# sẽ được ưu tiên giữ lại khi có ID trùng lặp.
NODE_FILES = [
    'result_final/graph_nodes_2013.csv', 
    'result_final/graph_nodes_2024.csv'
]

# Liệt kê tất cả các file edge đầu vào.
EDGE_FILES = [
    'result_final/graph_edges_2013.csv', 
    'result_final/graph_edges_2024.csv', 
    'result_final/graph_edges_comparison.csv'
]

# Tên các file đầu ra cuối cùng
FINAL_NODE_FILE = 'result_final/nodes_final.csv'
FINAL_EDGE_FILE = 'result_final/relationships_final.csv'

def finalize_files_for_import(node_files, edge_files, final_node_file, final_edge_file):
    """
    Hàm chính để thực hiện việc gộp và chuẩn hóa dữ liệu.
    """
    
    # ==========================================================================
    # PHẦN 1: XỬ LÝ VÀ HỢP NHẤT CÁC FILE NODE
    # ==========================================================================
    print("--- Bắt đầu gộp và chuẩn hóa các file Node ---")
    
    node_dfs_list = []
    for filepath in node_files:
        if not os.path.exists(filepath):
            print(f"Cảnh báo: Bỏ qua file node không tồn tại: '{filepath}'")
            continue
        
        try:
            df = pd.read_csv(filepath)
            print(f"Đã đọc thành công file node: '{filepath}' ({len(df)} dòng)")
            node_dfs_list.append(df)
        except Exception as e:
            print(f"Lỗi khi đọc file '{filepath}': {e}")

    if not node_dfs_list:
        print("Lỗi nghiêm trọng: Không có dữ liệu node nào để xử lý. Dừng chương trình.")
        sys.exit(1)

    # Gộp tất cả các DataFrame node thành một DataFrame duy nhất
    combined_nodes_df = pd.concat(node_dfs_list, ignore_index=True)
    initial_node_count = len(combined_nodes_df)
    print(f"\nTổng số dòng node từ tất cả các file: {initial_node_count}")

    # --- BƯỚC QUAN TRỌNG NHẤT: LOẠI BỎ TRÙNG LẶP ---
    # `drop_duplicates` sẽ tìm tất cả các dòng có cùng giá trị trong cột 'nodeId:ID'.
    # `keep='last'`: Khi tìm thấy các dòng trùng lặp, nó sẽ giữ lại dòng CUỐI CÙNG
    # và xóa các dòng trước đó. Điều này rất hữu ích vì chúng ta liệt kê file 2024
    # ở cuối, đảm bảo rằng thông tin từ luật mới hơn (nếu có) sẽ được ưu tiên.
    final_nodes_df = combined_nodes_df.drop_duplicates(subset=['nodeId:ID'], keep='last')
    final_node_count = len(final_nodes_df)
    
    print(f"Sau khi loại bỏ các node trùng lặp, còn lại: {final_node_count} nút duy nhất.")
    print(f"Số lượng nút đã được hợp nhất: {initial_node_count - final_node_count}")

    # Lưu file node cuối cùng
    # `index=False` để không ghi cột chỉ số của DataFrame vào file CSV.
    # `encoding='utf-8'` để đảm bảo tiếng Việt được lưu chính xác.
    final_nodes_df.to_csv(final_node_file, index=False, encoding='utf-8')
    print(f"Đã tạo file node cuối cùng: '{final_node_file}'")


    # ==========================================================================
    # PHẦN 2: XỬ LÝ VÀ HỢP NHẤT CÁC FILE EDGE
    # ==========================================================================
    print("\n--- Bắt đầu gộp các file Edge ---")

    edge_dfs_list = []
    for filepath in edge_files:
        if not os.path.exists(filepath):
            print(f"Cảnh báo: Bỏ qua file edge không tồn tại: '{filepath}'")
            continue
            
        try:
            df = pd.read_csv(filepath)
            print(f"Đã đọc thành công file edge: '{filepath}' ({len(df)} dòng)")
            edge_dfs_list.append(df)
        except Exception as e:
            print(f"Lỗi khi đọc file '{filepath}': {e}")
            
    if not edge_dfs_list:
        print("Lỗi nghiêm trọng: Không có dữ liệu edge nào để xử lý. Dừng chương trình.")
        sys.exit(1)

    # Đối với các mối quan hệ, chúng ta chỉ cần gộp chúng lại.
    # Việc trùng lặp cạnh thường ít xảy ra và không gây lỗi import như node.
    final_edges_df = pd.concat(edge_dfs_list, ignore_index=True)
    final_edge_count = len(final_edges_df)
    
    # Lưu file edge cuối cùng
    final_edges_df.to_csv(final_edge_file, index=False, encoding='utf-8')
    print(f"Đã tạo file edge cuối cùng: '{final_edge_file}' với tổng cộng {final_edge_count} mối quan hệ.")

# --- ĐIỂM BẮT ĐẦU THỰC THI SCRIPT ---
if __name__ == "__main__":
    finalize_files_for_import(NODE_FILES, EDGE_FILES, FINAL_NODE_FILE, FINAL_EDGE_FILE)
    print("\n--- Hoàn tất! Bạn có thể sử dụng 2 file `_final.csv` để import vào Neo4j. ---")