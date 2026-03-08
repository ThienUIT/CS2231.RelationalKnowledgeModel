# -*- coding: utf-8 -*-
"""
Script 05: Gộp và Chuẩn Hóa Dữ Liệu Cuối Cùng cho Neo4j Bulk Import.

Mục đích:
1.  Đọc và gộp nhiều file CSV chứa thông tin về các nút (nodes).
2.  Thực hiện bước quan trọng nhất: **Loại bỏ các nút bị trùng lặp** dựa trên ID duy nhất ('nodeId:ID'),
    đảm bảo mỗi thực thể chỉ tồn tại một lần duy nhất trong đồ thị.
3.  **CHUẨN HÓA CỘT LABEL:** Loại bỏ dấu tiếng Việt và định dạng PascalCase cho các label.
4.  Đọc và gộp nhiều file CSV chứa thông tin về các cạnh (relationships) thành một file duy nhất.
5.  Xuất ra 2 file CSV cuối cùng, sẵn sàng để sử dụng với lệnh `neo4j-admin database import full`.
"""

import pandas as pd
import os
import sys
from unidecode import unidecode
import re

# --- CẤU HÌNH ---
NODE_FILES = ['result_final/graph_nodes_2013.csv', 'result_final/graph_nodes_2024.csv']
EDGE_FILES = ['result_final/graph_edges_2013.csv', 'result_final/graph_edges_2024.csv', 'result_final/graph_edges_comparison.csv']
FINAL_NODE_FILE = 'result_final/nodes_final.csv'
FINAL_EDGE_FILE = 'result_final/relationships_final.csv'

# ==============================================================================
# DANH SÁCH CÁC NHÓM THỰC THỂ ĐỒNG NGHĨA
# ==============================================================================
synonym_groups = [
    # --- NHÓM CÁC CƠ QUAN HÀNH CHÍNH NHÀ NƯỚC ---
    {"chuthe_ubndcaccap", "chuthe_uybannhandancaccap", "chuthe_ubanndancaccap", "chuthe_uybanndcaccap", "chuthe_uybannhandan"},
    {"chuthe_ubndcapptinh", "chuthe_ubndcaptinh", "chuthe_ubndcappt", "chuthe_uybanndancaptinh", "chuthe_uybanndcaptinh", "chuthe_uybannhandancaptinh", "chuthe_ubndcoptinh", "chuthe_ubndcadtinh", "chuthe_uybanNDcấptinh", "chuthe_uybannd", "chuthe_uybanndcancaptinh", "chuthe_uybanndcapptinh", "chuthe_ubndcactinh", "chuthe_ubnd_cap_tinh"},
    {"chuthe_chutichubndcaptinh", "chuthe_chutichuybanndcaptinh", "chuthe_chutichtinh", "chuthe_chu_tichubndcaptinh", "chuthe_chutich_ubnd_captinh"},
    {"chuthe_ubndcaphuyen", "chuthe_uybannhandancaphuyen", "chuthe_uybanndcphuyen", "chuthe_uybanndcaphuyen", "chuthe_ubndcabhuyen", "chuthe_uybanNDcấphuyen", "chuthe_uybanndancaphuyen", "chuthe_ubndcph", "chuthe_ubanndcaphuyen", "chuthe_ubnd_cap_huyen"},
    {"chuthe_chutichubndcaphuyen", "chuthe_chutichuybanndcaphuyen", "chuthe_chutichhuyen", "chuthe_chutichuybannhandancaphuyen", "chuthe_chutich_ubnd_caphuyen"},
    {"chuthe_ubndcapxa", "chuthe_uybannhandancapxa", "chuthe_uybanndcpxa", "chuthe_uybanndancapxa", "chuthe_uybanndcapxa", "chuthe_ubanndancapxa", "chuthe_ubndcpx", "chuthe_ubnd_cap_xa"},
    {"chuthe_chutichubndcapxa", "chuthe_chutichuybanndcapxa", "chuthe_chutichubndancapxa"},
    {"chuthe_botainguyenvamoitruong", "chuthe_botnmt", "chuthe_botainguyenvamientruong", "chuthe_botainnguyenvamoitruong", "chuthe_bothonguyenmoitruong", "chuthe_botainguyenvamotruong", "chuthe_botnguyenvamoitruong", "chuthe_botainguyenvmoitruong", "chuthe_botainguyenvamitruong"},
    {"chuthe_botruongbotainguyenvamoitruong", "chuthe_bothutruongbotainguyenmoitruong", "chuthe_botruongbotainuyenvamoi", "chuthe_botruongbotnmt", "chuthe_botruongbotainguon", "chuthe_botruongbotainguonvamoitruong", "chuthe_botruongbotainguyenvamt", "chuthe_botruongbotainuyenvamoitruong", "chuthe_botruongbotnvmt", "chuthe_botruongbotnguyenvamoitruong", "chuthe_botruongbotainnguyenvamoitruong", "khainiem_botruongbotainuyenvmt", "chuthe_botruongbotainuyenmoitruong", "chuthe_bo_truong_bo_tainguyenvmoitruong"},
    {"chuthe_boquocphong", "chuthe_botruongboquocphong", "chuthe_bo_truong_bo_quoc_phong"},
    {"chuthe_bocongan", "chuthe_botruongbocongan", "chuthe_bo_truong_bo_cong_an"},
    {"chuthe_chinhphu", "chuthe_chinphu"},
    {"chuthe_quochoi", "chuthe_quochoc"},
    {"chuthe_coquannhanuoc", "chuthe_coquannhanhanuoc"},
    {"chuthe_coquancothamquyen", "chuthe_coquanthamquyen", "chuthe_coquan_co_tham_quyen", "chuthe_coquanauthomquyen", "chuthe_cơquanbothamquyen", "chuthe_coquancthamquyen", "chuthe_coquanhuuthamquyen", "chuthe_coquanauthuyen"},

    # --- NHÓM CÁC CHỦ THỂ & KHÁI NIỆM NGƯỜI/TỔ CHỨC ---
    {"chuthe_nhanuoc", "chuthe_nha_nuoc", "khainiem_nhanuoc"},
    {"chuthe_nguoisudungdat", "khainiem_nguoisudungdat", "khainiem_nguoichusudungdat"},
    {"chuthe_canhan", "chuthe_canhansudungdat"},
    {"chuthe_hogiadinh", "chuthe_hogiadinhsudungdat"},
    {"chuthe_congdongdancu", "chuthe_conggdongdancu", "chuthe_congdong_dancu", "chuthe_congdongdancusudungdat", "chuthe_congdong"},
    {"chuthe_tochuc", "chuthe_tochuckinhte", "chuthe_tochucdanhtaichinh", "chuthe_tochuc_kinhte", "chuthe_tochucchinhte", "chuthe_tochucconomi", "chuthe_tochucKT", "chuthe_tochucKinhte", "chuthe_tochucdoanhnghiep", "chuthe_tochucdongnghiep", "chuthe_tochucyte", "chuthe_tochucdong", "chuthe_tochucdanhmuc", "chuthe_tochucKinhTe", "chuthe_tochuc اقتصادي", "chuthe_tochuc_kinh_te", "chuthe_tochickinhte"},
    {"chuthe_tochuckinhtecovondautunuocngoai", "chuthe_tochucchinhtecovondautunuocngoai", "chuthe_tcktvonnuocngoai", "chuthe_tochuc_kinhte_covondautunuocngoai", "chuthe_tochuc_kinhtecovondautunuocngoai", "chuthe_tochuccovondautunuocngoai", "chuthe_tochucconomicovondautunuocngoai", "chuthe_tochucKTvonnuocngoai", "chuthe_tochucKinhtecovondautunuocngoai", "chuthe_tochucvondautunuocngoai", "chuthe_tochuc_kinhtecovondautunuocngoai", "chuthe_tochuckinhtevondautu"},
    {"chuthe_nguoigocvietnamdinhcuonuocngoai", "chuthe_nguoigocvietnam", "chuthe_nguoigocvietnamdincuonnuocngoai", "chuthe_nguoigocviet", "chuthe_nguoigocVietNamdinhcuonnuocngoai", "chuthe_nguoivietnamdincuonnuocngoai", "chuthe_nguoivietnamdinhcuonuocngoai", "chuthe_nguoivietnamdicuonuocngoai", "chuthe_nguoiVietNamdinhcuonuocto"},
    {"chuthe_donvisunghiepclap", "chuthe_donvisunghiepcong", "chuthe_donvisungnghiepclap", "chuthe_donvisunghiệpconglap", "chuthe_donvisungghieplap", "chuthe_donvisunghieplapcong", "chuthe_donvisunghiệpcong", "chuthe_donvisungiep"},

    # --- NHÓM CÁC HÀNH VI & KHÁI NIỆM PHÁP LÝ CỐT LÕI ---
    {"hanhviphaply_thuhoidat", "chetai_thuhoidat", "hanhviphaply_nhanuoc_thuhoidat", "hanhviphaply_nhanuocchothue", "khainiem_nhanuocthuhoidat", "hanhviphaply_nhanuoc_thuhoildat", "hanhviphaply_thuhhoi"},
    {"hanhviphaply_boithuong", "khainiem_boithuong", "chetai_boithuong"},
    {"hanhviphaply_boithuongvedat", "khainiem_boiduongvedat", "khainiem_bồi_thườngvềđất", "khainiem_boithuongvedat"},
    {"hanhviphaply_boithuonghotrotaidinhcu", "khainiem_boiduonghotrotaidinhcu"},
    {"khainiem_quyhoachsudungdat", "khainiem_quydatsdd"},
    {"khainiem_kehoachsudungdat", "khainiem_kếhoachsudungdat", "khainiem_kehachsudungdat"},
    {"hanhviphaply_giaodat", "hanhvi_giaodat", "khainiem_giaodat"},
    {"hanhviphaply_chothuedat", "hanhvi_thuedat", "khainiem_chothuedat", "dieukien_thuedat"},
    {"khainiem_giaychungnhanquyensudungdat", "khainiem_giaychungnhanquyensdd", "khainiem_giaychungnhan"},
    {"khainiem_taisanganlienvoidat", "hanhviphaply_taisanganlienvoidat", "khainiem_taisangganlienvoiđat", "khainiem_taisangganlienvoithat", "khainiem_taisanganglienvoidat"},
    {"hanhviphaply_chuyenmucdichsudungdat", "hanhviphaply_chuyenmucdichsd", "hanhviphaply_chuyenmucdich", "hanhviphaply_chuyenmucdich_sudungdat", "khainiem_chuyenmucdichsudungdat"},
    {"hanhviphaply_chuyennhuongquyen", "hanhviphaply_chuyennhuongquyensudungdat", "hanhviphaply_chuyennhuongdat", "hanhviphaply_chuyennhuongquysudungdat", "khainiem_chuyennhuongquyensudungdat"},
    {"hanhviphaply_gopvon", "khainiem_gopvon"},
    {"hanhviphaply_thua_ke", "hanhviphaply_thuaKe", "hanhviphaply_thừa_kế"},
    {"hanhviphaply_thechap", "khainiem_thechap"},
    {"khainiem_banggiadat", "khainiem_banggia"},
    {"hanhviphaply_dangkydatdai", "hanhviphaply_thutucdangkydatdai", "hanhviphaply_thutudangkydatdai"},

    # --- NHÓM CÁC BIẾN THỂ VIẾT TẮT, LỖI GÕ MÁY ---
    {"chuthe_botruongbocongan", "chuthe_bo_truong_bo_cong_an"},
    {"chuthe_botruongbocongthuong", "chuthe_bo_truong_bo_congthuong"},
    {"chuthe_botruongbogiaothongvantai", "chuthe_bo_truong_bo_giaothongvantai"},
    {"chuthe_botruongbonongnghiepvaphattriennongthon", "chuthe_bo_truong_bo_nongnghiepvaphattriennongthon"},
    {"chuthe_botruongboyte", "chuthe_bo_truong_bo_yte"},
    {"chuthe_bo", "chuthe_bocoquanngangbo", "chuthe_bo,nganh", "chuthe_bonganh"},

    # lỗi :END_ID thiếu - Sửa manual
    {"khainiem_dattinnguong", "khainiem_dattinnnguong"}, 
    {"dieukien_khong_cap_gcn", "dieukien_dieukienkhongcapgiaychungnhan"},
    {'dieukien_khongcapgiaychungnhan', 'dieukien_dieukienkhongcapgiaychungnhan'},
]

def create_semantic_mapping(synonym_groups):
    """Tạo một từ điển ánh xạ từ các ID biến thể sang ID chuẩn."""
    mapping = {}
    for group in synonym_groups:
        # Chọn ID ngắn nhất và không chứa ký tự lặp làm ID chuẩn
        canonical_id = sorted(list(group), key=len)[0] 
        for synonym in group:
            if synonym != canonical_id:
                mapping[synonym] = canonical_id
    return mapping

def normalize_id(entity_id):
    if not isinstance(entity_id, str): return ""
    normalized = unidecode(entity_id)
    normalized = normalized.lower()
    normalized = re.sub(r'[\s\W]+', '_', normalized)
    return normalized.strip('_')

def normalize_label(label):
    """
    Hàm mới để chuẩn hóa label:
    1. Bỏ dấu (ví dụ: "ĐiềuLuật" -> "DieuLuat")
    2. Chuyển thành PascalCase (ví dụ: "hanh vi phap ly" -> "HanhViPhapLy")
    """
    if not isinstance(label, str): return "UnknownLabel"
    
    # Bỏ dấu
    no_accent_label = unidecode(label)
    
    # Tạo PascalCase: Chuyển các từ được phân tách bởi khoảng trắng hoặc gạch dưới
    # thành dạng viết hoa chữ cái đầu và nối lại.
    words = re.split(r'[\s_-]+', no_accent_label)
    pascal_case_label = "".join(word.capitalize() for word in words)
    
    return pascal_case_label

def finalize_files_for_import(node_files, edge_files, final_node_file, final_edge_file, synonym_groups):
    """
    Hàm chính để thực hiện việc gộp và chuẩn hóa dữ liệu.
    """
    print("--- Tạo từ điển ánh xạ ngữ nghĩa ---")
    semantic_map = create_semantic_mapping(synonym_groups)
    
    # ==========================================================================
    # PHẦN 1: XỬ LÝ VÀ HỢP NHẤT NODE
    # ==========================================================================
    print("\n--- Bắt đầu gộp và chuẩn hóa các file Node ---")
    
    node_dfs_list = [pd.read_csv(f) for f in node_files if os.path.exists(f)]
    if not node_dfs_list:
        print("Lỗi: Không có file node nào. Dừng chương trình.")
        sys.exit(1)
        
    combined_nodes_df = pd.concat(node_dfs_list, ignore_index=True)
    
    # *** BƯỚC MỚI: CHUẨN HÓA CỘT LABEL ***
    print("Áp dụng chuẩn hóa cho cột ':LABEL'...")
    combined_nodes_df[':LABEL'] = combined_nodes_df[':LABEL'].apply(normalize_label)

    print("Áp dụng chuẩn hóa và ánh xạ ngữ nghĩa cho cột 'nodeId:ID'...")
    combined_nodes_df['nodeId:ID'] = combined_nodes_df['nodeId:ID'].apply(
        lambda x: semantic_map.get(normalize_id(x), normalize_id(x))
    )

    initial_node_count = len(combined_nodes_df)
    final_nodes_df = combined_nodes_df.drop_duplicates(subset=['nodeId:ID'], keep='last')
    final_node_count = len(final_nodes_df)
    
    print(f"Tổng số node ban đầu: {initial_node_count}. Sau khi hợp nhất: {final_node_count} nút duy nhất.")
    final_nodes_df.to_csv(final_node_file, index=False, encoding='utf-8')
    print(f"Đã tạo file node cuối cùng: '{final_node_file}'")

    # ==========================================================================
    # PHẦN 2: XỬ LÝ VÀ HỢP NHẤT EDGE
    # ==========================================================================
    print("\n--- Bắt đầu gộp các file Edge ---")
    edge_dfs_list = [pd.read_csv(f) for f in edge_files if os.path.exists(f)]
    if not edge_dfs_list:
        print("Lỗi: Không có file edge nào. Dừng chương trình.")
        sys.exit(1)
        
    final_edges_df = pd.concat(edge_dfs_list, ignore_index=True)
    print(f"Tổng số mối quan hệ ban đầu: {len(final_edges_df)}")

    print("Áp dụng ánh xạ ngữ nghĩa cho START_ID và END_ID của các mối quan hệ...")
    final_edges_df[':START_ID'] = final_edges_df[':START_ID'].apply(
        lambda x: semantic_map.get(normalize_id(x), normalize_id(x))
    )
    final_edges_df[':END_ID'] = final_edges_df[':END_ID'].apply(
        lambda x: semantic_map.get(normalize_id(x), normalize_id(x))
    )
    
    final_edges_df.drop_duplicates(inplace=True)
    print(f"Số mối quan hệ sau khi chuẩn hóa và loại bỏ trùng lặp: {len(final_edges_df)}")

    final_edges_df.to_csv(final_edge_file, index=False, encoding='utf-8')
    print(f"Đã tạo file edge cuối cùng: '{final_edge_file}'")

# --- Main Execution ---
if __name__ == "__main__":
    finalize_files_for_import(NODE_FILES, EDGE_FILES, FINAL_NODE_FILE, FINAL_EDGE_FILE, synonym_groups)
    print("\n--- Hoàn tất! Dữ liệu đã được chuẩn hóa. Hãy chạy lại script 06 để kiểm tra lần cuối trước khi import. ---")