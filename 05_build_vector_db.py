import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
import sys
from kg_connector import KGConnector # Đảm bảo file kg_connector.py ở cùng thư mục

# --- Cấu hình ---
# Sử dụng mô hình được tối ưu cho tiếng Việt để có độ chính xác cao nhất.
MODEL_NAME = 'bkai-foundation-models/vietnamese-bi-encoder'
FAISS_INDEX_PATH = "faiss_index.bin"
LAW_IDS_PATH = "law_ids.json"

def build_vector_database():
    """
    Kết nối đến Neo4j, lấy dữ liệu các điều luật, tạo vector embedding,
    xây dựng index FAISS và lưu kết quả một cách an toàn.
    """
    
    # --- Bước 1: Kết nối và lấy dữ liệu từ Knowledge Graph ---
    print("Đang khởi tạo kết nối đến Knowledge Graph...")
    all_laws = []
    try:
        with KGConnector() as kg:
            # Hàm này đã được xác thực là hoạt động đúng
            all_laws = kg.get_all_laws_for_vectordb() 
    except Exception as e:
        print(f"Lỗi nghiêm trọng khi kết nối hoặc lấy dữ liệu từ Neo4j: {e}", file=sys.stderr)
        sys.exit(1) # Dừng hẳn chương trình nếu không thể kết nối KG

    if not all_laws:
        print("Cảnh báo: Không có dữ liệu luật nào được trả về từ KG. Dừng quá trình.")
        return

    print(f"Đã lấy thành công {len(all_laws)} điều luật từ KG.")

    # Tách riêng ID và nội dung để embedding
    law_ids = [law['id'] for law in all_laws]
    contents_to_embed = [
        f"Tên điều luật: {law.get('name', '')}. Nội dung: {law.get('content', '')}" 
        for law in all_laws
    ]

    # --- Bước 2: Tải mô hình và tạo Embeddings ---
    try:
        print(f"Đang tải mô hình embedding: '{MODEL_NAME}' (có thể mất một lúc)...")
        # Sử dụng GPU nếu có, nếu không tự động chuyển về CPU
        model = SentenceTransformer(MODEL_NAME, device='cuda' if torch.cuda.is_available() else 'cpu') 
    except Exception as e:
        print(f"Lỗi khi tải mô hình SentenceTransformer: {e}", file=sys.stderr)
        print("Hãy đảm bảo bạn đã cài đặt: pip install sentence-transformers torch", file=sys.stderr)
        sys.exit(1)

    print("Bắt đầu tạo vector embeddings...")
    embeddings = model.encode(
        contents_to_embed, 
        show_progress_bar=True, 
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    if embeddings is None or embeddings.shape[0] != len(law_ids):
        print("Lỗi: Quá trình encoding thất bại hoặc số lượng vector không khớp với số lượng ID.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Đã tạo thành công {embeddings.shape[0]} vector với chiều là {embeddings.shape[1]}.")

    # --- Bước 3 & 4: Xây dựng và Lưu Index FAISS và IDs ---
    try:
        print("Đang xây dựng FAISS index...")
        dimension = embeddings.shape[1]
        
        # IndexFlatIP (Inner Product) là lựa chọn tối ưu khi normalize_embeddings=True
        # vì nó tương đương với Cosine Similarity và rất nhanh cho brute-force.
        index = faiss.IndexFlatIP(dimension)
        
        # FAISS yêu cầu kiểu dữ liệu là float32
        index.add(embeddings.astype('float32'))
        
        # --- Lưu file một cách an toàn ---
        # Chỉ ghi các file khi tất cả các bước trước đó đã thành công
        print("Đang lưu các file index và ID...")
        faiss.write_index(index, FAISS_INDEX_PATH)
        print(f"-> Đã lưu FAISS index vào: '{FAISS_INDEX_PATH}'")

        with open(LAW_IDS_PATH, 'w', encoding='utf-8') as f:
            json.dump(law_ids, f)
        print(f"-> Đã lưu danh sách ID điều luật vào: '{LAW_IDS_PATH}'")
    
    except Exception as e:
        print(f"Lỗi trong quá trình xây dựng hoặc lưu file FAISS: {e}", file=sys.stderr)
        sys.exit(1)
    
    print("\n--- Hoàn thành xây dựng Vector Database! ---")

# Thêm import torch để kiểm tra CUDA
if __name__ == "__main__":
    try:
        import torch
    except ImportError:
        print("Lỗi: Thư viện PyTorch chưa được cài đặt. Vui lòng cài đặt: pip install torch", file=sys.stderr)
        sys.exit(1)
    build_vector_database()