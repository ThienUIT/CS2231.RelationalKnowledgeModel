import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import sys

# --- Cấu hình (PHẢI KHỚP VỚI FILE BUILD) ---
MODEL_NAME = 'bkai-foundation-models/vietnamese-bi-encoder'
FAISS_INDEX_PATH = "faiss_index.bin"
LAW_IDS_PATH = "law_ids.json"

class SemanticRetriever:
    """
    Lớp để tải index và thực hiện tìm kiếm ngữ nghĩa (semantic search).
    Khởi tạo một lần và tái sử dụng cho nhiều truy vấn.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        """
        Triển khai theo mẫu Singleton để đảm bảo mô hình và index chỉ được tải một lần.
        """
        if not cls._instance:
            cls._instance = super(SemanticRetriever, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        # Kiểm tra xem đã khởi tạo chưa để tránh chạy lại __init__ nhiều lần
        if hasattr(self, 'model'):
            return

        print("Đang khởi tạo Semantic Retriever (chỉ chạy một lần)...")
        self.model = None
        self.index = None
        self.law_ids = None

        try:
            # Kiểm tra sự tồn tại của các file cần thiết
            if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(LAW_IDS_PATH):
                raise FileNotFoundError(
                    f"Không tìm thấy file '{FAISS_INDEX_PATH}' hoặc '{LAW_IDS_PATH}'. "
                    f"Vui lòng chạy script build_vector_database_from_kg.py trước."
                )

            # Tải mô hình embedding
            print(f" - Đang tải mô hình: '{MODEL_NAME}'...")
            self.model = SentenceTransformer(MODEL_NAME)
            print("   -> Tải mô hình thành công.")

            # Tải FAISS index
            print(f" - Đang tải FAISS index từ '{FAISS_INDEX_PATH}'...")
            self.index = faiss.read_index(FAISS_INDEX_PATH)
            print(f"   -> Tải FAISS index thành công. Chứa {self.index.ntotal} vector.")

            # Tải danh sách ID
            print(f" - Đang tải danh sách ID từ '{LAW_IDS_PATH}'...")
            with open(LAW_IDS_PATH, 'r', encoding='utf-8') as f:
                self.law_ids = json.load(f)
            print(f"   -> Tải danh sách ID thành công. Chứa {len(self.law_ids)} ID.")

            # Kiểm tra tính nhất quán quan trọng
            if self.index.ntotal != len(self.law_ids):
                print("\nCẢNH BÁO NGHIÊM TRỌNG: Số lượng vector trong index và số lượng ID không khớp!")
                raise ValueError("Dữ liệu index và ID không đồng bộ.")

            print("\n>>> Semantic Retriever đã sẵn sàng. <<<")

        except Exception as e:
            print(f"Lỗi nghiêm trọng khi khởi tạo SemanticRetriever: {e}", file=sys.stderr)
            # Đặt các thuộc tính về None để hàm search biết không thể hoạt động
            self.model = None
            self.index = None
            self.law_ids = None
            raise  # Ném lại lỗi để ứng dụng chính biết và dừng lại nếu cần

    def search(self, query: str, top_k: int = 5, score_threshold: float = 0.3):
        """
        Thực hiện tìm kiếm ngữ nghĩa.
        Input: 
            - query (str): Câu hỏi của người dùng.
            - top_k (int): Số lượng kết quả hàng đầu cần trả về.
            - score_threshold (float): Ngưỡng điểm tương đồng tối thiểu để chấp nhận kết quả.
        Output: 
            - Danh sách các tuple (id_điều_luật, điểm_tương_đồng).
        """
        if not all([self.model, self.index, self.law_ids]):
            print("Lỗi: Retriever chưa được khởi tạo đúng cách.", file=sys.stderr)
            return []

        # Chuyển câu hỏi thành vector, đảm bảo chuẩn hóa giống như lúc build index
        query_vector = self.model.encode([query], normalize_embeddings=True)
        
        # Tìm kiếm trong index
        # faiss trả về (distances, indices)
        scores, indices = self.index.search(query_vector.astype('float32'), top_k)
        
        # Lấy ra các ID và điểm số tương ứng
        results = []
        for i, score in zip(indices[0], scores[0]):
            # Lọc bỏ các kết quả có điểm tương đồng thấp hơn ngưỡng
            if score >= score_threshold:
                results.append((self.law_ids[i], float(score)))
        
        return results

# --- Ví dụ sử dụng và kiểm tra ---
if __name__ == '__main__':
    try:
        # Khởi tạo retriever (chỉ cần làm một lần khi ứng dụng khởi động)
        retriever = SemanticRetriever()
    
        print("\n--- Bắt đầu kiểm tra tìm kiếm ---")
        
        test_query_1 = "hạn mức nhận chuyển nhượng đất nông nghiệp của hộ gia đình là bao nhiêu?"
        print(f"\nCâu hỏi test 1: '{test_query_1}'")
        results_1 = retriever.search(test_query_1, top_k=3)
        print("Kết quả (ID, Score):", results_1)

        test_query_2 = "khi nào thì nhà nước thu hồi đất do vi phạm pháp luật?"
        print(f"\nCâu hỏi test 2: '{test_query_2}'")
        results_2 = retriever.search(test_query_2, top_k=3)
        print("Kết quả (ID, Score):", results_2)
        
        test_query_3 = "tôi muốn biết về lấn biển"
        print(f"\nCâu hỏi test 3: '{test_query_3}'")
        results_3 = retriever.search(test_query_3, top_k=3, score_threshold=0.3)
        print(f"Kết quả (ID, Score) với ngưỡng > 0.6:", results_3)

    except Exception as e:
        print(f"\nKhông thể chạy ví dụ do lỗi trong quá trình khởi tạo: {e}")