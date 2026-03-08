from sentence_transformers import CrossEncoder
import torch

# --- Cấu hình ---
# Chọn một mô hình Cross-Encoder được huấn luyện cho tiếng Việt
RERANKER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

class Reranker:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Reranker, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if hasattr(self, 'model'):
            return
            
        print("Đang khởi tạo Reranker (chỉ chạy một lần)...")
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # Tải mô hình Cross-Encoder. 
            # max_length cho phép xử lý các đoạn văn bản dài hơn.
            self.model = CrossEncoder(RERANKER_MODEL_NAME, max_length=512, device=device)
            print(f" -> Tải mô hình Reranker '{RERANKER_MODEL_NAME}' trên '{device}' thành công.")
        except Exception as e:
            print(f"Lỗi nghiêm trọng khi tải mô hình Reranker: {e}")
            self.model = None
            raise

    def rerank(self, query: str, documents: list[dict]):
        """
        Sắp xếp lại một danh sách các tài liệu dựa trên mức độ liên quan với câu hỏi.

        Args:
            query (str): Câu hỏi của người dùng.
            documents (list[dict]): Một danh sách các dictionary, 
                                     mỗi dict chứa 'id' và 'content' của tài liệu.
                                     Ví dụ: [{'id': 'dieu_1_2024', 'content': 'Nội dung điều 1...'}, ...]

        Returns:
            list[dict]: Danh sách các tài liệu đã được sắp xếp lại, 
                        mỗi dict có thêm một trường 'rerank_score'.
        """
        if not self.model or not documents:
            return []

        # Cross-Encoder yêu cầu một danh sách các cặp [query, document_content]
        pairs = [[query, doc['content']] for doc in documents]
        
        print(f" -> Reranking {len(pairs)} tài liệu...")
        
        # model.predict() sẽ trả về một danh sách các điểm số
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Gắn điểm số vào lại các tài liệu
        for i, doc in enumerate(documents):
            doc['rerank_score'] = scores[i]
            
        # Sắp xếp danh sách tài liệu dựa trên điểm rerank_score, từ cao đến thấp
        sorted_documents = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
        
        return sorted_documents

# --- Ví dụ sử dụng ---
if __name__ == '__main__':
    try:
        reranker = Reranker()
        
        # Giả sử đây là kết quả từ SemanticRetriever
        initial_results = [
            {'id': 'dieu_16_2013', 'content': 'Nhà nước quyết định thu hồi đất, trưng dụng đất...'},
            {'id': 'dieu_139_2024', 'content': 'Giải quyết đối với trường hợp ... sử dụng đất có vi phạm pháp luật...'},
            {'id': 'dieu_81_2024', 'content': 'Các trường hợp thu hồi đất do vi phạm pháp luật về đất đai...'}
        ]
        
        test_query = "khi nào thì nhà nước thu hồi đất do vi phạm pháp luật?"
        
        print("\n--- Kết quả ban đầu (từ Bi-Encoder) ---")
        for doc in initial_results:
            print(f"- ID: {doc['id']}")

        reranked_results = reranker.rerank(test_query, initial_results)
        
        print("\n--- Kết quả sau khi Reranking (từ Cross-Encoder) ---")
        for doc in reranked_results:
            print(f"- ID: {doc['id']}, Score: {doc['rerank_score']:.4f}")

    except Exception as e:
        print(f"Không thể chạy ví dụ: {e}")