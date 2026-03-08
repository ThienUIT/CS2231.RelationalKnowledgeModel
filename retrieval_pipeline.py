# file: retrieval_pipeline.py

from semantic_retriever import SemanticRetriever
from reranker import Reranker
from kg_connector import KGConnector

class ComprehensiveRetriever:
    def __init__(self):
        # ... (phần khởi tạo giữ nguyên)
        self.kg_connector = KGConnector()
        self.semantic_retriever = SemanticRetriever()
        self.reranker = Reranker()

    def retrieve(self, query: str, initial_k: int = 20, final_k: int = 5):
        """
        Thực hiện pipeline truy xuất hoàn chỉnh: Search -> Rerank.

        Args:
            query (str): Câu hỏi của người dùng.
            initial_k (int): Số lượng ứng viên ban đầu cần lấy từ Semantic Search.
            final_k (int): Số lượng kết quả cuối cùng sau khi đã rerank.
        """
        print(f"\n===== Bắt đầu Pipeline Truy xuất cho câu hỏi: '{query}' =====")
        
        # --- Giai đoạn 1: Tìm kiếm ứng viên (Candidate Retrieval) ---
        print(f"\n[Bước 1] Tìm kiếm ngữ nghĩa để lấy top {initial_k} ứng viên...")
        
        # *** THAY ĐỔI Ở ĐÂY ***
        # Truyền giá trị `initial_k` vào hàm search của retriever
        candidate_ids_with_scores = self.semantic_retriever.search(query, top_k=initial_k)
        
        if not candidate_ids_with_scores:
            print("Không tìm thấy ứng viên nào từ Semantic Search.")
            return []
            
        print(f" -> Tìm thấy {len(candidate_ids_with_scores)} ứng viên.")

        # Lấy nội dung chi tiết của các ứng viên từ KG
        candidate_docs = []
        for law_id, semantic_score in candidate_ids_with_scores:
            node_properties = self.kg_connector.get_node_by_id(law_id)
            if node_properties:
                candidate_docs.append({
                    'id': law_id,
                    'name': node_properties.get('name', ''),
                    'content': f"Tên điều luật: {node_properties.get('name', '')}. Nội dung: {node_properties.get('noi_dung', '')}",
                    'semantic_score': semantic_score
                })

        # --- Giai đoạn 2: Sắp xếp lại (Reranking) ---
        print(f"\n[Bước 2] Sắp xếp lại {len(candidate_docs)} ứng viên bằng Cross-Encoder...")
        reranked_docs = self.reranker.rerank(query, candidate_docs)

        # *** THAY ĐỔI Ở ĐÂY ***
        # Lấy `final_k` kết quả cuối cùng
        final_results = reranked_docs[:final_k]
        
        print(f"\n===== Kết quả cuối cùng (Top {final_k} sau Reranking) =====")
        for doc in final_results:
             print(f"- ID: {doc['id']}, Name: {doc['name']}, Rerank Score: {doc.get('rerank_score'):.4f}")
             
        return final_results
        
    def close(self):
        self.kg_connector.close()

# --- Ví dụ sử dụng ---
if __name__ == '__main__':
    try:
        retriever_pipeline = ComprehensiveRetriever()
        
        test_query = "khi nào thì nhà nước thu hồi đất do vi phạm pháp luật?"
        
        # Bạn có thể điều chỉnh các giá trị k ngay tại đây khi gọi hàm
        final_documents = retriever_pipeline.retrieve(test_query, initial_k=10, final_k=3)
        
        retriever_pipeline.close()
    except Exception as e:
        print(f"Lỗi trong quá trình chạy pipeline: {e}")