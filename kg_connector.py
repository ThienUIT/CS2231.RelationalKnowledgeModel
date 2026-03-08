# -*- coding: utf-8 -*-
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
import json

class KGConnector:
    """
    Lớp kết nối và thực hiện các truy vấn trên Knowledge Graph pháp lý.
    Sử dụng context manager ('with' statement) để quản lý kết nối tự động.
    """
    def __init__(self):
        """Khởi tạo kết nối đến Neo4j."""
        load_dotenv()
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")
        
        if not all([uri, user, password]):
            raise ValueError("Vui lòng thiết lập NEO4J_URI, NEO4J_USER, và NEO4J_PASSWORD trong file .env")
            
        self._driver = None
        try:
            self._driver = GraphDatabase.driver(uri, auth=(user, password))
            self._driver.verify_connectivity()
            print("Kết nối đến Neo4j thành công!")
        except Exception as e:
            print(f"Lỗi kết nối đến Neo4j: {e}")
            # Gán lại _driver = None để các hàm khác biết kết nối đã hỏng
            self._driver = None
            raise ConnectionError("Không thể kết nối tới cơ sở dữ liệu Neo4j.")

    def close(self):
        """Đóng kết nối đến driver Neo4j."""
        if self._driver is not None:
            self._driver.close()
            print("Đã đóng kết nối Neo4j.")

    # --- Hỗ trợ Context Manager ---
    def __enter__(self):
        """Cho phép sử dụng 'with KGConnector() as kg:'."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Tự động đóng kết nối khi ra khỏi khối 'with'."""
        self.close()

    def _run_query(self, query, parameters=None):
        """Hàm nội bộ để chạy một truy vấn Cypher."""
        if self._driver is None:
            print("Lỗi: Không có kết nối Neo4j hợp lệ.")
            return []
        try:
            with self._driver.session() as session:
                result = session.run(query, parameters or {})
                # Trả về danh sách các dictionary
                return [record.data() for record in result]
        except Exception as e:
            print(f"Lỗi thực thi truy vấn Cypher: {e}")
            print(f"Query: {query}")
            return []

    # --- Các hàm truy vấn chuyên biệt ---

    def get_all_laws_for_vectordb(self):
        """
        Lấy tất cả các nút Điều Luật có nội dung để xây dựng Vector Database.
        Hàm này đã được cập nhật để trả về cả `name` (tên điều luật)
        để có thể xây dựng "siêu văn bản" (super-document) cho embedding.
        """
        print("Đang lấy dữ liệu (ID, name, noi_dung) của tất cả các Điều Luật từ KG...")
        
        # LABEL đã được chuẩn hóa thành 'DieuLuat'
        # Thuộc tính ID đã được chuẩn hóa thành 'nodeId'
        query = """
        MATCH (n:DieuLuat)
        WHERE n.noi_dung IS NOT NULL AND trim(n.noi_dung) <> ''
        RETURN n.nodeId AS id, n.name AS name, n.noi_dung AS content
        """
        return self._run_query(query)

    def get_node_by_id(self, node_id: str):
        """Lấy toàn bộ thuộc tính của một nút dựa trên nodeId của nó."""
        if not node_id: return None
        
        query = "MATCH (n {nodeId: $node_id}) RETURN properties(n) AS properties"
        parameters = {"node_id": node_id}
        result = self._run_query(query, parameters)
        
        return result[0]['properties'] if result else None

    def find_comparison_by_law_id(self, law_id_2024: str):
        """
        Tìm kiếm sự so sánh cho một điều luật 2024.
        Trả về điều luật 2024, điều luật 2013 tương ứng và loại thay đổi.
        """
        if not law_id_2024: return None
        
        # Các loại quan hệ so sánh có thể có
        comparison_types = "SUA_DOI_BO_SUNG|THAY_THE_HOAN_TOAN|GIU_NGUYEN"
        
        query = f"""
        MATCH (new_law:DieuLuat {{nodeId: $law_id}})
        // Tùy chọn khớp với mối quan hệ so sánh
        OPTIONAL MATCH (new_law)-[r:{comparison_types}]->(old_law:DieuLuat)
        RETURN 
            properties(new_law) as new_law_props, 
            properties(old_law) as old_law_props,
            type(r) as change_type
        """
        parameters = {"law_id": law_id_2024}
        result = self._run_query(query, parameters)
        
        return result[0] if result else None
    
    def find_laws_by_concept_name(self, concept_name: str, law_year: int = None, limit: int = 10):
        """
        Tìm các điều luật liên quan đến một khái niệm, chủ thể, hoặc hành vi.
        Có thể lọc theo năm luật.
        """
        print(f"Đang tìm các luật liên quan đến '{concept_name}'...")
        
        law_filter_clause = ""
        parameters = {"concept_name": concept_name, "limit": limit, "law_year": law_year}
        
        # Chỉ thêm mệnh đề WHERE nếu law_year thực sự được cung cấp
        if law_year is not None:
            # *** THAY ĐỔI QUAN TRỌNG: ÉP KIỂU VỀ SỐ NGUYÊN TRƯỚC KHI SO SÁNH ***
            law_filter_clause = "WHERE toInteger(d.phien_ban) = $law_year"
            
        query = f"""
            MATCH (c)
            WHERE (c:KhaiNiem OR c:ChuThe OR c:HanhViPhapLy) 
            AND toLower(c.name) CONTAINS toLower($concept_name)
            MATCH (c)<-[r]-(d:DieuLuat)
            WITH d
            {law_filter_clause}
            RETURN d.nodeId AS id, d.name AS name, d.phien_ban AS phien_ban
            ORDER BY d.phien_ban DESC, toInteger(d.ma_dieu) ASC
            LIMIT $limit
        """
        
        # Loại bỏ law_year khỏi parameters nếu nó là None để tránh lỗi Cypher
        if law_year is None:
            del parameters["law_year"]
            
        return self._run_query(query, parameters)

    def keyword_search_laws(self, keyword: str, law_year: int = None, limit: int = 5):
        """
        Tìm kiếm từ khóa bằng Full-Text Index để có hiệu năng cao nhất.
        Yêu cầu: Đã tạo index bằng lệnh:
        CREATE FULLTEXT INDEX lawTextIndex FOR (n:DieuLuat) ON EACH [n.name, n.noi_dung]
        """
        print(f"Đang tìm kiếm từ khóa (Full-Text): '{keyword}'...")

        # Truy vấn tìm kiếm bằng index
        search_query = "CALL db.index.fulltext.queryNodes('lawTextIndex', $keyword) YIELD node, score"
        
        # Mệnh đề WHERE để lọc thêm
        where_clauses = []
        parameters = {"keyword": keyword, "limit": limit}
        if law_year is not None:
            where_clauses.append("toInteger(node.phien_ban) = $law_year")
            parameters["law_year"] = law_year
            
        full_where_clause = ""
        if where_clauses:
            full_where_clause = "WHERE " + " AND ".join(where_clauses)

        query = f"""
        {search_query}
        WITH node, score
        {full_where_clause}
        RETURN node.nodeId AS id, node.name AS name, node.phien_ban AS phien_ban, score
        ORDER BY score DESC
        LIMIT $limit
        """
        return self._run_query(query, parameters)

# --- VÍ DỤ SỬ DỤNG VÀ KIỂM TRA ---
if __name__ == '__main__':
    try:
        with KGConnector() as kg:
            # 1. Test lấy dữ liệu cho VectorDB
            all_laws = kg.get_all_laws_for_vectordb()
            print(f"\n1. Tìm thấy {len(all_laws)} điều luật có nội dung để tạo VectorDB.")
            if all_laws:
                print("   -> Ví dụ điều luật đầu tiên:", all_laws[0])
            else:
                print("   -> Không tìm thấy điều luật nào có nội dung.")

            # 2. Test lấy chi tiết một nút
            test_node_id = "dieu_27_2024"
            print(f"\n2. Test: Lấy chi tiết nút '{test_node_id}'")
            details = kg.get_node_by_id(test_node_id)
            if details:
                print(json.dumps(details, indent=2, ensure_ascii=False))
            else:
                print(f"   -> Không tìm thấy thông tin cho '{test_node_id}'.")
            
            # 3. Test tìm kiếm so sánh
            test_comparison_id = "dieu_81_2024"
            print(f"\n3. Test: Tìm so sánh cho '{test_comparison_id}'")
            comparison = kg.find_comparison_by_law_id(test_comparison_id)
            if comparison and comparison.get('old_law_props'):
                new_name = comparison['new_law_props'].get('name', 'N/A')
                old_name = comparison['old_law_props'].get('name', 'N/A')
                change_type = comparison.get('change_type', 'N/A')
                print(f"   -> '{new_name}' [{change_type}] '{old_name}'") # Thêm dấu nháy để rõ ràng hơn
            elif comparison:
                new_name = comparison['new_law_props'].get('name', 'N/A')
                print(f"   -> '{new_name}' là một ĐIỀU LUẬT MỚI.")
            else:
                print(f"   -> Không tìm thấy thông tin so sánh cho '{test_comparison_id}'.")
            
            # 4. Test tìm luật theo khái niệm (cho cả 2 luật)
            test_concept = "Hộ gia đình"
            print(f"\n4. Test: Tìm luật liên quan đến khái niệm '{test_concept}'")
            related_laws = kg.find_laws_by_concept_name(test_concept) 
            print(f"   -> Các luật liên quan đến '{test_concept}':")
            for law in related_laws:
                print(f"      - {law['id']}: {law['name']}")

            # 5. Test tìm luật theo khái niệm (chỉ trong luật 2024)
            print(f"\n5. Test: Tìm luật liên quan đến '{test_concept}' (chỉ trong Luật 2024)")
            related_laws_2024 = kg.find_laws_by_concept_name(test_concept, law_year=2024)
            print(f"   -> Các luật liên quan (2024):")
            for law in related_laws_2024:
                print(f"      - {law['id']}: {law['name']}")

            # 6. Test tìm kiếm từ khóa
            test_keyword = "lấn chiếm"
            print(f"\n6. Test: Tìm kiếm từ khóa '{test_keyword}'")
            keyword_results = kg.keyword_search_laws(test_keyword, limit=3)
            print(f"   -> Kết quả tìm kiếm cho '{test_keyword}':")
            for res in keyword_results:
                 print(f"      - {res['id']}: {res['name']}")

    except ConnectionError as e:
        print(f"\nKhông thể chạy ví dụ do lỗi kết nối: {e}")