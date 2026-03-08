import streamlit as st
from kg_connector import KGConnector
from semantic_retriever import SemanticRetriever
from reranker import Reranker
from llm_callers import LLMClient
import re

# --- CÁC HÀM TIỆN ÍCH ---
def clean_query(query: str) -> str:
    """
    Hàm làm sạch câu hỏi của người dùng trước khi xử lý.
    """
    if not isinstance(query, str):
        return ""
    
    cleaned = query.lower()
    trigger_words = [
        "ok google", "hey siri", "alexa", "cho tôi hỏi", "cho mình hỏi", 
        "giúp tôi với", "giải thích", "định nghĩa", "là gì", "[help]"
    ]
    for word in trigger_words:
        cleaned = cleaned.replace(word, "")
    
    cleaned = re.sub(r'[^a-zA-Z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\s]', '', cleaned)
    cleaned = " ".join(cleaned.split())
    return cleaned.strip()

@st.cache_resource
def initialize_components():
    """
    Khởi tạo và cache lại các đối tượng KGConnector, SemanticRetriever và Reranker.
    """
    print("--- Đang khởi tạo các thành phần cốt lõi (chỉ chạy một lần) ---")
    try:
        kg = KGConnector()
        retriever = SemanticRetriever()
        reranker = Reranker()
        print("--- Khởi tạo hoàn tất ---")
        return kg, retriever, reranker
    except Exception as e:
        raise RuntimeError(f"Lỗi khởi tạo thành phần cốt lõi: {e}")

try:
    kg_connector, semantic_retriever, reranker = initialize_components()
except RuntimeError as e:
    st.error(f"Không thể khởi động ứng dụng. {e}")
    st.stop()

@st.cache_data(show_spinner=False)
def retrieval_pipeline(_query: str, initial_k: int = 20, final_k: int = 5):
    """
    Thực hiện pipeline truy xuất hoàn chỉnh: Search -> Rerank.
    Sử dụng _query với gạch dưới để Streamlit hiểu đây là hàm cache.
    """
    print(f"\n[PIPELINE] Bắt đầu truy xuất cho câu hỏi: '{_query}'")
    
    candidate_results = semantic_retriever.search(_query, top_k=initial_k, score_threshold=0.3)
    if not candidate_results:
        print("[PIPELINE] Không tìm thấy ứng viên nào từ Semantic Search.")
        return []
        
    print(f"[PIPELINE] -> Tìm thấy {len(candidate_results)} ứng viên.")

    candidate_docs = []
    for law_id, semantic_score in candidate_results:
        node_properties = kg_connector.get_node_by_id(law_id)
        if node_properties:
            super_content = f"Tên điều luật: {node_properties.get('name', '')}. Nội dung: {node_properties.get('noi_dung', '')}"
            candidate_docs.append({
                'id': law_id,
                'name': node_properties.get('name', ''),
                'phien_ban': node_properties.get('phien_ban', ''),
                'ma_dieu': node_properties.get('ma_dieu', ''),
                'content': super_content,
                'raw_content': node_properties.get('noi_dung', ''),
                'semantic_score': semantic_score
            })

    print(f"[PIPELINE] Bước 2: Sắp xếp lại {len(candidate_docs)} ứng viên...")
    reranked_docs = reranker.rerank(_query, candidate_docs)
    final_results = reranked_docs[:final_k]
    
    print("[PIPELINE] -> Hoàn thành truy xuất và reranking.")
    return final_results

# --- CÁC HÀM TẠO PROMPT ---
def build_qa_prompt(query, context):
    return f"""
        # CONTEXT
        Bạn đang hỗ trợ một hệ thống tri thức pháp lý về Luật Đất Đai Việt Nam.

        Hệ thống này cung cấp thông tin và giải đáp liên quan đến:
        - Luật Đất đai 2013
        - Luật Đất đai 2024
        - Các tình huống pháp lý liên quan đến đất đai tại Việt Nam

        # ROLE
        Bạn là chuyên gia pháp lý cao cấp về Luật Đất Đai Việt Nam.

        Bạn có nhiệm vụ:
        - Giải thích quy định pháp luật một cách chính xác
        - Phân tích các tình huống thực tế dựa trên quy định pháp luật
        - Trích dẫn điều luật khi cần thiết

        # OBJECTIVES
        1. Trả lời câu hỏi liên quan đến Luật Đất đai 2013 và 2024.
        2. Phân tích và đưa ra hướng xử lý cho các tình huống pháp lý về đất đai.
        3. Giải thích quy định pháp luật bằng ngôn ngữ dễ hiểu nhưng chính xác.

        # RULES
        - Chỉ sử dụng thông tin từ dữ liệu được cung cấp hoặc từ quy định của Luật Đất đai 2013 và 2024.
        - Không được tự ý bịa ra thông tin pháp lý.
        - Nếu không có đủ thông tin trong dữ liệu, hãy trả lời: 
        "Tôi không tìm thấy thông tin phù hợp trong dữ liệu hiện có."
        - Khi có thể, hãy trích dẫn:
        - Điều luật
        - Khoản
        - Năm của luật (2013 hoặc 2024)
        #Context là phần chứa các điều luật liên quan đã được truy xuất, sẽ giúp bạn trả lời chính xác hơn. Hãy sử dụng chúng một cách hiệu quả trong câu trả lời của bạn.
        {context}
        #USER QUESTION
        {query}
        # ANSWER FORMAT
        Nếu câu hỏi là **giải thích luật**:

        1. Tóm tắt quy định
        2. Trích dẫn điều luật
        3. Giải thích chi tiết
        4. Ví dụ minh họa (nếu cần)

        Nếu câu hỏi là **tình huống pháp lý**:

        1. Phân tích tình huống
        2. Áp dụng điều luật liên quan
        3. Kết luận hoặc hướng xử lý

        # RESPONSE STYLE
        - Rõ ràng
        - Chính xác
        - Có cấu trúc
        - Tránh suy đoán
    """

def build_comparison_prompt(query, context):
    return f"""
        # ROLE
        Bạn là chuyên gia pháp lý cao cấp về Luật Đất Đai Việt Nam.

        # CONTEXT
        Hệ thống tri thức này hỗ trợ phân tích và giải thích:
        - Luật Đất đai 2013
        - Luật Đất đai 2024

        # TASK
        Khi người dùng yêu cầu so sánh giữa các quy định pháp luật,
        hãy phân tích sự khác nhau giữa các phiên bản luật.

        # RULES
        - Chỉ sử dụng thông tin từ Luật Đất đai 2013 và 2024 hoặc dữ liệu được cung cấp.
        - Không suy đoán hoặc bịa thông tin.
        - Khi có thể, hãy trích dẫn:
        - Điều
        - Khoản
        - Luật năm nào

        # OUTPUT FORMAT

        Nếu câu hỏi yêu cầu **so sánh luật**, hãy trả lời theo cấu trúc:

        1. Quy định trong Luật Đất đai 2013
        - Trích dẫn điều luật
        - Tóm tắt nội dung

        2. Quy định trong Luật Đất đai 2024
        - Trích dẫn điều luật
        - Tóm tắt nội dung

        3. Điểm giống nhau
        - Liệt kê các điểm tương đồng

        4. Điểm thay đổi hoặc bổ sung
        - Liệt kê các thay đổi chính

        5. Ý nghĩa hoặc tác động thực tế
        - Giải thích việc thay đổi ảnh hưởng thế nào đến người dân, doanh nghiệp hoặc quản lý đất đai

        # STYLE
        - Trình bày rõ ràng
        - Có cấu trúc
        - Dễ hiểu nhưng chính xác

        #Context là phần chứa các điều luật liên quan đã được truy xuất, sẽ giúp bạn phân tích và so sánh chính xác hơn. Hãy sử dụng chúng một cách hiệu quả trong bài phân tích của bạn.
        {context}
        --- # USER QUESTION ---
        {query}
    """

# --- THIẾT KẾ GIAO DIỆN NGƯỜI DÙNG ---
st.set_page_config(layout="wide", page_title="Trợ lý Pháp lý Đất đai")

st.title("🏛️ Trợ lý Pháp lý Thông minh về Luật Đất đai")
st.write("Hỏi đáp, tra cứu và so sánh về Luật Đất đai 2013 và 2024.")

# Khởi tạo session state để quản lý trạng thái
if 'qa_query_count' not in st.session_state:
    st.session_state['qa_query_count'] = 0
if 'comp_query_count' not in st.session_state:
    st.session_state['comp_query_count'] = 0

# Tạo 2 tab cho 2 chức năng chính
tab1, tab2 = st.tabs(["❓ Hỏi-Đáp & Tra cứu", "⚖️ So sánh Luật"])

# --- XỬ LÝ TAB 1: HỎI-ĐÁP TÌNH HUỐNG ---
with tab1:
    st.header("Đặt câu hỏi hoặc tra cứu theo từ khóa")
    
    with st.form(key="qa_form"):
        user_query_qa = st.text_input("Nhập câu hỏi của bạn vào đây:", key="qa_input_box", placeholder="Ví dụ: Hạn mức nhận chuyển nhượng đất nông nghiệp là bao nhiêu?")
        submit_button_qa = st.form_submit_button(label="🔍 Gửi câu hỏi")

    if submit_button_qa and user_query_qa:
        # Tăng bộ đếm để tạo key duy nhất cho các nút phản hồi
        st.session_state['qa_query_count'] += 1
        
        with st.spinner("🧠 Đang phân tích và tìm kiếm trong cơ sở tri thức..."):
            cleaned_query = clean_query(user_query_qa)
            st.info(f"Đang tìm kiếm cho: '{cleaned_query}'")
            
            retrieved_docs = retrieval_pipeline(cleaned_query, initial_k=20, final_k=5)
            
            context = ""
            if not retrieved_docs:
                 st.warning("Không tìm thấy điều luật nào có liên quan.")
            else:
                for doc in retrieved_docs:
                    doc_info = f"Trích dẫn từ Điều {doc['ma_dieu']} Luật Đất đai {int(float(doc['phien_ban']))}"
                    context += f"--- {doc_info} ---\n{doc['raw_content']}\n\n"

            if context:
                final_prompt = build_qa_prompt(user_query_qa, context)
                try:
                    final_answer = LLMClient().generate(final_prompt)
                    
                    st.markdown("### 📝 Câu trả lời:")
                    st.markdown(final_answer)

                    # --- PHẦN PHẢN HỒI NGƯỜI DÙNG ---
                    st.write("")
                    feedback_key = f"feedback_qa_{st.session_state['qa_query_count']}"
                    if feedback_key not in st.session_state:
                        st.session_state[feedback_key] = None
                    
                    col1, col2, _ = st.columns([1, 1, 8])
                    if col1.button("👍 Hữu ích", key=f"up_{feedback_key}"):
                        st.session_state[feedback_key] = "positive"
                    if col2.button("👎 Không hữu ích", key=f"down_{feedback_key}"):
                        st.session_state[feedback_key] = "negative"
                    
                    if st.session_state[feedback_key] == "positive":
                        st.success("Cảm ơn bạn đã đánh giá!")
                    elif st.session_state[feedback_key] == "negative":
                        st.warning("Cảm ơn bạn đã đánh giá! Chúng tôi sẽ xem xét để cải thiện câu trả lời này.")
                    # --- KẾT THÚC PHẦN PHẢN HỒI ---

                    with st.expander("🔍 Xem các điều luật liên quan nhất đã được sử dụng"):
                        for doc in retrieved_docs:
                            st.markdown(f"**Điều {doc['ma_dieu']} Luật Đất đai {int(float(doc['phien_ban']))} (Điểm liên quan: {doc.get('rerank_score'):.4f})**")
                            st.text(doc['raw_content'])
                except Exception as e:
                    st.error(f"Đã có lỗi xảy ra khi gọi đến mô hình ngôn ngữ: {e}")
            else:
                # Thông báo này được hiển thị khi retriever không tìm thấy gì
                st.error("Không thể xây dựng ngữ cảnh từ các điều luật truy xuất được.")

# --- XỬ LÝ TAB 2: SO SÁNH LUẬT ---
with tab2:
    st.header("So sánh sự khác biệt giữa Luật 2013 và 2024")

    with st.form(key="compare_form"):
        comparison_query = st.text_input("Nhập chủ đề bạn muốn so sánh:", key="compare_input_box", placeholder="Ví dụ: So sánh quy định về thu hồi đất để phát triển kinh tế - xã hội")
        submit_button_comp = st.form_submit_button(label="⚖️ So sánh")

    if submit_button_comp and comparison_query:
        # Tăng bộ đếm để tạo key duy nhất
        st.session_state['comp_query_count'] += 1

        with st.spinner("⚖️ Đang đối chiếu các phiên bản luật..."):
            cleaned_query = clean_query(comparison_query)
            st.info(f"Đang tìm kiếm cho: '{cleaned_query}'")

            retrieved_docs = retrieval_pipeline(cleaned_query, initial_k=30, final_k=5)
            
            context = ""
            if not retrieved_docs:
                st.warning("Không tìm thấy điều luật nào liên quan đến chủ đề này.")
            else:
                for doc in retrieved_docs:
                    doc_info = f"Trích dẫn từ Điều {doc['ma_dieu']} Luật Đất đai {int(float(doc['phien_ban']))}"
                    context += f"--- {doc_info} ---\n{doc['raw_content']}\n\n"

            if context:
                final_prompt = build_comparison_prompt(comparison_query, context)
                try:
                    final_answer = LLMClient().generate(final_prompt)
                    st.markdown("### 📊 Bài phân tích so sánh:")
                    st.markdown(final_answer)

                    # --- PHẦN PHẢN HỒI NGƯỜI DÙNG ---
                    st.write("")
                    feedback_key = f"feedback_comp_{st.session_state['comp_query_count']}"
                    if feedback_key not in st.session_state:
                        st.session_state[feedback_key] = None
                    
                    col3, col4, _ = st.columns([1, 1, 8])
                    if col3.button("👍 Hữu ích", key=f"up_{feedback_key}"):
                        st.session_state[feedback_key] = "positive"
                    if col4.button("👎 Không hữu ích", key=f"down_{feedback_key}"):
                        st.session_state[feedback_key] = "negative"
                    
                    if st.session_state[feedback_key] == "positive":
                        st.success("Cảm ơn bạn đã đánh giá!")
                    elif st.session_state[feedback_key] == "negative":
                        st.warning("Cảm ơn bạn đã đánh giá!")
                    # --- KẾT THÚC PHẦN PHẢN HỒI ---

                    with st.expander("🔍 Xem các điều luật liên quan đã được sử dụng để so sánh"):
                        for doc in retrieved_docs:
                            st.markdown(f"**Điều {doc['ma_dieu']} Luật Đất đai {int(float(doc['phien_ban']))} (Điểm liên quan: {doc.get('rerank_score'):.4f})**")
                            st.text(doc['raw_content'])
                except Exception as e:
                    st.error(f"Đã có lỗi xảy ra khi gọi đến mô hình ngôn ngữ: {e}")