### **BÁO CÁO CUỐI KỲ**

**Tên dự án: Xây Dựng Hệ Thống Hỏi Đáp và So Sánh Luật Đất Đai Dựa Trên Kiến Trúc RAG Tăng Cường bằng Đồ Thị Tri Thức**

---

#### **Tóm Tắt (Abstract)**

Luật Đất đai, với những thay đổi quan trọng giữa hai phiên bản 2013 và 2024, đặt ra một thách thức lớn cho cả người dân và các chuyên gia trong việc nắm bắt, tra cứu và so sánh thông tin. Dự án này trình bày một giải pháp toàn diện để giải quyết vấn đề trên bằng cách xây dựng một hệ thống Trợ lý Pháp lý Thông minh. Hệ thống ứng dụng kiến trúc tiên tiến **Retrieval-Augmented Generation (RAG)**, kết hợp sức mạnh của các **Mô hình Ngôn ngữ Lớn (LLM)** với một nền tảng tri thức có cấu trúc kép: **(1) một Đồ thị Tri thức (Knowledge Graph - KG)** biểu diễn các thực thể và mối quan hệ pháp lý phức tạp, và **(2) một Cơ sở dữ liệu Vector (Vector Database)** cho phép tìm kiếm ngữ nghĩa nhanh chóng. Báo cáo sẽ trình bày chi tiết toàn bộ quy trình, từ tiền xử lý dữ liệu thô, xây dựng Đồ thị Tri thức bán tự động, triển khai pipeline truy xuất hai giai đoạn (**Search-then-Rerank**), đến kỹ thuật thiết kế prompt (**Prompt Engineering**) để đảm bảo câu trả lời được sinh ra là chính xác, có căn cứ và phù hợp với ngữ cảnh.

---

#### **1. Giới Thiệu (Introduction)**

##### **1.1. Bối cảnh và Vấn đề**
Việc ban hành Luật Đất đai 2024, thay thế cho phiên bản 2013, đã tạo ra một nhu cầu cấp thiết trong xã hội về việc hiểu rõ các điểm mới, so sánh sự khác biệt và tra cứu các tình huống pháp lý cụ thể. Tuy nhiên, văn bản pháp luật có đặc thù ngôn ngữ phức tạp, cấu trúc chặt chẽ và các mối quan hệ tham chiếu chéo giữa các điều khoản, gây khó khăn cho việc tiếp cận của người không chuyên. Các hệ thống hỏi-đáp truyền thống dựa trên từ khóa (keyword-based) thường thất bại trong việc nắm bắt ngữ nghĩa và các mối liên hệ phức tạp này.

Trong bối cảnh đó, các Mô hình Ngôn ngữ Lớn (LLM) mở ra một tiềm năng to lớn trong việc hiểu và sinh ngôn ngữ tự nhiên. Tuy nhiên, việc sử dụng LLM một cách độc lập (LLM thuần) đi kèm với rủi ro cố hữu là "ảo giác" (hallucination) - bịa đặt thông tin không có trong nguồn dữ liệu, một điều không thể chấp nhận được trong lĩnh vực pháp lý đòi hỏi tính chính xác tuyệt đối.

##### **1.2. Mục tiêu Dự án**
Dự án đặt ra mục tiêu phát triển một hệ thống "Trợ lý Pháp lý Thông minh" có khả năng:
*   **Hỏi-đáp:** Trả lời chính xác các câu hỏi tình huống dựa trên nội dung của Luật Đất đai 2013 và 2024.
*   **So sánh:** Phân tích và chỉ ra các điểm khác biệt cốt lõi về một chủ đề cụ thể giữa hai phiên bản luật.
*   **Đáng tin cậy:** Mọi câu trả lời đều phải có căn cứ, đi kèm trích dẫn Điều/Khoản luật cụ thể.

##### **1.3. Phương pháp Tiếp cận**
Để đạt được các mục tiêu trên, chúng tôi đề xuất một kiến trúc **Retrieval-Augmented Generation (RAG)** nâng cao. Thay vì chỉ dựa vào tìm kiếm văn bản đơn thuần, hệ thống được tăng cường bởi một **Đồ thị Tri thức (Knowledge Graph - KG)** được xây dựng chuyên biệt cho lĩnh vực Luật Đất đai, cho phép truy xuất thông tin không chỉ dựa trên sự tương đồng ngữ nghĩa mà còn dựa trên các mối quan hệ pháp lý có cấu trúc.

---

#### **2. Kiến Trúc và Quy Trình Hệ Thống**

Hệ thống được chia thành hai giai đoạn chính: **Xây dựng Cơ sở Tri thức (Offline)** và **Xử lý Truy vấn (Online)**.

##### **2.1. Sơ đồ luồng Tổng thể**
*(So do o day)*

##### **2.2. Giai đoạn Offline: Xây dựng Cơ sở Tri thức**
Đây là giai đoạn nền tảng, thực hiện một lần để xây dựng toàn bộ "bộ não" cho hệ thống.

1.  **Tiền xử lý và Chunking:** Văn bản luật từ hai file PDF gốc được chuyển đổi sang định dạng text. Sau đó, văn bản được làm sạch và chia nhỏ (chunking) thành các đơn vị ngữ nghĩa cơ bản, mỗi đơn vị là một Điều luật, lưu thành các file `.txt` riêng biệt.

2.  **Trích xuất Tri thức bằng LLM (LLM-based Knowledge Extraction):** Chúng tôi áp dụng phương pháp bán tự động để xây dựng Đồ thị Tri thức.
    *   **Trích xuất Thực thể & Quan hệ Nội bộ:** Một prompt chi tiết được thiết kế để yêu cầu LLM (Google Gemini) đọc từng Điều luật và trích xuất các thực thể (`ĐiềuLuật`, `KháiNiệm`, `ChủThể`, `HànhViPhápLý`, `ChếTài`, `ĐiềuKiện`) và các mối quan hệ bên trong điều luật đó (`QUY_ĐỊNH_VỀ`, `ÁP_DỤNG_CHO`,...).
    *   **Trích xuất Quan hệ So sánh:** Một prompt thứ hai được sử dụng để yêu cầu LLM so sánh từng Điều của Luật 2024 với toàn bộ văn bản Luật 2013, từ đó xác định Điều luật tương ứng và loại thay đổi (`SỬA_ĐỔI_BỔ_SUNG`, `THAY_THẾ_HOÀN_TOÀN`, `ĐIỀU_LUẬT_MỚI`).

3.  **Chuẩn hóa và Hợp nhất Dữ liệu:** Dữ liệu JSON thô từ LLM chứa nhiều điểm không nhất quán (trùng lặp, lỗi chính tả). Một quy trình chuẩn hóa được thực hiện:
    *   **Tạo Từ điển Đồng nghĩa:** Một script (`helper_create_synonym_list.py`) được sử dụng để tổng hợp tất cả các thực thể, giúp người phát triển rà soát và tạo ra một danh sách các nhóm thực thể đồng nghĩa (`synonym_groups`).
    *   **Hợp nhất:** Một script khác (`normalize_and_merge_graph.py`) sử dụng từ điển này để hợp nhất các nút trùng lặp, chuẩn hóa ID và `label`, đảm bảo mỗi thực thể trong thế giới thực chỉ tương ứng với một nút duy nhất trong đồ thị.

4.  **Xây dựng Đồ thị Tri thức (KG):** Dữ liệu đã chuẩn hóa được import vào **Neo4j** bằng công cụ `neo4j-admin database import`. Kết quả là một Đồ thị Tri thức thống nhất, biểu diễn cả hai bộ luật và các mối quan hệ so sánh giữa chúng. Một **Full-Text Index** cũng được tạo trên thuộc tính `name` và `noi_dung` của các nút `DieuLuat` để tối ưu hóa tìm kiếm từ khóa.

5.  **Xây dựng Cơ sở dữ liệu Vector (Vector DB):**
    *   Dữ liệu nội dung của tất cả các điều luật được truy vấn từ Neo4j.
    *   Sử dụng mô hình embedding Bi-Encoder chuyên cho tiếng Việt (`bkai-foundation-models/vietnamese-bi-encoder`), chúng tôi chuyển đổi mỗi điều luật thành một vector ngữ nghĩa.
    *   Tất cả các vector được lưu trữ và lập chỉ mục trong **FAISS** (`IndexFlatIP`) để phục vụ cho việc tìm kiếm tương đồng ngữ nghĩa tốc độ cao.

##### **2.3. Giai đoạn Online: Xử lý Truy vấn**

1.  **Làm sạch Câu hỏi:** Câu hỏi của người dùng được xử lý sơ bộ để loại bỏ các từ/ký tự nhiễu.
2.  **Pipeline Truy xuất Hai Giai đoạn (Two-Stage Retrieval):**
    *   **Giai đoạn 1 - Candidate Retrieval:** Câu hỏi được nhúng thành vector và tìm kiếm trong FAISS để lấy ra một tập hợp lớn các ứng viên tiềm năng (ví dụ: top 20 điều luật).
    *   **Giai đoạn 2 - Reranking:** Các ứng viên này sau đó được đánh giá lại bằng một mô hình Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`). Mô hình này so sánh trực tiếp cặp `(câu hỏi, nội dung điều luật)` để đưa ra điểm số chính xác hơn. Top 5 điều luật có điểm số cao nhất sẽ được chọn làm ngữ cảnh cuối cùng.
3.  **Xây dựng Prompt và Sinh Câu trả lời:**
    *   Nội dung của 5 điều luật liên quan nhất được định dạng thành một chuỗi ngữ cảnh.
    *   Ngữ cảnh này, cùng với câu hỏi gốc, được đưa vào một **prompt được thiết kế chuyên biệt (Prompt Engineering)**. Prompt này hướng dẫn LLM thực hiện suy luận từng bước (Chain-of-Thought) và bắt buộc phải trích dẫn nguồn.
    *   LLM (Gemini) sinh ra câu trả lời cuối cùng và được hiển thị cho người dùng.

---

#### **3. Chi Tiết Triển Khai và Thảo Luận**

##### **3.1. Mô hình Biểu diễn Tri thức**
Mô hình Đồ thị Tri thức được chọn có schema gồm 6 loại nút chính và 5 loại cạnh cơ bản. Việc này cho phép biểu diễn các mối quan hệ phức tạp như: `(Điều 81) -[QUY_ĐỊNH_VỀ]-> (Hành vi: Vi phạm pháp luật)` và `(Điều 81) -[DẪN_ĐẾN]-> (Chế tài: Thu hồi đất)`. Đặc biệt, các cạnh so sánh (`SUA_DOI_BO_SUNG`, `THAY_THE_HOAN_TOAN`) là cầu nối, cho phép truy vấn so sánh trực tiếp trên đồ thị.

##### **3.2. Thuật toán Truy xuất**
Pipeline `Search-then-Rerank` là một lựa chọn cân bằng giữa tốc độ và độ chính xác.
*   **Semantic Search (Bi-Encoder):** Đảm bảo tốc độ và khả năng "gợi nhớ" (recall), không bỏ sót các văn bản liên quan.
*   **Reranking (Cross-Encoder):** Đảm bảo độ chính xác (precision) bằng cách sắp xếp lại và ưu tiên những văn bản thực sự phù hợp nhất.
Thử nghiệm cho thấy, với câu hỏi "khi nào thì nhà nước thu hồi đất do vi phạm pháp luật?", Semantic Search có thể xếp `Điều 16 (2013)` lên đầu, nhưng bước Rerank đã đẩy `Điều 81 (2024)` (câu trả lời đúng nhất) lên vị trí số 1.

##### **3.3. Prompt Engineering**
Đây là một trong những khâu quan trọng nhất. Sau nhiều lần thử nghiệm, chúng tôi nhận thấy các prompt đơn giản thường dẫn đến việc LLM "lười biếng" hoặc suy luận sai. Phiên bản prompt cuối cùng áp dụng các kỹ thuật:
*   **Phân vai (Role-playing):** "Bạn là một chuyên gia pháp lý..."
*   **Suy luận từng bước (Chain-of-Thought):** "QUY TRÌNH PHÂN TÍCH BẮT BUỘC..."
*   **Ràng buộc phủ định (Negative Constraints):** "TUYỆT ĐỐI KHÔNG ĐƯỢC kết luận rằng..."
Những kỹ thuật này đã cải thiện đáng kể chất lượng và độ tin cậy của câu trả lời được sinh ra.

---

#### **4. Kết Luận và Hướng Phát Triển**

Dự án đã xây dựng thành công một nguyên mẫu hoạt động của hệ thống Trợ lý Pháp lý Thông minh, giải quyết hiệu quả bài toán hỏi-đáp và so sánh hai phiên bản Luật Đất đai. Việc kết hợp Đồ thị Tri thức và pipeline RAG hai giai đoạn đã chứng tỏ là một hướng đi mạnh mẽ, cân bằng được tốc độ, độ chính xác và khả năng diễn giải của hệ thống.

**Hướng phát triển trong tương lai:**
1.  **Đánh giá định lượng:** Thực hiện so sánh hiệu quả giữa hệ thống RAG đã xây dựng với các LLM thuần (Gemini, GPT-4...) trên bộ 100 câu hỏi tình huống đã thu thập. Các chỉ số đánh giá có thể bao gồm: độ chính xác của câu trả lời, độ chính xác của trích dẫn, và đánh giá của người dùng.
2.  **Mở rộng Cơ sở Tri thức:** Tích hợp thêm các văn bản dưới luật (Nghị định, Thông tư) để làm giàu tri thức, cho phép trả lời các câu hỏi chi tiết hơn.
3.  **Tối ưu Pipeline Truy xuất:** Triển khai các chiến lược truy xuất lai (Hybrid Retrieval) phức tạp hơn, kết hợp điểm số từ Semantic Search, Full-Text Search và các truy vấn đồ thị để có được tập ngữ cảnh tối ưu nhất.
4.  **Cải thiện Giao diện:** Bổ sung các tính năng trực quan hóa đồ thị để người dùng có thể tự khám phá các mối quan hệ giữa các điều luật.

---

#### **Phụ lục: Hướng dẫn Cài đặt và Chạy**

*Huong dan cai dat va chay*