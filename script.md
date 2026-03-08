#### **Slide 1: Trang bìa**

*   **Nội dung cần nói:**
    > "Kính chào thầy và cả lớp.
    >
    > Hôm nay em sẽ đại diện nhóm 7, trình bày báo cáo đồ án cuối kỳ, với đề tài: **Xây Dựng Hệ Thống Hỏi Đáp và So Sánh Luật Đất Đai Dựa Trên Kiến Trúc RAG Tăng Cường bằng Đồ Thị Tri Thức.**
    Nhóm em gồm 4 thành viên là: Em,Văn Đức Ngọ, bạn Phạm Thăng Long, anh Võ Lê Phú Xuân, và bạn Nguyễn Hoàng Hải.
    >

#### **Slide 2: Overview**

*   **Nội dung cần nói:**
    > "Bài trình bày của nhóm em hôm nay sẽ đi xuyên suốt từ  **Bối cảnh và Vấn đề**, **Mục tiêu và Giải pháp**, **Kiến trúc Hệ thống**, các **Kỹ thuật Cốt lõi**, **Đánh giá Hiệu quả**, **Kết luận và Hướng phát triển** và **Demo** 

---

#### **Slide 3 & 4: Bối Cảnh & Vấn Đề**

*   **Nội dung cần nói:**
    > (Slide 3) "Đầu tiên, chúng ta hãy cùng đến với phần Bối cảnh và Vấn đề."
    >
    > (Slide 4) "**Bối cảnh** của dự án bắt nguồn từ việc Luật Đất đai 2024 ra đời, thay thế cho phiên bản 2013 với rất nhiều thay đổi đột phá, tạo ra một 'khoảng trống thông tin' rất lớn.
    >
    > Từ đó, nảy sinh hai **vấn đề** chính:
    >
    > *   **Thứ nhất, đối với người dùng,**  việc tiếp cận, hiểu và so sánh hai bộ luật này là cực kỳ khó khăn do ngôn ngữ pháp lý phức tạp và khối lượng thông tin khổng lồ.
    > *   **Thứ hai, các công cụ tìm kiếm truyền thống,** vốn chỉ dựa trên từ khóa, tỏ ra không hiệu quả. Chúng không thể hiểu được ngữ nghĩa hay các mối quan hệ logic phức tạp ẩn sau các điều khoản."

---

#### **Slide 5: Mục Tiêu Dự Án**

*   **Nội dung cần nói:**
    > "Để giải quyết các vấn đề trên, nhóm đã đặt ra mục tiêu là **Xây dựng một Trợ lý Pháp lý Thông minh** chuyên sâu về Luật Đất đai 2013 và 2024.
    >
    > Hệ thống này phải đáp ứng được 3 yêu cầu cốt lõi:
    >
    > 1.  **Hỏi-Đáp Tình huống:** Trả lời chính xác và dễ hiểu các câu hỏi phức tạp.
    > 2.  **So sánh Chuyên sâu:** Tự động đối chiếu và làm nổi bật sự khác biệt giữa hai phiên bản luật.
    > 3.  Và quan trọng nhất, **Đảm bảo Tính Tin cậy:** Mọi câu trả lời đều phải có trích dẫn nguồn luật rõ ràng để người dùng có thể kiểm chứng."

---

#### **Slide 6 & 7: Thách Thức Cần Giải Quyết & Vấn Đề Của Trí Tuệ Nhân Tạo**

*   **Nội dung cần nói:**
    > (Slide 6) "Vậy, thách thức công nghệ chính mà chúng ta cần giải quyết là gì?"
    >
    > (Slide 7) "Với sự phát triển của Trí tuệ Nhân tạo, **cơ hội** đã mở ra. Các Mô hình Ngôn ngữ Lớn (LLM) có tiềm năng trở thành các trợ lý pháp lý ảo.
    >
    > Tuy nhiên, chúng ta đối mặt với một **thách thức lớn nhất**, đó là hiện tượng **'ảo giác' (Hallucination)**. LLM có thể 'sáng tạo' ra các thông tin sai lệch. Trong lĩnh vực pháp lý, nơi mà tính chính xác và căn cứ là yêu cầu tối thượng, điều này là không thể chấp nhận được.
    >
    > Do đó, chúng ta có thể kết luận rằng: **Việc áp dụng LLM thuần là quá rủi ro.**"

---

#### **Slide 8 & 9: Giải Pháp & Kiến Trúc**

*   **Nội dung cần nói:**
    > (Slide 8) "Để giải quyết vấn đề 'ảo giác', giải pháp được nhóm lựa chọn là kiến trúc **RAG - Retrieval-Augmented Generation.**
    >
    > *   Về cơ bản, RAG là một kiến trúc nền tảng, giúp LLM trả lời dựa trên một nguồn kiến thức tin cậy được cung cấp, thay vì dựa vào "trí nhớ" của chính nó.
    >
    > Tuy nhiên, để giải quyết bài toán pháp lý một cách triệt để, nhóm đề xuất một kiến trúc nâng cao: **KG-RAG**.
    >
    > *   Kiến trúc này **Tăng cường RAG bằng Đồ thị Tri thức (Knowledge Graph - KG)** và xây dựng một **Nền tảng Tri thức Lai (Hybrid Knowledge Base)** để khai thác cả tri thức có cấu trúc và phi cấu trúc."
    >
    > (Slide 9) "Nền tảng tri thức lai của chúng em bao gồm hai thành phần bổ trợ cho nhau:
    >
    > *   **Đồ thị Tri thức (KG) bằng Neo4j, đóng vai trò là "BỘ NÃO" của hệ thống.** Nó lưu trữ tri thức có cấu trúc, biểu diễn các thực thể như Điều luật, Khái niệm và các mối quan hệ logic như `SỬA_ĐỔI`, `THAY_THẾ`. Điều này cho phép hệ thống có thể thực hiện các truy vấn suy luận logic phức tạp.
    > Và thành phần thứ hai là *   **Cơ sở dữ liệu Vector bằng FAISS, đóng vai trò là "TRÍ NHỚ" của hệ thống.** Nó lưu trữ tri thức phi cấu trúc dưới dạng vector, cho phép tìm kiếm ngữ nghĩa (semantic search) với tốc độ cao."

---

#### **Slide 10-13: Kiến trúc Hệ thống & Các Giai đoạn**

*   **Nội dung cần nói:**
    > (Slide 10-11) "Về kiến trúc hệ thống, Toàn bộ hệ thống hoạt động theo 2 giai đoạn độc lập: Giai đoạn Offline và Giai đoạn Online."
    >
    > (Slide 12) "**Giai đoạn Offline** là quá trình xây dựng Cơ sở Tri thức, được thực hiện một lần.
    >
    > * Bắt đầu từ 2 file PDF tương ứng cho hai bộ luật, chúng em thực hiện **Tiền xử lý** để tách luật thành các file text theo từng điều luật.
    > *   Sau đó, dùng **LLM** để tự động **Trích xuất** các Thực thể và Quan hệ thành dữ liệu JSON.
    > *   Dữ liệu thô này sau đó được **Chuẩn hóa** để làm sạch và hợp nhất.
    *   Cuối cùng, chúng em **Xây dựng đồ thị tri thức** bằng cách import vào Neo4j và **Xây dựng Vector DB** bằng cách tạo embedding và lưu vào FAISS."
    >
    > (Slide 13) "**Giai đoạn Online** là pipeline xử lý truy vấn theo thời gian thực.
    >
    > *   (Chỉ vào sơ đồ) Khi người dùng đặt câu hỏi, hệ thống sẽ **Phân loại và Làm sạch** câu hỏi.
    > *   Câu hỏi được đưa vào **Pipeline Truy xuất (Search-then-Rerank)** để tìm ra 5 ngữ cảnh chính xác nhất.
    > *   Tiếp theo, hệ thống **Xây dựng một Prompt hoàn chỉnh**.
    > *   Prompt được gửi đến LLM để **Sinh ra câu trả lời**.
    > *   Cuối cùng, câu trả lời và các trích dẫn được **Hiển thị** cho người dùng."

---

#### **Slide 14-19: Các Kỹ thuật Cốt lõi & Trực quan hóa**

*   **Nội dung cần nói:**
    > (Slide 14-15) "Để làm được điều này, nhóm đã tập trung vào hai Kỹ thuật Cốt lõi. **Kỹ thuật thứ nhất là Truy xuất Hai Giai đoạn (Search-then-Rerank).**
    > *   **Semantic Search** (dùng Bi-Encoder) hoạt động như một tấm lưới lớn, nhanh chóng lấy được lên 20 điều luật có khả năng liên quan, ưu tiên tốc độ.
    > *   **Reranking** (dùng Cross-Encoder) hoạt động như một chuyên gia, phân tích sâu 20 kết quả đó để chọn ra 5 chính xác nhất, ưu tiên "Độ chính xác" (Precision)."
    >
    > (Slide 16) "**Kỹ thuật thứ hai là Prompt Engineering.** Vấn đề là, cung cấp ngữ cảnh tốt thôi chưa đủ, cần phải 'hướng dẫn' LLM suy luận. Giải pháp của chúng em là tích hợp các kỹ thuật vào prompt như: **Phân vai**, **Suy luận từng bước (Chain-of-Thought)** và **Ràng buộc Nghiêm ngặt**." để đặt LLM vào vai trò là chuyên gia pháp lý, buộc LLM sũy nghĩ có cấu trúc và ngăn chặn các hành phi không mong muốn.
    >
    > (Slide 17-19) "Và đây là kết quả của việc xây dựng KG, minh chứng cho sức mạnh của nó.
    > *   (Chỉ vào hình 1, slide 17) Sơ đồ này minh họa cho **"Cầu nối tri thức"**, cho thấy các điều luật của hai phiên bản được liên kết với nhau.
    > *   (Chỉ vào hình 2, slide 18) Sơ đồ này minh họa cho sự **"Hội tụ khái niệm"**, khi nhiều điều luật cùng quy chiếu về một khái niệm duy nhất, ví dụ như "Thu hồi đất".
    > *   (Chỉ vào hình 3, slide 19) Và cuối cùng, sơ đồ này cho thấy sự **"Phân rã cấu trúc"** của một điều luật thành một hệ sinh thái các thực thể pháp lý liên quan." như điều luật, hanh vi pháp lý, khái niệm, chủ thể...

---

#### **Slide 20-25: Thí nghiệm & Đánh giá**

*   **Nội dung cần nói:**
    > (Slide 20-21) "Để đo lường hiệu quả, nhóm đã tiến hành một thí nghiệm. Chúng em đã so sánh **Hệ thống RAG-KG** của mình với một **LLM thuần (Pure LLM - Gemini 2.0)** trên một bộ dữ liệu gồm **100 câu hỏi tình huống thực tế**." với các chỉ số đo lường là: Độ liên quan (Câu trả lời có tập trung đúng vào câu hỏi không?) và độ chính xác (Nội dung câu trả lời so với đáp án chuẩn.)
    >
    > (Slide 22-23) "**Và đây là kết quả.**
    > *   (Chỉ vào biểu đồ 1a) Về **Độ liên quan và Độ chính xác**, hệ thống RAG-KG của chúng em (cột màu xanh) đã vượt trội hơn hẳn so với LLM thuần (cột màu cam).
    > *   (Chỉ vào biểu đồ 1b) Quan trọng hơn, về **hành vi lỗi**, LLM thuần đã trả lời **sai hoàn toàn 22 lần** so với 12 lần của RAG-KG. Trong khi đó, hệ thống RAG-KG, trong 16 trường hợp không tìm được ngữ cảnh, đã chủ động **từ chối trả lời**. Đây là một hành vi an toàn, "biết mình không biết", thay vì "đoán mò" như LLM thuần."
    >
    > (Slide 24) "Một ca điển hình là với câu hỏi "Kỳ quy hoạch sử dụng đất cấp huyện theo LĐĐ 2024 là bao nhiêu năm?", LLM thuần đã "đoán mò" và trả lời sai là "5 năm", trong khi hệ thống RAG-KG đã truy xuất đúng Điều 62 và trả lời chính xác là "10 năm"."
    >
    > (Slide 25) "Qua đó, chúng em có thể thảo luận rằng: Kiến trúc RAG-KG được thiết kế tốt đã vượt trội hơn hẳn. Việc cung cấp ngữ cảnh chính xác là cơ chế hiệu quả để kiềm chế ảo giác. Và hành vi từ chối trả lời là một tính năng an toàn, cần thiết cho lĩnh vực pháp lý."

---

#### **Slide 26-31: Kết luận, Hướng phát triển & Demo**

*   **Nội dung cần nói:**
    > (Slide 26) "Cuối cùng, em xin đi đến phần Kết luận và Hướng phát triển."
    >
    > (Slide 27-28) "**Về Kết luận,** qua dự án này, nhóm đã thành công trong việc: Xây dựng pipeline bán tự động để chuyển đổi văn bản luật thành KG, triển khai và xác thực hiệu quả của pipeline truy xuất hai giai đoạn, và khẳng định vai trò quyết định của Kỹ thuật Prompt Engineering.
    > **Về Hạn chế,** Tuy nhiên, hệ thống vẫn còn một số điểm có thể cải thiện như: Giới hạn tri thức chỉ trong 2 bộ luật gốc, các văn phản dưới luật như Thông Tư, Nghị định vẫn chưa được đưa vào hệ thống. Việc truy xuất lai vẫn còn đơn giản, không khai thác được tối đa khả năng suy luận logic của KG.Chất lượng Chunking theo điều luật không hiệu quả nếu tìm kiểm theo các khái niệm nhỏ., và hiệu quả của hệ thống vẫn phụ thuộc nhiều vào chất lượng của Prompt"
    >
    > (Slide 29) "Do đó, **Hướng phát triển trong tương lai** của nhóm bao gồm: Mở rộng cơ sở tri thức, Tối ưu pipeline truy xuất lai, Xây dựng chiến lược Chunking thông minh hơn và Cải thiện tương tác người dùng."
    >
    > (Slide 30-31) "Bài trình bày của nhóm em đến đây là kết thúc. Em xin chân thành cảm ơn thầy và các bạn đã lắng nghe. Tiếp theo sẽ là phần Demo sản phẩm của nhóm."