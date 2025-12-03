# Trợ Lý Phân Loại Cảm Xúc Tiếng Việt (Vietnamese Sentiment Assistant)

## 📖 Giới thiệu
Đây là đồ án môn học Seminar Chuyên đề, xây dựng một ứng dụng web đơn giản sử dụng mô hình **Transformer** để phân loại cảm xúc của câu văn tiếng Việt (Tích cực, Tiêu cực, Trung tính).

Ứng dụng được xây dựng bằng **Python**, sử dụng thư viện **Streamlit** cho giao diện và model **DistilBERT** (hoặc PhoBERT) từ Hugging Face.

## 🚀 Tính năng chính
- **Phân loại cảm xúc:** Nhận diện câu tiếng Việt và trả về nhãn POSITIVE, NEGATIVE, hoặc NEUTRAL.
- **Xử lý ngôn ngữ tự nhiên:** Tự động chuẩn hóa văn bản, xử lý các từ viết tắt cơ bản (teencode).
- **Lưu trữ lịch sử:** Lưu lại các câu đã phân tích vào cơ sở dữ liệu SQLite cục bộ.
- **Giao diện thân thiện:** Hiển thị trực quan, dễ sử dụng.

## 🛠 Yêu cầu hệ thống
- Python 3.8 trở lên.
- Kết nối Internet (để tải model lần đầu tiên).

## ⚙️ Cài đặt

1. **Clone hoặc tải source code về máy:**
   Giải nén thư mục đồ án.

2. **Cài đặt các thư viện phụ thuộc:**
   Mở terminal (Command Prompt) tại thư mục dự án và chạy lệnh:
   ```bash
   pip install -r requirements.txt
