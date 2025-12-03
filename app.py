import streamlit as st
from transformers import pipeline
import sqlite3
from datetime import datetime
import pandas as pd
import underthesea  # Dùng để tách từ nếu cần

# --- 1. CẤU HÌNH & LOAD MODEL (NLP ENGINE) ---
# Sử dụng @st.cache_resource để cache model, tránh load lại mỗi khi reload trang (Tối ưu hiệu suất)
@st.cache_resource
def load_sentiment_pipeline():
    # Sử dụng model PhoBERT đã fine-tune cho sentiment để đảm bảo độ chính xác cao
    model_name = "wonrax/phobert-base-vietnamese-sentiment"
    # Nếu máy yếu có thể dùng "distilbert-base-multilingual-cased" nhưng độ chính xác tiếng Việt thấp hơn
    sentiment_task = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)
    return sentiment_task

# Khởi tạo pipeline
try:
    classifier = load_sentiment_pipeline()
except Exception as e:
    st.error(f"Lỗi tải model: {e}")
    st.stop()

# --- 2. CƠ SỞ DỮ LIỆU (SQLITE) ---
DB_NAME = "sentiments.db"

def init_db():
    """Tạo bảng database nếu chưa tồn tại"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS sentiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def save_to_db(text, sentiment):
    """Lưu kết quả phân loại vào DB (Vấn đề kỹ thuật: SQL Injection -> Dùng tham số ?)"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO sentiments (text, sentiment, timestamp) VALUES (?, ?, ?)", 
              (text, sentiment, timestamp))
    conn.commit()
    conn.close()

def get_history():
    """Lấy lịch sử 50 dòng mới nhất"""
    conn = sqlite3.connect(DB_NAME)
    # Load vào DataFrame để hiển thị đẹp hơn trên Streamlit
    df = pd.read_sql_query("SELECT text, sentiment, timestamp FROM sentiments ORDER BY timestamp DESC LIMIT 50", conn)
    conn.close()
    return df

# --- 3. XỬ LÝ LOGIC (PREPROCESSING & MAPPING) ---
def normalize_text(text):
    """Chuẩn hóa văn bản cơ bản (Section VII.1)"""
    if not text: 
        return ""
    text = text.strip().lower()
    
    # Từ điển chuẩn hóa nhỏ (viết tắt -> đầy đủ)
    replace_dict = {
        "rat": "rất",
        "hok": "không",
        "ko": "không",
        "bt": "bình thường",
        "ok": "ổn",
        "wa": "quá"
    }
    
    words = text.split()
    words = [replace_dict.get(w, w) for w in words]
    return " ".join(words)

def map_label(label):
    """Chuyển đổi nhãn của model sang định dạng yêu cầu (POSITIVE, NEGATIVE, NEUTRAL)"""
    # Model wonrax trả về: NEG, POS, NEU
    if label == "POS": return "POSITIVE"
    if label == "NEG": return "NEGATIVE"
    if label == "NEU": return "NEUTRAL"
    return "NEUTRAL" # Mặc định

# --- 4. GIAO DIỆN STREAMLIT (UI) ---

# Khởi tạo DB khi chạy app
init_db()

st.set_page_config(page_title="Trợ lý Cảm xúc Tiếng Việt", page_icon="🤖")

st.title("🤖 Trợ Lý Phân Loại Cảm Xúc Tiếng Việt")
st.markdown("Đồ án môn học: **Xây dựng trợ lý phân loại cảm xúc sử dụng Transformer**")

# Chia cột giao diện
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Nhập liệu")
    user_input = st.text_area("Nhập câu tiếng Việt của bạn:", height=100, placeholder="Ví dụ: Hôm nay tôi rất vui")

    if st.button("Phân loại cảm xúc", type="primary"):
        if not user_input or len(user_input.strip()) < 5:
            st.warning("⚠️ Vui lòng nhập câu dài hơn 5 ký tự!")
        else:
            with st.spinner("Đang phân tích..."):
                # 1. Tiền xử lý
                clean_text = normalize_text(user_input)
                
                # 2. Gọi Pipeline
                # Cắt ngắn nếu quá dài (limit model)
                result = classifier(clean_text[:512])[0] 
                
                # 3. Mapping nhãn
                sentiment_label = map_label(result['label'])
                score = result['score']
                
                # 4. Hiển thị kết quả
                st.success("Đã phân tích xong!")
                
                # Tạo dictionary kết quả như yêu cầu đề bài
                result_dict = {
                    "text": user_input,
                    "sentiment": sentiment_label
                }
                
                st.json(result_dict) # Hiển thị dạng JSON
                
                # Hiển thị UI thân thiện
                if sentiment_label == "POSITIVE":
                    st.info(f"Dự đoán: **TÍCH CỰC** (Độ tin cậy: {score:.2f})")
                elif sentiment_label == "NEGATIVE":
                    st.error(f"Dự đoán: **TIÊU CỰC** (Độ tin cậy: {score:.2f})")
                else:
                    st.warning(f"Dự đoán: **TRUNG TÍNH** (Độ tin cậy: {score:.2f})")
                
                # 5. Lưu vào DB
                save_to_db(user_input, sentiment_label)

with col2:
    st.subheader("Lịch sử phân loại")
    if st.button("Làm mới danh sách"):
        st.rerun()
    
    history_df = get_history()
    if not history_df.empty:
        st.dataframe(history_df, hide_index=True)
    else:
        st.write("Chưa có dữ liệu.")

# --- FOOTER ---
st.markdown("---")
st.caption("Sinh viên thực hiện: Nguyễn Minh Trí - Model: PhoBERT")