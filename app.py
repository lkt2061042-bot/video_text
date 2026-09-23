import streamlit as st
import cv2
import pytesseract
import tempfile
import os

st.set_page_title_config = st.title("🎬 影片文字自動提取工具")
st.markdown("請上傳影片，系統將會逐影格識別文字，並自動過濾連續重複的內容，最後產出清單供您下載。")

# 檔案上傳元件
uploaded_file = st.file_uploader("選擇影片檔案 (支援 MP4, AVI, MOV)", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # 將上傳的影片存為暫存檔以便 OpenCV 讀取
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    if st.button("🚀 開始處理影片"):
        with st.spinner("正在逐影格讀取並識別文字，請稍候..."):
            cap = cv2.VideoCapture(video_path)
            extracted_texts = []
            last_text = None
            frame_count = 0
            
            # 建立進度條
            progress_bar = st.progress(0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                if total_frames > 0:
                    progress_bar.progress(min(frame_count / total_frames, 1.0))
                
                # 轉灰階並進行 OCR 識別 (若需繁體中文可改用 lang='chi_tra')
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                text = pytesseract.image_to_string(gray, lang='eng').strip()
                text = " ".join(text.split())
                
                # 去除連續重複文字
                if text and text != last_text:
                    extracted_texts.append(text)
                    last_text = text

            cap.release()
            progress_bar.empty()

            # 顯示結果並提供下載
            if extracted_texts:
                st.success(f"處理完成！總共提取出 {len(extracted_texts)} 筆不重複的文字紀錄。")
                
                txt_content = "\n".join(extracted_texts)
                
                # 預覽結果
                st.text_area("識別結果預覽：", txt_content, height=200)
                
                # 下載按鈕
                st.download_button(
                    label="📥 下載 TXT 文字清單",
                    data=txt_content,
                    file_name="video_text_list.txt",
                    mime="text/plain"
                )
            else:
                st.warning("在影片中未識別到任何文字，請確認影片內容或語言設定。")