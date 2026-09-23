import streamlit as st
import cv2
import numpy as np
import easyocr
import tempfile
import os

st.set_page_config(page_title="LCD 綠色螢幕文字智慧提取工具", layout="centered")
st.title("📟 LCD 綠色螢幕文字智慧提取工具 (強效對比版)")
st.markdown("已加入區域對比度增強（CLAHE），能同時精準讀取正常區域與深色背景反白區域的所有文字。")

@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'])

reader = load_reader()

uploaded_file = st.file_uploader("選擇影片檔案 (支援 MP4, AVI, MOV)", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    if st.button("🚀 開始深度擷取文字"):
        with st.spinner("正在進行 AI 影像增強與文字辨識，請稍候..."):
            cap = cv2.VideoCapture(video_path)
            extracted_texts = []
            last_text = None
            frame_count = 0
            
            progress_bar = st.progress(0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                if total_frames > 0:
                    progress_bar.progress(min(frame_count / total_frames, 1.0))
                
                # 1. 捕捉綠色螢幕範圍
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                lower_green = np.array([30, 20, 20])
                upper_green = np.array([90, 255, 255])
                mask = cv2.inRange(hsv, lower_green, upper_green)
                
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                text_line = ""
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(largest_contour) > 1000:
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        
                        # 2. 裁切出綠色螢幕畫面
                        roi = frame[y:y+h, x:x+w]
                        
                        # 3. 影像前處理與對比度增強 (CLAHE)
                        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        gray = cv2.resize(gray, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                        
                        # 局部對比度增強，解決深色背景與亮字亮度不均的問題
                        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                        enhanced = clahe.apply(gray)
                        
                        # 4. 使用 EasyOCR 辨識增強後的影像
                        results = reader.readtext(enhanced)
                        detected_words = [res[1] for res in results if res[2] > 0.15]
                        text_line = " ".join(detected_words)

                # 過濾連續重複與空值
                if text_line and text_line != last_text:
                    extracted_texts.append(text_line)
                    last_text = text_line

            cap.release()
            progress_bar.empty()

            if extracted_texts:
                st.success(f"處理完成！總共提取出 {len(extracted_texts)} 筆螢幕文字。")
                
                txt_content = "\n".join(extracted_texts)
                
                st.text_area("識別結果預覽：", txt_content, height=250)
                
                st.download_button(
                    label="📥 下載完整 TXT 清單",
                    data=txt_content,
                    file_name="lcd_full_text.txt",
                    mime="text/plain"
                )
            else:
                st.warning("未能從影片中辨識出文字。")
