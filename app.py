import streamlit as st
import cv2
import numpy as np
import pytesseract
import tempfile
import os

st.set_page_config(page_title="LCD 綠色螢幕文字提取工具", layout="centered")
st.title("📟 影片綠色 LCD 螢幕文字提取工具")
st.markdown("請上傳影片，系統將會自動鎖定並裁切畫面中的**綠色液晶螢幕**區域，精準擷取其中的文字並去重。")

uploaded_file = st.file_uploader("選擇影片檔案 (支援 MP4, AVI, MOV)", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    if st.button("🚀 開始擷取螢幕文字"):
        with st.spinner("正在分析影片並尋找綠色螢幕，請稍候..."):
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
                
                # 1. 轉換成 HSV 色彩空間來精準捕捉綠色螢幕
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                # 設定綠色螢幕的色相、彩度、明度範圍
                lower_green = np.array([35, 40, 40])
                upper_green = np.array([85, 255, 255])
                mask = cv2.inRange(hsv, lower_green, upper_green)
                
                # 2. 尋找畫面中最大的綠色區塊（即 LCD 螢幕位置）
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                text = ""
                if contours:
                    # 找出面積最大且合理的輪廓作為螢幕
                    largest_contour = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(largest_contour) > 1000: # 濾掉太小的雜訊
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        
                        # 3. 裁切出綠色螢幕範圍
                        roi = frame[y:y+h, x:x+w]
                        
                        # 4. 對裁切區進行灰階化與對比度增強，提升 OCR 準確率
                        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        gray_roi = cv2.resize(gray_roi, (0, 0), fx=2, fy=2) # 放大以利辨識
                        
                        # 5. 辨識螢幕文字
                        text = pytesseract.image_to_string(gray_roi, config='--psm 6').strip()
                        text = " ".join(text.split())

                # 過濾連續重複的文字
                if text and text != last_text:
                    extracted_texts.append(text)
                    last_text = text

            cap.release()
            progress_bar.empty()

            if extracted_texts:
                st.success(f"處理完成！總共提取出 {len(extracted_texts)} 筆不重複的螢幕文字紀錄。")
                
                txt_content = "\n".join(extracted_texts)
                
                st.text_area("識別結果預覽：", txt_content, height=250)
                
                st.download_button(
                    label="📥 下載 TXT 文字清單",
                    data=txt_content,
                    file_name="lcd_screen_text.txt",
                    mime="text/plain"
                )
            else:
                st.warning("未能從影片中偵測到綠色螢幕或螢幕無文字，請確認影片角度或畫質。")
