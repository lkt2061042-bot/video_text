import streamlit as st
import cv2
import numpy as np
import pytesseract
import tempfile
import os

st.set_page_config(page_title="LCD 綠色螢幕文字提取工具", layout="centered")
st.title("📟 LCD 綠色螢幕文字精準提取工具")
st.markdown("已針對 LCD 點陣字體進行影像優化（去除網格、二值化處理），大幅提升辨識準確率。")

uploaded_file = st.file_uploader("選擇影片檔案 (支援 MP4, AVI, MOV)", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    if st.button("🚀 開始精準擷取文字"):
        with st.spinner("正在進行高精度 LCD 畫面解析，請稍候..."):
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
                
                # 1. 轉換成 HSV 尋找綠色螢幕區域
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                lower_green = np.array([35, 30, 30])
                upper_green = np.array([85, 255, 255])
                mask = cv2.inRange(hsv, lower_green, upper_green)
                
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                text = ""
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(largest_contour) > 1000:
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        
                        # 2. 裁切出綠色螢幕範圍
                        roi = frame[y:y+h, x:x+w]
                        
                        # 3. 針對 LCD 點陣螢幕的高級前置處理
                        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        
                        # 放大 3 倍以利 OCR 辨識小字
                        gray = cv2.resize(gray, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                        
                        # 去除微小噪點並平滑邊緣
                        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
                        
                        # Otsu 二值化：將背景轉為純白、文字轉為純黑，去除液晶網格線
                        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        
                        # 4. 辨識文字 (設定 psm 6 視為統一的文字區塊)
                        text = pytesseract.image_to_string(thresh, config='--psm 6').strip()
                        
                        # 清理特殊符號與多餘空白
                        text = " ".join(text.split())

                # 過濾連續重複的文字
                if text and text != last_text and len(text) > 2: # 濾掉太短的誤判雜訊
                    extracted_texts.append(text)
                    last_text = text

            cap.release()
            progress_bar.empty()

            if extracted_texts:
                st.success(f"處理完成！總共提取出 {len(extracted_texts)} 筆清晰的螢幕文字。")
                
                txt_content = "\n".join(extracted_texts)
                
                st.text_area("識別結果預覽：", txt_content, height=250)
                
                st.download_button(
                    label="📥 下載精準 TXT 清單",
                    data=txt_content,
                    file_name="lcd_accurate_text.txt",
                    mime="text/plain"
                )
            else:
                st.warning("未能準確抓取文字，請確認影片中的綠色螢幕是否夠清晰或角度是否正確。")
