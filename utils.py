import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
import pytesseract
import re
import os
from paddleocr import PaddleOCR

def is_cn_block(text):
    # Đếm số lượng kí tự CN text
    cn_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    chars_count = len(text.strip())
    if chars_count == 0: return False
    
    # Tỉ lệ chữ tiếng Trung > 20% tổng số -> là block tiếng Trung
    return (cn_chars / chars_count) > 0.5

def ocr_layout(pil_img, paddle_ocr):
    cn_text = ""
    vn_text = ""
    
    img = np.array(pil_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Upscale 2 lần  
    #scaled = cv2.resize(img, None, fx=2, fy=2, interpolation= cv2.INTER_CUBIC)  # INTER_CUBIC giữ nét khi phóng to
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Tạo khối
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,10))
    dilated = cv2.dilate(thresh, kernel, iterations = 1)
    
    # Tìm contours (các khung bao quanh văn bản)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    blocks = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 50 and h > 10:
            blocks.append((x,y,w,h))
            
    # Sort lại theo trục y
    blocks.sort(key = lambda x : x[1])
    
    final_result = ""
    
    # Duyệt từng block và xử lý theo tiếng Trung - Viêt
    
    if not os.path.exists('debug_images'):
        os.makedirs('debug_images')
    
    for i, (x,y,w,h) in enumerate(blocks):
        
        h_img, w_img = gray.shape
        margin = 10
        y_min, y_max = max(0, y-margin), min(h_img, y+h+margin)
        x_min, x_max = max(0, x-margin), min(w_img, x+w+margin)
        
        roi_gray = gray[y_min:y_max, x_min:x_max]
        #text_chi = pytesseract.image_to_string(roi, lang='chi_sim', config='--psm 6')
        roi_color = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
        roi_padded = cv2.copyMakeBorder(roi_color, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        
        cv2.imwrite(f'debug_images/roi_block_{i}.jpg', roi_padded)
        
        paddle_result = paddle_ocr.predict(roi_padded)
        
        text_list = []
        
        for res in paddle_result:
            text_list.append(res['rec_texts'])  
            
        with open("debug_text.txt", "a", encoding = 'utf-8') as f:
            f.write(text_list)
        text_chi = "".join(text_list[0])
        
        if is_cn_block(text_chi):
            cn_text += text_chi.strip() + "\n"
        else:
            text_vie = pytesseract.image_to_string(roi_gray, lang='vie', config='--psm 6')
            vn_text += text_vie.strip() + "\n"
            
    return cn_text, vn_text


def is_pinyin(text):
    text = text.lower().strip()

    # Nếu có z trong câu -> từ pinyin
    if 'z' in text:
        return True

    alien_chars = r'[öäüāēīōūǖǎǒǐǔ]'
    if re.search(alien_chars, text):
        return True

    # Kết thúc bằng j, z hoặc bắt đầu bằng f, w, z, j -> là từ pinyin
    if re.search(r'\b\w*[jz]\b', text) or re.search(r'\b[jfwz]\w*\b', text):
        return True
    
    return False


def extract_vn_letters(raw_text):
    lines = raw_text.split('\n')
    letters = []
    current_letter = []
    is_recording = False
    
    vn_char_pattern = r'[đươâêôăạảãậẩẫắằặẳẵệểễộổỗợởỡựửữỳýỵỷỹ]'
    vn_keywords = [' là ', ' và ', ' của ', ' không ', ' có ', ' những ', ' người ', ' cho ', ' dù ', ' hâm ', ' mộ ']

    for line in lines:
        line = line.strip()
        if not line: continue

        # 1. Bắt đầu ghi khi gặp "Bức thư"
        if re.search(r"(?i)bức\s+thư", line):
            if current_letter: 
                letters.append(" ".join(current_letter))
            current_letter = [line]
            is_recording = True
            continue

        if not is_recording: continue

        # Kiểm tra xem có phải pinyin không
        if is_pinyin(line):
            continue

        is_vietnamese = False
        if re.search(vn_char_pattern, line.lower()):
            is_vietnamese = True
        else:
            for word in vn_keywords:
                if word in line.lower():
                    is_vietnamese = True
                    break
        
        if line.isdigit(): is_vietnamese = False

        if is_vietnamese:
            current_letter.append(line)

    if current_letter: letters.append(" ".join(current_letter))
    return letters


def extract_cn_letters(raw_text):
    text = raw_text.replace(r'\n', '\n')
    letters = []
    
    # Header pattern
    pattern = r"(写\s*给\s*自\s*己\s*的\s*第\s*\d+\s*封\s*信)"
    # 信
    parts = re.split(pattern, raw_text)
    
    for i in range(1, len(parts),2):
        header = parts[i].strip()
        if i + 1 < len(parts):
            content = parts[i+1].strip()
        else:
            content = ""
            
        full_letter = header + " " + content.replace("\n", " ")
        letters.append(full_letter)
        
    return letters
