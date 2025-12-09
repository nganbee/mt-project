import cv2
import numpy as np
import pytesseract
import re
import regex
import unicodedata
from collections import deque
import pandas as pd

def is_cn_block(text):
    # Đếm số lượng kí tự CN text
    cn_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    chars_count = len(text.strip())
    if chars_count == 0: return False
    
    # Tỉ lệ chữ tiếng Trung > 20% tổng số -> là block tiếng Trung
    return (cn_chars / chars_count) > 0.5

def preprocess_roi(roi_img):
    if roi_img is None or roi_img.size == 0:
        return None
    
    if len(roi_img.shape) == 3:
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi_img
    gray = gray.astype(np.uint8)

    # Nếu ảnh nhỏ sẽ upscale
    if gray.shape[0] < 40:
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    # Shaperning
    kernel = np.array([[-1, -1, -1], 
                       [-1,  9, -1], 
                       [-1, -1, -1]])
    sharp = cv2.filter2D(gray, -1, kernel)
    
    denoised = cv2.fastNlMeansDenoising(sharp, None, 10, 7, 21)

    return denoised

def ocr_layout(pil_img, paddle_ocr, index):
    cn_text = ""
    vn_text = ""
    
    img = np.array(pil_img)
    if len(img.shape) == 3 and img.shape[2] == 3: # Đảm bảo ảnh input đúng hệ màu
         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #thresh_global = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
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
    
    # if not os.path.exists('debug_images'):
    #     os.makedirs('debug_images')
        
    h_img, w_img = img.shape[:2]
    
    # Duyệt từng block và xử lý theo tiếng Trung - Viêt
    for _, (x,y,w,h) in enumerate(blocks):
        margin = 10
        y_min, y_max = max(0, y-(margin+5)), min(h_img, y+h+margin+5)
        x_min, x_max = max(0, x-margin), min(w_img, x+w+margin)
        
        roi = img[y_min:y_max, x_min:x_max]
        
        if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
            continue
        
        processed_roi = preprocess_roi(roi)
        
        # Padding
        roi_padded = cv2.copyMakeBorder(processed_roi, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        roi_input = cv2.cvtColor(roi_padded, cv2.COLOR_GRAY2BGR)
        
        #cv2.imwrite(f'debug_images/roi_block_{index}_{i}.jpg', roi_padded)
        
        paddle_result = paddle_ocr.ocr(roi_input)
        
        text_list = []
        
        for res in paddle_result:
            text_list.append(res['rec_texts'])  
            
        text_chi = "".join(text_list[0])
        
        if is_cn_block(text_chi):
            cn_text += text_chi.strip() + "\n"
        else:
            text_vie = pytesseract.image_to_string(processed_roi, lang='vie', config='--psm 6')
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
    raw_text = unicodedata.normalize('NFC', raw_text)
    
    lines = raw_text.split('\n')
    letters = []
    current_letter = []
    is_recording = False
    
    # Đặt target để extract letter nếu ocr bị thiếu
    target_letter = "bức thư gửi đến chính mình"
    threshold = 90
        
    vn_char_pattern = r'[đươâêôăạảãậẩẫắằặẳẵệểễộổỗợởỡựửữỳýỵỷỹ]'
    vn_keywords = [' là ', ' và ', ' của ', ' không ', ' có ', ' những ', ' người ', ' cho ', ' dù ', ' hâm ', ' mộ ']

    pattern = r"(?i)^[\W_]*(bức\s+thư\s+){e<=1}"

    for line in lines:
        line = line.strip()
        line = unicodedata.normalize('NFC', line)
        if not line: continue

        # 1. Bắt đầu ghi khi gặp "Bức thư"
        #similarity_score = fuzz.partial_ratio(target_letter.lower(), line.lower())
        if regex.search(pattern, line):
            if current_letter: 
                letters.append(" ".join(current_letter))
                
            clean_line = regex.sub(r"^[\W_]+", "", line)
            current_letter = [clean_line]
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




def extract_letters_index(vi_letters, cn_letters, start_num, end_num):
    
    def get_clean_list(text_list):
        # Gom nhóm dữ liệu
        raw_map = {}
        for text in text_list:
            matches = re.findall(r'\d+', text)
            if matches:
                curr_id = int(matches[0])
                if curr_id not in raw_map:
                    raw_map[curr_id] = []
                raw_map[curr_id].append(text)
        
        cleaned_result = {}
        overflow_queue = deque() # Hàng đợi chứa các dòng bị thừa
        
        for i in range(start_num, end_num + 1):
            candidates = raw_map.get(i, [])
            
            final_text = ""
            
            # Lấy dữ liệu tại chỗ
            if candidates:
                final_text = candidates[0] # Lấy cái đầu tiên
                
                # Nếu thừa -> đưa vào hàng
                if len(candidates) > 1:
                    for extra in candidates[1:]:
                        overflow_queue.append(extra)
            
            # Nếu rỗng sẽ điền trống
            if not final_text and overflow_queue:

                final_text = overflow_queue.popleft() 
                
            cleaned_result[i] = final_text
            
        return cleaned_result


    dict_vi = get_clean_list(vi_letters)
    dict_cn = get_clean_list(cn_letters)
    
    rows = []
    for i in range(start_num, end_num + 1):
        rows.append({
            "id": i,
            "vi": dict_vi.get(i, ""),
            "cn": dict_cn.get(i, "")
        })
        
    return pd.DataFrame(rows) 
    
