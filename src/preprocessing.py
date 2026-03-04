"""
Text Preprocessing Module cho Toxic Content Detection

Module này chứa các hàm để làm sạch và normalize văn bản trước khi
đưa vào model. Bao gồm:
    - Xử lý obfuscation (che giấu từ ngữ độc hại)
    - Normalize slang và abbreviations
    - Xử lý ngữ cảnh tích cực (profanity trong câu khen)
    - Xử lý từ đa nghĩa như "killer"
    - Normalize leet speak
    - Loại bỏ URLs và IP addresses
"""

import re

# ==============================================================================
# 1. OBFUSCATION PATTERNS
# ==============================================================================
# Các pattern để phát hiện và normalize các từ độc hại bị che giấu
# Ví dụ: "f u c k", "f@ck", "f*ck" -> "fuck"
PROFANITY_PATTERNS = [
    (r'f[\W_]*u[\W_]*c[\W_]*k', 'fuck'),  # f u c k, f@ck, f*ck -> fuck
    (r'sh[\W_]*i[\W_]*t', 'shit'),  # sh!t, sh*t -> shit
    (r'b[\W_]*i[\W_]*t[\W_]*c[\W_]*h', 'bitch'),  # b!tch, b*tch -> bitch
    (r'a[\W_]*s[\W_]*s[\W_]*h?[\W_]*o?[\W_]*l[\W_]*e?', 'asshole'),  # a$$hole -> asshole
    (r'd[\W_]*a[\W_]*m[\W_]*n', 'damn'),  # d@mn -> damn
    (r'idi0t', 'idiot'),  # idi0t -> idiot (leet speak)
]

# ==============================================================================
# 2. SLANG MAP
# ==============================================================================
# Dictionary để normalize các từ lóng và abbreviations phổ biến
SLANG_MAP = {
    r'\bkys\b': 'kill yourself',  # kys -> kill yourself
    r'\bstfu\b': 'shut the fuck up',  # stfu -> shut the fuck up
    r'\bgtfo\b': 'get the fuck out',  # gtfo -> get the fuck out
    r'\bu\b': 'you',  # u -> you
    r'\bur\b': 'your',  # ur -> your
    r'\br\b': 'are',  # r -> are
}

# ==============================================================================
# 3. POSITIVE CONTEXT PATTERNS
# ==============================================================================
# Danh sách các tính từ tích cực để phát hiện profanity trong ngữ cảnh khen ngợi
# Ví dụ: "fucking amazing" -> "very amazing" (không độc hại)
POSITIVE_ADJECTIVES = [
    "amazing", "awesome", "brilliant", "excellent", "fantastic", 
    "good", "great", "incredible", "love", "lovely", "magnificent", 
    "nice", "perfect", "spectacular", "superb", "wonderful", "beautiful",
    "best", "better", "genius", "talented", "smart", "funny", "cool"
]
# Tạo pattern regex từ danh sách tính từ
positive_pattern = "|".join(POSITIVE_ADJECTIVES)
# Pattern để phát hiện profanity + tính từ tích cực
CONTEXT_PATTERN = re.compile(
    rf"\b(fucking|fuckin|damn|bloody)\s+({positive_pattern})\b",
    flags=re.IGNORECASE
)

def clean_text_bert(text):
    """
    Làm sạch và normalize văn bản trước khi đưa vào model
    
    Quy trình xử lý:
        1. Chuyển về lowercase
        2. Xử lý từ "killer" trong ngữ cảnh đặc biệt
        3. Normalize obfuscated profanity
        4. Normalize slang và abbreviations
        5. Xử lý profanity trong ngữ cảnh tích cực
        6. Normalize leet speak
        7. Loại bỏ URLs, IP addresses và normalize khoảng trắng
    
    Args:
        text (str): Văn bản gốc cần xử lý
        
    Returns:
        str: Văn bản đã được làm sạch và normalize
    """
    # Kiểm tra type - nếu không phải string thì trả về chuỗi rỗng
    if not isinstance(text, str):
        return ""
        
    # Chuyển về lowercase để chuẩn hóa
    text = text.lower()
    
    # ========================================================================
    # XỬ LÝ NGỮ CẢNH ĐẶC BIỆT: "Killer"
    # ========================================================================
    # Nếu "killer" đi với "at/in/on/with" -> thường là khen (giỏi về cái gì đó)
    # Ví dụ: "killer at chess" -> "expert at chess" (an toàn)
    text = re.sub(r'\bkiller\s+(at|in|on|with)\b', r'expert \1', text)
    
    # Nếu "killer" đi với các từ tích cực -> slang tích cực (an toàn)
    # Ví dụ: "killer app", "killer feature", "killer move"
    text = re.sub(r'\bkiller\s+(app|feature|move|shot|deal)\b', r'great \1', text)

    # ========================================================================
    # 1. NORMALIZE OBFUSCATED PROFANITY
    # ========================================================================
    # Phát hiện và normalize các từ độc hại bị che giấu
    # Ví dụ: "f u c k" -> "fuck", "f@ck" -> "fuck"
    for pattern, repl in PROFANITY_PATTERNS:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    
    # ========================================================================
    # 2. NORMALIZE SLANG
    # ========================================================================
    # Chuyển các từ lóng và abbreviations về dạng đầy đủ
    # Ví dụ: "kys" -> "kill yourself", "u" -> "you"
    for pattern, repl in SLANG_MAP.items():
        text = re.sub(pattern, repl, text)

    # ========================================================================
    # 3. HANDLE POSITIVE CONTEXT
    # ========================================================================
    # Phát hiện profanity trong ngữ cảnh tích cực và thay thế
    # Ví dụ: "fucking amazing" -> "very amazing" (không độc hại)
    text = CONTEXT_PATTERN.sub(lambda m: f"very {m.group(2)}", text)

    # ========================================================================
    # 4. NORMALIZE LEET SPEAK & CHARACTERS
    # ========================================================================
    # Chuyển các ký tự leet speak về dạng bình thường
    text = re.sub(r'@', 'a', text)  # @ -> a
    text = re.sub(r'\$', 's', text)  # $ -> s
    # Giảm số lần lặp lại ký tự (coool -> cool)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # ========================================================================
    # 5. CLEANUP
    # ========================================================================
    # Loại bỏ IP addresses (ví dụ: 192.168.1.1)
    text = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' ', text)
    # Loại bỏ URLs (http://..., www....)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    # Normalize khoảng trắng (nhiều khoảng trắng -> 1 khoảng trắng)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text