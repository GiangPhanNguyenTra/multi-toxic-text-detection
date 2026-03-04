"""
Script test inference cho Toxic Content Detection Model

Module này dùng để kiểm tra model trên các test cases cụ thể,
đặc biệt là các edge cases như:
    - "killer" trong ngữ cảnh khen ngợi (an toàn) vs đe dọa (độc hại)
    - Profanity trong ngữ cảnh tích cực (an toàn)
    - Các trường hợp đặc biệt khác
"""

from src.predict import ToxicPredictor

# Khởi tạo predictor (load model và tokenizer)
predictor = ToxicPredictor()

# Danh sách các test cases để kiểm tra model
test_cases = [
    "You're a killer at playing chess!",       # Case: "killer" trong ngữ cảnh khen ngợi (an toàn)
    "This is a killer app.",                   # Case: "killer" như slang tích cực (an toàn)
    "He is a serial killer.",                  # Case: "killer" thực sự độc hại (không có 'at')
    "I will kill you.",                        # Case: Threat rõ ràng (độc hại)
    "This is fucking amazing"                  # Case: Profanity trong ngữ cảnh tích cực (an toàn)
]

# Chạy test trên từng test case
print("-" * 50)
for text in test_cases:
    # Dự đoán độc tính của văn bản
    result = predictor.predict(text)
    
    # In kết quả
    print(f"Input:    {text}")
    print(f"Cleaned:  {result['cleaned_text']}")  # Văn bản sau preprocessing
    print(f"Is Toxic: {result['is_toxic']}")      # Có độc hại hay không
    
    # Nếu độc hại, in ra các labels cụ thể có điểm số > 0.5
    if result['is_toxic']:
        print("Labels:", {k:v for k,v in result['predictions'].items() if v > 0.5})
    print("-" * 50)