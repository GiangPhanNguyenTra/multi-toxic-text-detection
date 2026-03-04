"""
Flask Web Application API

Module này cung cấp REST API để phát hiện nội dung độc hại trong văn bản.
Sử dụng Flask framework để tạo web server với 2 endpoints:
    - GET / : Trang web giao diện người dùng
    - POST /predict : API endpoint để dự đoán độc tính của văn bản
"""

from flask import Flask, request, jsonify, render_template
from src.predict import ToxicPredictor
import os

# Khởi tạo Flask application
app = Flask(__name__)

# Load model ONCE when app starts
# Model được load một lần khi server khởi động để tránh phải load lại nhiều lần
# (tiết kiệm thời gian và bộ nhớ)
print("Initializing Model for API...")
predictor = ToxicPredictor()

@app.route('/')
def home():
    """
    Endpoint chính - Trả về trang web giao diện người dùng
    
    Returns:
        HTML template: Trang index.html để người dùng nhập văn bản và xem kết quả
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint để dự đoán độc tính của văn bản
    
    Request Body (JSON):
        {
            "text": "Văn bản cần kiểm tra"
        }
    
    Returns:
        JSON response chứa:
        - text: Văn bản gốc
        - cleaned_text: Văn bản sau khi preprocessing
        - predictions: Dictionary chứa điểm số cho từng label (toxic, severe_toxic, obscene, threat, insult, identity_hate)
        - is_toxic: Boolean cho biết văn bản có độc hại hay không
        
    Status Codes:
        - 200: Thành công
        - 400: Lỗi - không có văn bản được cung cấp
        - 500: Lỗi server - có exception xảy ra
    """
    try:
        # Lấy dữ liệu JSON từ request
        data = request.json
        text = data.get('text', '')
        
        # Kiểm tra xem có văn bản được cung cấp hay không
        if not text:
            return jsonify({'error': 'No text provided'}), 400
            
        # Chạy prediction sử dụng model đã load
        result = predictor.predict(text)
        return jsonify(result)
        
    except Exception as e:
        # Xử lý lỗi và trả về thông báo lỗi
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Chạy Flask development server
    # debug=True: Bật chế độ debug để hiển thị lỗi chi tiết
    # use_reloader=False: Tắt auto-reload để tránh load model 2 lần (tiết kiệm bộ nhớ)
    print("Server running at http://127.0.0.1:5000")
    app.run(debug=True, use_reloader=False)