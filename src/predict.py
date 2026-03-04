"""
Prediction Module cho Toxic Content Detection

Module này cung cấp class ToxicPredictor để load model và dự đoán
độc tính của văn bản. Sử dụng RoBERTa model đã được fine-tune.
"""

import torch
# from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from src.config import MODEL_SAVE_PATH, MODEL_NAME, MAX_LEN, DEVICE, LABEL_COLS
from src.preprocessing import clean_text_bert

class ToxicPredictor:
    """
    Class để load model và thực hiện prediction trên văn bản
    
    Class này quản lý việc load model, tokenizer và thực hiện inference.
    Model được load một lần khi khởi tạo để tối ưu hiệu suất.
    """
    
    def __init__(self, model_path=None):
        """
        Khởi tạo predictor và load model
        
        Args:
            model_path (str, optional): Đường dẫn đến model đã train.
                                      Nếu None, dùng MODEL_SAVE_PATH từ config.
        """
        self.device = DEVICE
        path = model_path if model_path else MODEL_SAVE_PATH
        print(f"Loading model from {path}...")
        
        try:
            # Load tokenizer và model từ đường dẫn đã chỉ định
            self.tokenizer = RobertaTokenizerFast.from_pretrained(path)
            self.model = RobertaForSequenceClassification.from_pretrained(path).to(self.device)
            # Đặt model ở chế độ evaluation (tắt dropout, batch norm updates)
            self.model.eval()
            print("Model loaded successfully!")
        except Exception as e:
            # Nếu không load được model (ví dụ: chưa train), chỉ load tokenizer để test
            print(f"Error loading model: {e}")
            print("Using default tokenizer for fallback/testing setup...")
            self.tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)
            self.model = None

    def predict(self, text):
        """
        Dự đoán độc tính của văn bản
        
        Args:
            text (str): Văn bản cần kiểm tra
            
        Returns:
            dict: Dictionary chứa:
                - text: Văn bản gốc
                - cleaned_text: Văn bản sau preprocessing
                - predictions: Dictionary với điểm số cho từng label
                - is_toxic: Boolean cho biết có độc hại hay không (bất kỳ label nào > 0.5)
                - error: Thông báo lỗi nếu model chưa được load
        """
        # Kiểm tra xem model đã được load chưa
        if self.model is None:
            return {"error": "Model not loaded"}

        # Bước 1: Preprocessing - làm sạch và normalize văn bản
        cleaned_text = clean_text_bert(text)
        
        # Bước 2: Tokenize - chuyển văn bản thành token IDs
        encoding = self.tokenizer.encode_plus(
            cleaned_text,
            add_special_tokens=True,  # Thêm [CLS] và [SEP]
            max_length=MAX_LEN,  # Giới hạn độ dài
            return_token_type_ids=False,  # RoBERTa không cần
            padding='max_length',  # Padding đến max_length
            truncation=True,  # Cắt bớt nếu quá dài
            return_attention_mask=True,  # Cần attention mask
            return_tensors='pt',  # PyTorch tensors
        )
        
        # Di chuyển tensors lên device (GPU hoặc CPU)
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Bước 3: Inference - chạy model để lấy predictions
        with torch.no_grad():  # Tắt gradient computation để tiết kiệm bộ nhớ
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # Raw logits từ model
            # Áp dụng sigmoid để chuyển logits thành probabilities (0-1)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        # Bước 4: Format kết quả
        result = {
            "text": text,  # Văn bản gốc
            "cleaned_text": cleaned_text,  # Văn bản sau preprocessing
            "predictions": {},  # Dictionary chứa điểm số cho từng label
            "is_toxic": False  # Mặc định là không độc hại
        }
        
        # Lặp qua từng label và lưu điểm số
        for idx, label in enumerate(LABEL_COLS):
            score = float(probs[idx])
            result["predictions"][label] = score
            # Nếu bất kỳ label nào có điểm > 0.5, đánh dấu là độc hại
            if score > 0.5:
                result["is_toxic"] = True
                
        return result