"""
Configuration file cho Toxic Content Detection Model

Module này chứa tất cả các cấu hình cần thiết cho training và inference:
    - Đường dẫn đến dữ liệu và model
    - Cấu hình model (tên model, max length)
    - Hyperparameters cho training (batch size, learning rate, epochs)
    - Danh sách các labels cần dự đoán
"""

import torch
import os

# ==============================================================================
# DEVICE CONFIGURATION
# ==============================================================================
# Tự động chọn GPU nếu có, nếu không thì dùng CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================
# Lấy đường dẫn thư mục gốc của project (Module_Tranformer)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Đường dẫn đến file dữ liệu training
DATA_PATH = os.path.join(BASE_DIR, 'Data', 'train.csv')

# Đường dẫn để lưu model sau khi training
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'roberta_toxic') 

# ==============================================================================
# MODEL CONFIGURATION
# ==============================================================================
# Tên pre-trained model từ HuggingFace (RoBERTa base)
MODEL_NAME = 'roberta-base' 

# Độ dài tối đa của sequence (số tokens)
# RoBERTa có giới hạn 512, nhưng dùng 128 để tiết kiệm bộ nhớ và tăng tốc
MAX_LEN = 128

# ==============================================================================
# TRAINING HYPERPARAMETERS
# ==============================================================================
# Config tối ưu cho RTX 3060 (6GB VRAM)
BATCH_SIZE = 8  # Batch size cho mỗi device (giảm nếu hết VRAM)
GRADIENT_ACCUMULATION_STEPS = 2  # Tích lũy gradient để mô phỏng batch size = 16
EPOCHS = 4  # Số epoch training
LEARNING_RATE = 1e-5  # Learning rate cho fine-tuning (thấp hơn pre-training) 
FP16 = True  # Bật mixed precision training để tiết kiệm VRAM và tăng tốc

# ==============================================================================
# LABEL CONFIGURATION
# ==============================================================================
# Danh sách 6 labels cho multi-label classification
# Mỗi văn bản có thể có nhiều labels cùng lúc (ví dụ: vừa toxic vừa insult)
LABEL_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']