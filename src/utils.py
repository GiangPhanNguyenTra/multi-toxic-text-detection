"""
Utility Functions cho Training và Evaluation

Module này chứa các hàm tiện ích để tính toán metrics trong quá trình
training và evaluation của model.
"""

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

def compute_metrics(eval_pred):
    """
    Tính toán các metrics cho multi-label classification
    
    Metrics được tính:
        - F1 Score (micro và macro average)
        - Accuracy
        - ROC AUC Score (macro average)
    
    Args:
        eval_pred (tuple): Tuple chứa (logits, labels)
            - logits: Raw logits từ model (numpy array, shape: [batch_size, num_labels])
            - labels: Ground truth labels (numpy array, shape: [batch_size, num_labels])
    
    Returns:
        dict: Dictionary chứa các metrics:
            - f1_micro: F1 score với micro averaging (tính trên tất cả samples và labels)
            - f1_macro: F1 score với macro averaging (tính trung bình F1 của từng label)
            - roc_auc: ROC AUC score với macro averaging
            - accuracy: Accuracy score
    """
    logits, labels = eval_pred
    
    # Chuyển logits thành probabilities bằng sigmoid
    # Công thức: sigmoid(x) = 1 / (1 + exp(-x))
    probs = 1 / (1 + np.exp(-logits))
    
    # Áp dụng threshold 0.5 để chuyển probabilities thành binary predictions
    # Mỗi label được phân loại độc lập (multi-label classification)
    y_pred = (probs > 0.5).astype(int)
    
    # Tính F1 Score
    # Micro: Tính trên tất cả samples và labels (overall F1)
    f1_micro = f1_score(labels, y_pred, average='micro')
    # Macro: Tính trung bình F1 của từng label (F1 cho mỗi label rồi average)
    f1_macro = f1_score(labels, y_pred, average='macro')
    
    # Tính Accuracy (tỷ lệ predictions đúng)
    accuracy = accuracy_score(labels, y_pred)
    
    # Tính ROC AUC Score
    # Macro: Tính ROC AUC cho từng label rồi average
    # Xử lý exception nếu chỉ có một class trong batch (không thể tính ROC AUC)
    try:
        roc_auc = roc_auc_score(labels, probs, average='macro')
    except ValueError:
        # Nếu có lỗi (ví dụ: chỉ có một class), đặt ROC AUC = 0.0
        roc_auc = 0.0
    
    return {
        'f1_micro': f1_micro,  
        'f1_macro': f1_macro, 
        'roc_auc': roc_auc,   
        'accuracy': accuracy 
    }