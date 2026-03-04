"""
Custom Trainer Module cho Multi-Label Classification với Weighted Loss

Module này định nghĩa custom Trainer class để xử lý class imbalance
trong multi-label classification bằng cách sử dụng weighted loss.
"""

import torch
import torch.nn as nn
from transformers import Trainer
from src.config import DEVICE

class MultiLabelTrainer(Trainer):
    """
    Custom Trainer để xử lý Class Imbalance sử dụng Weighted Loss
    
    Trainer này kế thừa từ HuggingFace Trainer và override phương thức
    compute_loss để sử dụng BCEWithLogitsLoss với pos_weight.
    Điều này giúp model học tốt hơn trên các class hiếm (như severe_toxic).
    """
    
    def __init__(self, class_weights, *args, **kwargs):
        """
        Khởi tạo custom trainer với class weights
        
        Args:
            class_weights (array-like): Weights cho từng class để xử lý imbalance
            *args, **kwargs: Các tham số khác cho Trainer (model, args, datasets, etc.)
        """
        super().__init__(*args, **kwargs)
        # Chuyển weights thành tensor và di chuyển lên device (GPU/CPU)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Tính toán loss với weighted BCEWithLogitsLoss
        
        Args:
            model: Model để forward pass
            inputs (dict): Dictionary chứa input_ids, attention_mask, labels
            return_outputs (bool): Có trả về outputs hay không
            **kwargs: Các tham số khác
            
        Returns:
            loss (tensor): Weighted loss value
            (loss, outputs) (tuple): Nếu return_outputs=True
        """
        # Lấy labels từ inputs
        labels = inputs.get("labels")
        
        # Forward pass qua model
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Sử dụng Weighted BCEWithLogitsLoss
        # pos_weight: weight cho positive class (class hiếm sẽ có weight cao hơn)
        # Công thức: loss = -[y*log(sigmoid(x)) * pos_weight + (1-y)*log(1-sigmoid(x))]
        loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
        loss = loss_fct(logits, labels)
        
        # Trả về loss (và outputs nếu cần)
        return (loss, outputs) if return_outputs else loss