"""
PyTorch Dataset cho Toxic Content Detection

Module này định nghĩa custom Dataset class để xử lý dữ liệu văn bản
cho training và inference. Dataset này:
    - Tokenize văn bản sử dụng tokenizer
    - Padding và truncation để đảm bảo độ dài đồng nhất
    - Trả về input_ids, attention_mask và labels (nếu có)
"""

import torch
from torch.utils.data import Dataset

class ToxicDataset(Dataset):
    """
    Custom PyTorch Dataset cho toxic content classification
    
    Dataset này xử lý văn bản và labels, tokenize chúng và trả về
    dạng tensor phù hợp cho model RoBERTa.
    """
    
    def __init__(self, texts, labels, tokenizer, max_len):
        """
        Khởi tạo dataset
        
        Args:
            texts (list/array): Danh sách các văn bản cần xử lý
            labels (list/array or None): Danh sách labels tương ứng (None nếu inference)
            tokenizer: Tokenizer từ transformers (RoBERTaTokenizerFast)
            max_len (int): Độ dài tối đa của sequence sau khi tokenize
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        """
        Trả về số lượng samples trong dataset
        
        Returns:
            int: Số lượng samples
        """
        return len(self.texts)

    def __getitem__(self, item):
        """
        Lấy một sample từ dataset tại index `item`
        
        Args:
            item (int): Index của sample cần lấy
            
        Returns:
            dict: Dictionary chứa:
                - input_ids: Token IDs của văn bản (tensor)
                - attention_mask: Attention mask (tensor)
                - labels: Labels của sample (tensor, chỉ có khi training)
        """
        # Chuyển văn bản sang string để đảm bảo type consistency
        text = str(self.texts[item])
        
        # Nếu đang train/val thì có labels, nếu infer thì label có thể None
        labels = self.labels[item] if self.labels is not None else None

        # Tokenize văn bản
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # Thêm [CLS] và [SEP] tokens
            max_length=self.max_len,  # Giới hạn độ dài
            return_token_type_ids=False,  # RoBERTa không cần token_type_ids
            padding='max_length',  # Padding đến max_length
            truncation=True,  # Cắt bớt nếu quá dài
            return_attention_mask=True,  # Trả về attention mask
            return_tensors='pt',  # Trả về PyTorch tensors
        )

        # Tạo dictionary chứa input_ids và attention_mask
        item_dict = {
            'input_ids': encoding['input_ids'].flatten(),  # Flatten để loại bỏ dimension thừa
            'attention_mask': encoding['attention_mask'].flatten()
        }
        
        # Thêm labels nếu có (cho training/validation)
        if labels is not None:
            item_dict['labels'] = torch.tensor(labels, dtype=torch.float)
            
        return item_dict