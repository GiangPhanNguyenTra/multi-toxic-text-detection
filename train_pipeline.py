"""
Training Pipeline cho Toxic Content Detection Model

Module này thực hiện toàn bộ quy trình training mô hình RoBERTa để phát hiện nội dung độc hại.
Bao gồm:
    - Load và merge dữ liệu gốc với dữ liệu augmented
    - Preprocessing văn bản
    - Chia train/validation split
    - Tính toán class weights để xử lý class imbalance
    - Khởi tạo model, tokenizer, và trainer
    - Training và evaluation
    - Lưu model đã train
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
# from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, TrainingArguments
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, TrainingArguments

# Imports from our src package
from src import config
from src.preprocessing import clean_text_bert
from src.dataset import ToxicDataset
from src.train import MultiLabelTrainer
from src.utils import compute_metrics

def main():
    """
    Hàm chính thực hiện toàn bộ pipeline training
    
    Quy trình:
        1. Load dữ liệu (gốc + augmented nếu có)
        2. Preprocessing văn bản
        3. Chia train/validation
        4. Tính class weights
        5. Khởi tạo tokenizer, dataset, model
        6. Setup trainer với custom loss
        7. Training
        8. Evaluation và lưu model
    """
    print(f"Starting Training Pipeline on {config.DEVICE}")
    
    # 1. Load Data
    print("Loading data...")
    if not os.path.exists(config.DATA_PATH):
        raise FileNotFoundError(f"Data not found at {config.DATA_PATH}")
        
    df_original = pd.read_csv(config.DATA_PATH)
    
    # Load thêm Augmented Data nếu có
    # Dữ liệu augmented giúp cải thiện hiệu suất model trên các edge cases
    aug_path = os.path.join(os.path.dirname(config.DATA_PATH), 'augmented_data.csv')
    if os.path.exists(aug_path):
        print("Found augmented data! Merging...")
        df_aug = pd.read_csv(aug_path)
        # Gộp 2 dataframe và xáo trộn
        df = pd.concat([df_original, df_aug], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"New dataset size: {len(df)} (Original: {len(df_original)} + Aug: {len(df_aug)})")
    else:
        df = df_original
    
    # 2. Preprocess
    # Làm sạch văn bản: normalize leet speak, xử lý obfuscation, chuẩn hóa khoảng trắng
    print("Preprocessing texts...")
    df['cleaned_text'] = df['comment_text'].apply(clean_text_bert)
    
    # 3. Split
    # Chia dữ liệu thành tập train (85%) và validation (15%)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['cleaned_text'].values, 
        df[config.LABEL_COLS].values, 
        test_size=0.15, 
        random_state=42  # Fixed seed để đảm bảo reproducibility
    )
    
    # 4. Compute Weights for Imbalance
    # Tính toán class weights để xử lý class imbalance (số lượng mẫu toxic ít hơn nhiều so với clean)
    # Công thức: weight = (total_samples - positive_samples) / positive_samples
    print("Computing class weights...")
    num_positives = np.sum(train_labels, axis=0)  # Số lượng mẫu positive cho mỗi class
    # Avoid division by zero - đảm bảo không có class nào có 0 positive samples
    num_positives = np.clip(num_positives, 1, None) 
    class_weights = (len(train_labels) - num_positives) / num_positives
    # Giới hạn weight tối đa là 3.0 để tránh over-weighting các class hiếm
    class_weights = np.clip(class_weights, 1.0, 3.0)
    print(f"Class Weights: {class_weights}")

    # 5. Tokenizer & Datasets
    # Khởi tạo tokenizer từ pre-trained RoBERTa
    tokenizer = RobertaTokenizerFast.from_pretrained(config.MODEL_NAME)
    # Tạo PyTorch Dataset cho train và validation
    train_dataset = ToxicDataset(train_texts, train_labels, tokenizer, config.MAX_LEN)
    val_dataset = ToxicDataset(val_texts, val_labels, tokenizer, config.MAX_LEN)

    # 6. Model
    # Load pre-trained RoBERTa và thêm classification head cho multi-label classification
    model = RobertaForSequenceClassification.from_pretrained(
        config.MODEL_NAME, 
        num_labels=len(config.LABEL_COLS),  # 6 labels: toxic, severe_toxic, obscene, threat, insult, identity_hate
        problem_type="multi_label_classification"  # Mỗi sample có thể có nhiều labels
    )
    # Di chuyển model lên GPU nếu có
    model.to(config.DEVICE)

    # 7. Trainer Setup
    # Cấu hình các tham số training
    training_args = TrainingArguments(
        output_dir='./results_roberta',  # Thư mục lưu checkpoints
        num_train_epochs=config.EPOCHS,  # Số epoch training
        
        # Batch size và gradient accumulation để tối ưu cho GPU có VRAM hạn chế
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,  # Tích lũy gradient để mô phỏng batch size lớn hơn
        
        fp16=config.FP16,  # Mixed precision training để tiết kiệm VRAM và tăng tốc
        warmup_steps=500,  # Số bước warmup learning rate
        weight_decay=0.01,  # L2 regularization
        logging_dir='./logs',  # Thư mục lưu logs
        logging_steps=50,  # Log mỗi 50 steps
        eval_strategy="epoch",  # Evaluate sau mỗi epoch
        save_strategy="epoch",  # Lưu checkpoint sau mỗi epoch
        save_total_limit=1,  # Chỉ giữ 1 checkpoint mới nhất để tiết kiệm dung lượng
        learning_rate=config.LEARNING_RATE,  # Learning rate
        
        dataloader_num_workers=0,  # Số worker cho data loading (0 = main process)
        report_to="none"  # Không gửi metrics lên external services
    )

    # Khởi tạo custom trainer với weighted loss để xử lý class imbalance
    trainer = MultiLabelTrainer(
        class_weights=class_weights,  # Weights cho từng class
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics  # Hàm tính toán metrics (F1, ROC-AUC, accuracy)
    )

    # 8. Train
    # Bắt đầu quá trình training
    print("Training started...")
    trainer.train()

    # 9. Evaluate & Save
    # Đánh giá model trên validation set và in kết quả
    print("Evaluating...")
    results = trainer.evaluate()
    print(f"Results: {results}")

    # Lưu model và tokenizer đã được fine-tune
    print(f"Saving model to {config.MODEL_SAVE_PATH}")
    model.save_pretrained(config.MODEL_SAVE_PATH)
    tokenizer.save_pretrained(config.MODEL_SAVE_PATH)
    print("Pipeline Complete!")

if __name__ == "__main__":
    main()