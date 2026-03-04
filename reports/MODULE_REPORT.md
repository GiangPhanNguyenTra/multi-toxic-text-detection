# 🛡️ Toxic Comment Classification - Deep Learning Module

> **Model Architecture:** Fine-tuned RoBERTa-base  
> **Approach:** Hybrid (Deep Learning + Rule-based Preprocessing)  
> **Hardware:** NVIDIA RTX 3060 (6GB VRAM) w/ Mixed Precision (FP16)

---

## 1. Tổng Quan (Overview)

Module này giải quyết bài toán **Multi-label Text Classification** trên bộ dữ liệu Jigsaw Toxic Comment. Hệ thống phân loại bình luận vào 6 nhãn:
`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`.

Khác với các phương pháp cổ điển, module này sử dụng **RoBERTa** kết hợp với **Hybrid Preprocessing** để xử lý ngữ cảnh, tiếng lóng và các từ ngữ bị che giấu (obfuscation).

---

## 2. Kiến Trúc Hệ Thống (System Architecture)

```text
Module_Tranformer/
├── src/                        # Source code lõi
│   ├── config.py               # Cấu hình Hyperparameters & Paths
│   ├── preprocessing.py        # Logic Hybrid Cleaning (Regex + Context Rules)
│   ├── dataset.py              # PyTorch Dataset Class
│   ├── train.py                # Custom Trainer (xử lý Imbalance)
│   └── predict.py              # Inference Class cho API
├── models/                     # Lưu trữ Model Checkpoints
│   └── roberta_toxic/          # Model RoBERTa đã train
│   └── distilbert_toxic/       # Model DistilBERT đã train
├── reports/
│   ├── MODULE_REPORT.md        # Reports
├── templates/
│   ├── index.html              # UI for test
├── app.py                      # Flask API Server
└── train_pipeline.py           # Script quản lý luồng huấn luyện
```

---

## 3. Các Kỹ Thuật Áp Dụng (Key Techniques)

### 3.1. Mô hình: RoBERTa-base

Chuyển từ `DistilBERT` sang `RoBERTa-base` vì:

- **Dữ liệu lớn:** RoBERTa được train trên 160GB văn bản.
- **Ngữ cảnh tốt hơn:** Hiểu tốt hơn các sắc thái ngôn ngữ (sarcasm, slang).
- **Dynamic Masking:** Học biểu diễn từ vựng phong phú hơn.

### 3.2. Hybrid Preprocessing (Kỹ thuật Lai) 🌟

Đây là điểm cải tiến quan trọng nhất giúp giảm False Positives. Quy trình gồm:

1.  **Obfuscation Handling:** Chuẩn hóa từ bị viết sai.
    - Ví dụ: `f*ck` -> `fuck`, `@ss` -> `ass`.
2.  **Slang Translation:** Dịch từ lóng.
    - Ví dụ: `kys` -> `kill yourself`, `stfu` -> `shut the fuck up`.
3.  **Context-Aware Masking:** Regex phát hiện ngữ cảnh tích cực.
    - Logic: `[Profanity] + [Positive Adjective]` -> `Very + [Positive Adjective]`
    - Ví dụ: _"This is **fucking amazing**"_ -> _"This is **very amazing**"_ (Clean).
4.  **Edge-Case Handling:** Xử lý cụm từ đặc biệt.
    - Logic: _"Killer at/in/on"_ -> _"Expert at/in/on"_
    - Ví dụ: _"You're a **killer at** chess"_ -> _"You're a **expert at** chess"_ (Clean).

### 3.3. Xử lý Mất cân bằng dữ liệu (Imbalance Handling)

- Sử dụng **Weighted BCEWithLogitsLoss**.
- Tính trọng số `pos_weight` cho từng nhãn.
- `Clip` trọng số (range 1.0 - 8.0) để tránh phạt quá nặng.

### 3.4. Tối ưu phần cứng (Hardware Optimization)

Cho GPU 6GB VRAM:

- **Mixed Precision (FP16):** Giảm 50% VRAM, tăng tốc train.
- **Gradient Accumulation:** Tích lũy 2 bước (Batch size 8 -> Effective 16).

---

## 4. Cấu Hình Huấn Luyện (Training Configuration)

| Tham số           | Giá trị        | Giải thích                  |
| :---------------- | :------------- | :-------------------------- |
| **Model**         | `roberta-base` | Transformer Backbone        |
| **Epochs**        | 4              | Đủ để hội tụ, tránh Overfit |
| **Batch Size**    | 8              | Tối ưu cho 6GB VRAM         |
| **Accumulation**  | 2              | Effective Batch Size = 16   |
| **Learning Rate** | 2e-5           | Chuẩn cho Fine-tuning       |
| **Max Length**    | 128            | Bao phủ hầu hết comment     |

---

## 5. Kết Quả Thực Nghiệm (Results)

### Metrics (Sau 4 Epochs):

- **Training Runtime:** ~1 giờ 43 phút
- **Validation Loss:** `0.1747`
- **ROC-AUC Score:** `0.9897` (Rất tốt)
- **Accuracy:** `91.86%`
- **F1-Micro:** `0.7823`

### Môt số Test Case Thực Tế:

| Input Text                        | Cổ điển / DistilBERT        | **RoBERTa + Hybrid**                         | Kết quả |
| :-------------------------------- | :-------------------------- | :------------------------------------------- | :------ |
| _"This is fucking amazing!"_      | ❌ Toxic (Sai ngữ cảnh)     | ✅ **Clean** (Hiểu là "Very amazing")        | Pass    |
| _"Damn, that's a brilliant idea"_ | ❌ Toxic (Keyword bias)     | ✅ **Clean**                                 | Pass    |
| _"Kys you loser"_                 | ❌ Clean (Không hiểu slang) | ✅ **Toxic (Threat)** (Hiểu "kill yourself") | Pass    |
| _"You're a killer at chess"_      | ❌ Toxic (Sai nghĩa từ)     | ✅ **Clean** (Hiểu là "Expert")              | Pass    |
| _"f u c k this sh1t"_             | ❌ Clean (Bị che dấu)       | ✅ **Toxic** (Obfuscation handling)          | Pass    |

---

## 6. Kết Luận

Module Fine-tuned RoBERTa kết hợp Hybrid Preprocessing đã giải quyết triệt để các hạn chế của mô hình cổ điển:

1.  Hiểu ngữ cảnh tích cực chứa từ nhạy cảm.
2.  Phát hiện tốt các từ viết tắt/viết méo.
3.  Độ chính xác cao trên các nhãn hiếm (Threat, Identity Hate).
