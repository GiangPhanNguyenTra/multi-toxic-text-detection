# Toxic Text Detection Model

A machine learning project for detecting toxic content in text using a RoBERTa-based transformer model.

## Project Overview

This project implements a text classification system to detect toxic content using fine-tuned RoBERTa (Robustly Optimized BERT). It includes:

- Data preprocessing and augmentation
- Model training pipeline
- REST API for inference
- Web interface for testing

## Project Structure

```
.
├── app.py                      # Flask web application
├── train_pipeline.py           # Main training pipeline
├── test_inference.py           # Inference testing script
├── create_aug_data.py          # Data augmentation script
├── requirements.txt            # Python dependencies
│
├── src/
│   ├── config.py              # Configuration settings
│   ├── dataset.py             # Custom dataset classes
│   ├── train.py               # Training functions
│   ├── predict.py             # Prediction/inference functions
│   ├── preprocessing.py        # Text preprocessing utilities
│   └── utils.py               # Helper utilities
│
├── Data/
│   ├── train.csv              # Training dataset
│   ├── augmented_data.csv     # Augmented training data
│   └── file train.txt         # Additional training data
│
├── models/
│   └── roberta_toxic/         # Fine-tuned RoBERTa model
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       └── vocab.json
│
├── templates/
│   └── index.html             # Web interface
│
├── reports/
│   └── MODULE_REPORT.md       # Project report
│
└── results_roberta/           # Training results and logs
```

## Setup

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

1. Clone the repository

```bash
git clone <repository-url>
cd Module_Transformer
```

2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Training

Run the training pipeline:

```bash
python train_pipeline.py
```

This will:

- Load and preprocess the training data
- Fine-tune the RoBERTa model
- Evaluate the model performance
- Save the trained model

### Data Augmentation

Generate augmented training data:

```bash
python create_aug_data.py
```

### Inference

Test the model with your own text:

```bash
python test_inference.py
```

### Web Application

Launch the Flask web application:

```bash
python app.py
```

Then visit `http://localhost:5000` in your browser to use the web interface.

## Model Details

- **Model**: RoBERTa (Robustly Optimized BERT)
- **Task**: Binary text classification (toxic/non-toxic)
- **Fine-tuning**: Custom dataset specific training
- **Tokenizer**: RoBERTa tokenizer with custom vocabulary

## Configuration

Edit `src/config.py` to customize:

- Model hyperparameters
- Training settings
- Data paths
- Inference parameters

## Results

Training results and metrics are saved in the `results_roberta/` directory.

## Dependencies

Key packages:

- transformers: Hugging Face transformer models
- torch: PyTorch framework
- pandas: Data manipulation
- Flask: Web framework
- scikit-learn: Machine learning utilities

See `requirements.txt` for a complete list.

## License

[Add your license here]

## Authors

Phan Nguyen Tra Giang
