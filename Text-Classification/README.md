# PyTorch Text Classification & Spam Detection Engine

<div align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch Badge"/>
  <img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Sklearn Badge"/>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Badge" />
</div>

<br>

A professional-grade, deep-learning powered text classification system designed to efficiently detect and block SMS and Email Spam. Built completely from the ground up natively leveraging **PyTorch** for Neural Network modeling and **Scikit-learn** for robust machine learning preprocessing and tokenization.

---

**Designed and Engineered by:** Soft Game Studio  
**Data Source:** [Kaggle SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

---

## 🚀 Key Features
* **TF-IDF Vectorization Algorithm**: Upgraded from standard Count Vectors to utilize term importances natively dynamically scaling vocabulary matching.
* **Deep Neural Modeling**: A multi-layered PyTorch artificial brain equipped with dimension bottlenecking structures and `.Dropout()` gates minimizing model overfitting.
* **Intelligent PreprocessingPipeline**: Integrated text cleaning standardizes casing, eradicates invalid tokens, strips rogue punctuation, handling edge cases implicitly.
* **Production Split Distribution**: Data loader ensures exactly preserved class ratios using `stratified` random splitting avoiding accuracy bias.
* **Interactive Shell Runtime**: Natively interact directly via command prompt assessing custom phrases confidently immediately.

## 📂 Project Architecture

```plaintext
Text-Classification/
│
├── data/
│   ├── spam.csv                 # Put the Kaggle SMS dataset here
│   └── topics.csv               # Secondary placeholder for multiclass updates
│
├── docs/
│   └── explanation.md           # Engineering Deep Dive & Model theory Details
│
├── src/
│   ├── data_loader.py           # PyTorch Dataset Mappings, Train/Val Splits, Pandas
│   ├── model.py                 # PyTorch nn.Module architecture defining hidden layers
│   ├── train.py                 # Neural Net Training engine, BCE loss parsing, backprop
│   ├── predict.py               # Interactive CLI Inference Engine & Model Serializer
│   └── utils.py                 # String handlers, Regex mappings, and cleanups
│
├── models/
│   ├── saved_model.pth          # Exported Best-Epoch State Dict Weights (Generated)
│   └── vectorizer.pkl           # Persisted TF-IDF Scikit-Learn Model (Generated)
│
├── requirements.txt             # Python Package definitions
└── README.md                    # You are here!
```

## 📊 Dataset Requirements
This repository was fundamentally configured to utilize the **SMS Spam Collection dataset native to Kaggle**.
To prepare it:
1. Download standard `spam.csv` from Kaggle.
2. Place it into `/data/spam.csv`.
3. The engine automatically handles traditional format layouts `label/text` or Kaggle native default `v1/v2` identifiers gracefully!

## ⚙️ Installation Instructions

1. Ensure Python 3.8+ is installed on your machine.
2. Navigate effectively to this folder in your terminal.
3. Install the dependencies using pip.
```bash
pip install -r requirements.txt
```

## 🧠 Usage & Execution

### 1. Training the Model
From inside the current workspace path, execute the python training module to vectorize texts and train PyTorch on the dataset iteratively.
```bash
python -m src.train
```

### 2. Live Interactive Inference Prediction
Once a `.pth` memory file and `.pkl` vectorizer is saved successfully during the Training Step, boot up the test engine and feed the AI random strings in real-time.
```bash
python -m src.predict
```
*Tip: Type 'exit' to terminate the interactive console.*

---
🌟 **Soft Game Studio** 
*Building advanced machine learning architectures for modern solutions.*