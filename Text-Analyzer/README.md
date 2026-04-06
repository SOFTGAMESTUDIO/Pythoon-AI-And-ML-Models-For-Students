# Text Analyzer

A powerful and straightforward text analysis application built dynamically with Python and Streamlit. This tool utilizes industry-standard Natural Language Processing frameworks like SpaCy and NLTK to automatically clean input texts, extract semantic tokens, identify Parts-of-Speech (POS), and recognize Named Entities.

## ✨ Features
- **Data Cleansing**: Strips punctuation and normalizes string inputs into comprehensive lowercase characters strings.
- **Tokenize & POS Tagging**: Isolates significant text tokens, generating accurate linguistic lemmas and Parts of Speech parameters.
- **Named Entity Recognition (NER)**: Automatically scans input data to recognize and categorize meaningful text entities.
- **Dynamic Formatting**: Securely saves and architectures analytical results seamlessly into `.csv` datasets on the fly.

## ⚙️ Setup and Installation

1. **Clone or Navigate to the Repository**
   Access the appropriate local machine directory:
   ```bash
   cd Text-Analyzer
   ```

2. **Install Package Requirements**
   Use standard pip implementations to load dependencies securely via `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

3. **Validate & Download Language Modules**
   Initialize script mechanisms to load required NLP tokenizers and pipelines:
   ```bash
   # Download standard NLTK dependencies (stopwords & punkt)
   python SetupNLTK.py
   
   # Download the native English pipeline for SpaCy analyses
   python -m spacy download en_core_web_sm
   ```

## 🚀 How to Test & Use

You can seamlessly interact with the Text Analyzer in two specific ways: via visual stream logic (Streamlit) or programmatically via code execution.

### Method 1: Interactive Streamlit App (Recommended)
This method spins up a robust web configuration UI. 
```bash
streamlit run app.py
```
**Steps to Follow:**
1. Execute the stream command. 
2. A tab should seamlessly open in your browser interface (typically configured on `localhost`).
3. Deposit your text string into the visual designated field and hit **Analyze**.
4. The output entities, dataset tokens, and confirmations of CSV exports will all be represented visually.

### Method 2: System Console Interface
If looking for automated validation bypassing browser rendering, rely on the main executable loop.
```bash
python main.py
```
**Steps to Follow:**
1. Populate your target raw text explicitly inside a text file named exactly `sample.txt`, located neatly within the `data/` subdirectory.
2. Execute the python terminal process. 
3. Outputs are delivered natively onto the console, and absolute parsed fields will correctly generate internally to `output/results.csv`.

## 📖 Further Documentation
Need an under-the-hood breakdown defining the core modular scripts (`cleaner.py`, `processor.py`, and `exporter.py`)? Be sure to read the developer mapping guide inside [docs/Text_Analyzer.md](docs/Text_Analyzer.md).
