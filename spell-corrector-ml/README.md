# 🧠 Neural Spell Correction Model

### Developed by IST Soft Game Studio | Author: Livesh

> A Deep Learning-based **Spell & Sentence Correction System** powered by a **Seq2Seq (Encoder-Decoder) LSTM** architecture. The model operates at the **character level**, enabling it to handle rare words, names, and slang without out-of-vocabulary errors.

---

## 📌 Table of Contents

- [Features](#-features)
- [Architecture Overview](#-architecture-overview)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage & Testing](#-usage--testing)
- [How It Works](#-how-it-works)
- [Model Hyperparameters](#-model-hyperparameters)
- [Dataset Details](#-dataset-details)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🚀 Features

| Feature | Description |
|---|---|
| ✅ **Character-Level Seq2Seq** | Processes text one character at a time — no OOV (out-of-vocabulary) errors, works with names and new words. |
| ✅ **Hybrid Dataset** | Trained on 30,000+ samples from the Hugging Face `torinriley/spell-correction` dataset combined with custom synthetic noise. |
| ✅ **Smart Memory Cache** | Stores previously corrected sentences in `word_memory.json` for instant repeat lookups (zero inference cost). |
| ✅ **Safety Filters** | Protected-word system prevents important terms (e.g., names, abbreviations) from being altered. Length-ratio guard rejects low-confidence outputs. |
| ✅ **Teacher Forcing** | Uses teacher forcing during training for faster and more stable convergence. |
| ✅ **Identity Training** | Includes correct→correct pairs so the model learns to leave already-correct text unchanged. |

---

## 🏗️ Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                      TRAINING PIPELINE                       │
│                                                              │
│   HuggingFace Dataset ──┐                                    │
│                         ├──▶ Character Tokenizer ──▶ Padding │
│   Custom Dataset + Noise┘                      (Sequences)    │
│                                                    │         │
│                                                    ▼         │
│     ┌────────────┐        state_h, state_c      ┌────────────┐ │
│     │  Encoder   │ ───────────────────────────▶ │  Decoder   │ │
│     │   (LSTM)   │                              │ (LSTM +    │ │
│     └────────────┘                              │  Dense)    │ │
│           ▲                                     └────────────┘ │
│     Embedding Layer                                   │        │
│        (128-dim)                                  Softmax      │
│                                               (vocab_size)     │
└──────────────────────────────────────────────────────────────┘


┌──────────────────────────────────────────────────────────────┐
│                     INFERENCE PIPELINE                       │
│                                                              │
│   User Input ──▶ Memory Check ──▶ Protected Word Check        │
│                     │ miss               │ yes → return input │
│                     ▼                                          │
│            Encode → Decode (Autoregressive)                    │
│                     │                                          │
│                     ▼                                          │
│         Safety Filter (Length Ratio ≥ 0.7)                     │
│                     │                                          │
│                     ▼                                          │
│        Save to Memory ──▶ Return Final Output                  │
└──────────────────────────────────────────────────────────────┘
```

---

## 🔧 Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.8+ |
| Deep Learning | TensorFlow / Keras |
| Data Processing | NumPy, Pickle |
| Dataset Source | Hugging Face `datasets` library |
| Memory System | JSON file-based caching |
| Model Format | HDF5 (`.h5`) |

---

## 📁 Project Structure

```
spell-corrector-ml/
├── README.md                       # Project documentation (this file)
├── docs/
│   └── model_explanation.md        # Beginner-friendly model explanation
├── src/
│   ├── train_model.py              # Dataset preparation, model building & training
│   └── predict.py                  # Real-time inference with memory integration
├── model/
│   ├── spell_model.h5              # Trained model weights (~9.2 MB)
│   ├── tokenizer.pkl               # Character-level tokenizer
│   └── config.pkl                  # Saved config (max_len)
├── data/
│   ├── generated_dataset.txt       # Custom curated sentences (114 lines)
│   └── hf_dataset.txt              # Hugging Face dataset reference
└── memory/
    └── word_memory.json            # Cached correction results
```

---

## 📋 Prerequisites

Before running this project, make sure you have:

- **Python 3.8 or higher** installed ([download](https://www.python.org/downloads/))
- **pip** package manager (comes with Python)
- **~500 MB free disk space** (for TensorFlow and model weights)
- A working **internet connection** (required for first-time dataset download from Hugging Face)

### Recommended (for training)

- **8 GB+ RAM** — training loads 30,000+ samples into memory
- A **GPU** is optional but significantly speeds up training (CPU works fine for inference)

---

## 🛠️ Installation

1. **Clone or download** this repository:
   ```bash
   git clone https://github.com/SOFTGAMESTUDIO/Python-AI---ML-Models-For-Students.git
   cd spell-corrector-ml
   ```

2. **Install dependencies:**
   ```bash
   pip install tensorflow numpy datasets
   ```

3. **Verify installation:**
   ```bash
   python -c "import tensorflow; print(tensorflow.__version__)"
   ```

---

## 💻 Usage & Testing

### 1. Training the Model (Optional)

A pre-trained model is already included in `model/spell_model.h5`. If you want to retrain from scratch:

```bash
cd src
python train_model.py
```

**What happens during training:**
- Downloads the `torinriley/spell-correction` dataset from Hugging Face
- Loads custom sentences from `data/generated_dataset.txt`
- Applies synthetic noise (character replacement, deletion, swapping)
- Builds and trains a Seq2Seq LSTM model for **100 epochs** with batch size **64**
- Saves the model, tokenizer, and config to the `model/` directory

> ⏱️ **Training time:** ~30–60 minutes on CPU, ~5–10 minutes on GPU.

### 2. Running Predictions (Live Testing)

```bash
cd src
python predict.py
```

An interactive prompt will appear. Type any misspelled sentence and the model corrects it:

```
Enter sentence: i am goimg to schol
Output: i am going to school

Enter sentence: ths is a gret projct
Output: this is a great project

Enter sentence: plese hlp me wth my hmewrk
Output: please help me with my homework
```

Press `Ctrl + C` to exit.

---

## ⚙️ How It Works

### Training (`train_model.py`)

1. **Data Loading** — Fetches pairs from Hugging Face + reads local `generated_dataset.txt`
2. **Noise Injection** — The `add_noise()` function introduces realistic typos via three strategies:
   - **Replace** — swaps a random character with another letter
   - **Delete** — removes a random character
   - **Swap** — transposes two adjacent characters
3. **Tokenization** — Character-level tokenizer converts each character to a numeric ID
4. **Padding** — All sequences are padded to a uniform `max_len`
5. **Teacher Forcing** — Decoder receives the ground-truth output (shifted by one step) as input during training
6. **Seq2Seq Model** — Encoder LSTM compresses input → hidden state → Decoder LSTM generates output character by character
7. **Save** — Model (`.h5`), tokenizer (`.pkl`), and config (`.pkl`) are saved to disk

### Inference (`predict.py`)

1. **Load** — Loads the trained model, tokenizer, and config
2. **Protected Words** — Skips correction if input contains protected terms (`livesh`, `bca`, `ai`)
3. **Memory Lookup** — Returns cached result if the sentence was corrected before
4. **Autoregressive Decoding** — Feeds input to encoder, then feeds `START_TOKEN` to decoder and generates characters one at a time until `END_TOKEN` is predicted
5. **Safety Filter** — If the output is less than 70% the length of the input, the original sentence is returned (prevents garbage output)
6. **Cache & Return** — Saves the result to `word_memory.json` for future lookups

---

## 📊 Model Hyperparameters

| Parameter | Value |
|---|---|
| Embedding Dimension | 128 |
| LSTM Hidden Units | 256 |
| Max Training Samples (HF) | 30,000 |
| Epochs | 100 |
| Batch Size | 64 |
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |
| Tokenization | Character-level |
| Padding | Post-padding |

---

## 📚 Dataset Details

### Hugging Face Dataset
- **Source:** [`torinriley/spell-correction`](https://huggingface.co/datasets/torinriley/spell-correction)
- **Format:** Pairs of `(misspelled, correct)` text
- **Samples used:** Up to 30,000

### Custom Dataset (`data/generated_dataset.txt`)
- **114 curated sentences** across multiple domains:
  - 🎓 Academic & student life
  - 💼 Business & finance
  - 🏥 Healthcare & medical
  - 💻 Technology & software
  - 📞 Customer service
  - 📝 General English grammar
- Each sentence is used to generate **3 noisy variants** + **1 identity pair** during training

---

## 🔍 Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'tensorflow'` | Run `pip install tensorflow` |
| `ModuleNotFoundError: No module named 'datasets'` | Run `pip install datasets` |
| `FileNotFoundError: spell_model.h5` | Train the model first: `cd src && python train_model.py` |
| `Memory file corrupted` | Delete `memory/word_memory.json` — it will be recreated automatically |
| Model returns the input unchanged | Input may contain a protected word, or the output failed the safety filter |
| Training is very slow | Consider using a GPU, or reduce `MAX_SAMPLES` and `epochs` in `train_model.py` |
| `config.pkl` not found | Retrain the model — older versions may not have saved this file |

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork this repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

Please follow existing code style and include comments for any new functionality.

---

## 📄 License

This project is developed as an **academic project** by **Livesh** at **IST Soft Game Studio**.  
Free to use for **educational and research purposes**.

---

<p align="center">
  Made with ❤️ by <strong>IST Soft Game Studio</strong>
</p>