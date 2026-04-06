# 🧠 Model Explanation — How the Spell Corrector Works

> This document explains the complete inner workings of the Neural Spell Correction model in a **beginner-friendly** way. Whether you're a student, teacher, or curious developer — this guide breaks down every step so that anyone can understand how the system learns to fix spelling mistakes.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Part 1: Teaching the Model (train_model.py)](#-part-1-teaching-the-model-train_modelpy)
- [Part 2: Using the Model (predict.py)](#-part-2-using-the-model-predictpy)
- [Key Concepts Explained](#-key-concepts-explained)
- [Visual Flow Diagram](#-visual-flow-diagram)
- [Frequently Asked Questions](#-frequently-asked-questions)

---

## 🌟 Overview

This project uses a technique called **Sequence-to-Sequence (Seq2Seq) Learning** to correct misspelled text. Think of it as a translator — but instead of translating languages, it translates *bad spelling into good spelling*.

The system has two main components:

| Component | File | Purpose |
|---|---|---|
| 🏫 **Trainer** | `src/train_model.py` | Teaches the model by showing it thousands of wrong→correct pairs |
| 🗣️ **Predictor** | `src/predict.py` | Uses the trained model to fix new sentences in real-time |

---

## 🏫 Part 1: Teaching the Model (`train_model.py`)

This file is the **"school"** for our AI. We show it examples of bad spelling paired with correct spelling, and it learns the patterns.

### Step 1: Load the Data

```python
hf_data = load_dataset("torinriley/spell-correction")
```

The model learns from **two sources**:

| Source | Description | Example |
|---|---|---|
| **Hugging Face Dataset** | 30,000 real-world misspelled↔correct pairs from `torinriley/spell-correction` | `"recieve" → "receive"` |
| **Custom Dataset** | 114 hand-curated sentences in `data/generated_dataset.txt` covering academics, business, health, tech, and everyday English | `"i am a student of bachelor of computer applications"` |

### Step 2: Generate Noisy (Misspelled) Data

```python
def add_noise(s):
    # Randomly applies one of three typo strategies
```

The `add_noise()` function takes a correct sentence and introduces **realistic typos** to create training pairs. It randomly picks one of three strategies:

| Strategy | What It Does | Example |
|---|---|---|
| **Replace** | Swaps a random character with another letter | `"apple"` → `"appxe"` |
| **Delete** | Removes a random character | `"apple"` → `"aple"` |
| **Swap** | Transposes two adjacent characters | `"apple"` → `"aplpe"` |

Each custom sentence generates **3 noisy variants**, so the model sees many different mistake patterns for the same correct sentence.

### Step 3: Prepare the Dataset

```python
inputs.append(wrong.lower())
outputs.append(START_TOKEN + correct.lower() + END_TOKEN)
```

The dataset is organized as a list of input→output pairs:

- **Input:** The misspelled sentence
- **Output:** The correct sentence, wrapped with special tokens:
  - `\t` (tab) = **START token** — tells the model "begin generating here"
  - `\n` (newline) = **END token** — tells the model "stop generating here"

Additionally, **identity pairs** (correct→correct) are included so the model learns to leave already-correct text unchanged.

**Total training samples:** ~30,000 from Hugging Face + ~342 from custom dataset (114 × 3 noisy) + 114 identity = **~30,456+ pairs**

### Step 4: Tokenization (Text → Numbers)

```python
tokenizer = Tokenizer(char_level=True, lower=True, filters='')
```

Computers can't read letters — they only understand numbers. The **character-level tokenizer** assigns a unique number to every character:

```
'a' → 1,  'b' → 2,  'c' → 3, ... 'z' → 26,  ' ' → 27,  '\t' → 28,  '\n' → 29
```

> **Why character-level?** Word-level tokenizers fail on unseen or misspelled words. Character-level tokenization handles *any* text, including names, slang, and new words.

### Step 5: Padding (Equal-Length Sequences)

```python
input_seq = pad_sequences(input_seq, maxlen=max_len, padding='post')
```

Neural networks require all inputs to be the **same length**. Shorter sentences are padded with zeros at the end (post-padding):

```
"hello"    → [8, 5, 12, 12, 15, 0, 0, 0, 0, 0]
"good day" → [7, 15, 15, 4, 27, 4, 1, 25, 0, 0]
```

### Step 6: Teacher Forcing

```python
decoder_input_seq = output_seq[:, :-1]   # Everything except the last character
decoder_output_seq = output_seq[:, 1:]    # Everything except the first character
```

During training, the decoder receives the **actual correct output** (shifted by one position) as its input. This is called **teacher forcing** — it's like a teacher whispering the right answer so the student can learn faster.

### Step 7: Build the Model (Encoder-Decoder LSTM)

```python
embedding_dim = 128
latent_dim = 256
```

The model has two brains working together:

| Component | Role | Details |
|---|---|---|
| **Embedding Layer** | Converts character IDs into rich 128-dimensional vectors | Each character gets a "meaning" beyond just its number |
| **Encoder LSTM** | Reads the misspelled input and compresses its meaning into a hidden state | 256 hidden units — captures context and intent |
| **Decoder LSTM** | Receives the encoder's understanding and generates corrected text character by character | Also 256 hidden units |
| **Dense + Softmax** | Converts decoder output into a probability distribution over all characters | Picks the most likely next character |

### Step 8: Train the Model

```python
model.fit(
    [input_seq, decoder_input_seq],
    decoder_output_seq,
    epochs=100,
    batch_size=64
)
```

| Setting | Value | Meaning |
|---|---|---|
| **Epochs** | 100 | The model sees the entire dataset 100 times |
| **Batch Size** | 64 | Processes 64 sentence pairs at a time |
| **Optimizer** | Adam | Adaptive learning rate optimizer |
| **Loss** | Sparse Categorical Crossentropy | Measures how wrong the model's character guesses are |

### Step 9: Save Everything

After training finishes, three files are saved:

| File | Contents |
|---|---|
| `model/spell_model.h5` | Complete trained model weights (~9.2 MB) |
| `model/tokenizer.pkl` | Character-to-number mapping dictionary |
| `model/config.pkl` | Configuration values (e.g., `max_len`) |

---

## 🗣️ Part 2: Using the Model (`predict.py`)

This is where the trained model goes to work! You type a misspelled sentence, and it fixes it in real-time.

### Step 1: Load the Trained Model

```python
model = load_model("../model/spell_model.h5")
tokenizer = pickle.load(open("../model/tokenizer.pkl", "rb"))
config = pickle.load(open("../model/config.pkl", "rb"))
```

The saved model, tokenizer, and config are loaded from disk. The model is now ready to make predictions.

### Step 2: Protected Words System

```python
IMPORTANT_WORDS = ["livesh", "bca", "ai"]

def protect_words(sentence):
    return any(word in sentence.lower() for word in IMPORTANT_WORDS)
```

Some words should **never** be "corrected" — names, abbreviations, and technical terms. If the input contains any protected word, the model returns the original sentence unchanged.

### Step 3: Memory Cache System

```python
MEMORY_FILE = "../memory/word_memory.json"

if sentence in word_memory:
    return word_memory[sentence]  # Instant lookup — no model inference needed!
```

The model maintains a **diary** (`word_memory.json`). Whenever it corrects a sentence, it writes the result down. If the same sentence is submitted again:

- ✅ Returns the cached answer **instantly** (no GPU/CPU computation needed)
- ✅ Reduces response time from ~1 second to ~0 milliseconds
- ✅ Memory file persists across restarts

### Step 4: Autoregressive Decoding

```python
def predict_sentence(sentence):
    # 1. Tokenize and pad the input
    # 2. Feed to encoder
    # 3. Start decoder with START_TOKEN
    # 4. Generate one character at a time until END_TOKEN
```

This is the core prediction loop. Here's what happens step by step:

1. **Encode** — The input sentence is converted to numbers, padded, and fed through the encoder LSTM
2. **Initialize Decoder** — The decoder starts with the `START_TOKEN` (`\t`)
3. **Generate Loop** — For each step:
   - The decoder predicts a probability distribution over all characters
   - The character with the **highest probability** is selected (`argmax`)
   - If it's the `END_TOKEN` (`\n`), generation stops
   - Otherwise, the character is appended to the result and fed back as input for the next step
4. **Output** — All generated characters are joined together to form the corrected sentence

### Step 5: Safety Filter

```python
if len(output) < len(sentence) * 0.7:
    return sentence  # Reject — output is suspiciously short
```

If the model's output is **less than 70%** the length of the input, it's likely garbage. The original sentence is returned instead. This prevents the model from producing nonsensical short outputs.

### Step 6: Save & Return

```python
word_memory[sentence] = output
save_memory()
return output
```

The corrected sentence is saved to the memory file and returned to the user.

---

## 📖 Key Concepts Explained

### What is Seq2Seq?
**Sequence-to-Sequence** is a model architecture that takes a sequence of items (characters, words) as input and produces another sequence as output. It was originally designed for machine translation (e.g., English → French), but works perfectly for spell correction (misspelled → correct).

### What is LSTM?
**Long Short-Term Memory** is a type of recurrent neural network (RNN) that can remember information over long sequences. Unlike basic RNNs, LSTMs don't "forget" important information from earlier in the sentence.

### What is an Embedding?
An embedding converts discrete items (like characters) into continuous vectors. Instead of representing "a" as just the number 1, it becomes a **128-dimensional vector** that captures relationships between characters.

### What is Teacher Forcing?
During training, instead of allowing the decoder to generate freely (which would accumulate errors), we feed it the **actual correct output** shifted by one position. This lets the model learn much faster.

### What is Autoregressive Decoding?
During prediction (inference), the model generates output **one character at a time**. Each generated character is fed back as input for the next step. This continues until the model predicts the `END_TOKEN`.

---

## 🔄 Visual Flow Diagram

```
USER INPUT: "i am goimg to schol"
       │
       ▼
┌──────────────────────────┐
│  Is a protected word?    │──── YES ──▶ Return input unchanged
└──────────┬───────────────┘
           │ NO
           ▼
┌──────────────────────────┐
│  Found in memory cache?  │──── YES ──▶ Return cached result
└──────────┬───────────────┘
           │ NO
           ▼
┌──────────────────────────┐
│  Tokenize + Pad          │
│  "i am goimg..." → [9,   │
│   1, 13, 27, 7, ...]     │
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  Encoder LSTM             │
│  → Hidden State (256-d)  │
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  Decoder LSTM             │
│  START → 'i' → ' ' →    │
│  'a' → 'm' → ' ' →     │
│  'g' → 'o' → 'i' →     │
│  'n' → 'g' → ... → END │
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  Safety Check            │
│  len("i am going to      │
│  school") ≥ 70% of input │
└──────────┬───────────────┘
           │ PASS
           ▼
┌──────────────────────────┐
│  Save to Memory + Return │
│  → "i am going to school"│
└──────────────────────────┘
```

---

## ❓ Frequently Asked Questions

**Q: Does the model need internet to make predictions?**  
A: No. Once trained, the model runs completely offline. Internet is only needed during training to download the Hugging Face dataset.

**Q: Can it correct grammar?**  
A: It primarily focuses on spelling corrections, but since it learns on full sentences, it picks up some basic grammar patterns too.

**Q: Why character-level instead of word-level?**  
A: Word-level models fail on unseen or misspelled words (OOV errors). Character-level models can handle any text — including names, slang, and invented words.

**Q: How do I add new protected words?**  
A: Edit the `IMPORTANT_WORDS` list in `src/predict.py`:
```python
IMPORTANT_WORDS = ["livesh", "bca", "ai", "your_word_here"]
```

**Q: Can I add more training data?**  
A: Yes! Add new sentences to `data/generated_dataset.txt` (one per line), then retrain the model.

**Q: How do I reset the memory cache?**  
A: Delete `memory/word_memory.json`. It will be recreated automatically on the next prediction.

**Q: What Python version do I need?**  
A: Python 3.8 or higher is recommended. TensorFlow requires Python 3.8–3.11.

---

**That's it! You now understand how the entire Neural Spell Correction system works — from training to inference! 🎉**

<p align="center">
  Made with ❤️ by <strong>IST Soft Game Studio</strong> | Author: <strong>Livesh</strong>
</p>
