import numpy as np
import json
import os
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================
# LOAD MODEL + CONFIG
# =========================
model = load_model("../model/spell_model.h5")

with open("../model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# ✅ Load max_len safely
with open("../model/config.pkl", "rb") as f:
    config = pickle.load(f)

max_len = config["max_len"]

# =========================
# MEMORY
# =========================
MEMORY_FILE = "../memory/word_memory.json"

if os.path.exists(MEMORY_FILE):
    try:
        with open(MEMORY_FILE, "r") as f:
            word_memory = json.load(f)
    except json.JSONDecodeError:
        print("⚠️ Memory file corrupted. Resetting...")
        word_memory = {}
else:
    word_memory = {}

def save_memory():
    with open(MEMORY_FILE, "w") as f:
        json.dump(word_memory, f, indent=4)

# =========================
# CONFIG
# =========================
START_TOKEN = "\t"
END_TOKEN = "\n"

reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}

# =========================
# EXTRA SAFETY
# =========================
IMPORTANT_WORDS = ["livesh", "bca", "ai"]

def protect_words(sentence):
    return any(word in sentence.lower() for word in IMPORTANT_WORDS)

# =========================
# PREDICT
# =========================
def predict_sentence(sentence):

    sentence = sentence.lower().strip()

    # ✅ Protect important words
    if protect_words(sentence):
        return sentence

    # ✅ Memory check
    if sentence in word_memory:
        return word_memory[sentence]

    # ✅ Limit length
    if len(sentence) > max_len:
        sentence = sentence[:max_len]

    seq = tokenizer.texts_to_sequences([sentence])
    seq = pad_sequences(seq, maxlen=max_len, padding='post')

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index.get(START_TOKEN, 1)

    result = []

    for _ in range(max_len):
        pred = model.predict([seq, target_seq], verbose=0)
        next_id = np.argmax(pred[0, -1, :])

        if next_id == tokenizer.word_index.get(END_TOKEN):
            break

        char = reverse_word_index.get(next_id, '')
        result.append(char)

        target_seq = np.append(target_seq, [[next_id]], axis=1)

    output = "".join(result).strip()

    # ✅ Strong safety filter
    if len(output) < len(sentence) * 0.7:
        return sentence

    # ✅ Save memory
    word_memory[sentence] = output
    save_memory()

    return output

# =========================
# TEST
# =========================
if __name__ == "__main__":
    while True:
        text = input("Enter sentence: ")
        print("Output:", predict_sentence(text))