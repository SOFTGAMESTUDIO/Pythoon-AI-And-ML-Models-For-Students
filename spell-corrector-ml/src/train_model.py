import numpy as np
import random
import pickle
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datasets import load_dataset

# =========================
# 1. LOAD DATA
# =========================
print("Loading dataset...")

hf_data = load_dataset("torinriley/spell-correction")
hf_pairs = [(x['misspelled'], x['correct']) for x in hf_data['train']]

# =========================
# LOAD CUSTOM DATASET
# =========================
def load_generated_dataset(path):
    cleaned = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().lower()

            if not line:
                continue
            if "/" in line:
                continue
            if len(line) < 3:
                continue

            cleaned.append(line)

    return cleaned

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "..", "data", "generated_dataset.txt")

base_sentences = load_generated_dataset(data_path)

# =========================
# 2. NOISE FUNCTION
# =========================
def add_noise(s):
    chars = list(s)

    if len(chars) > 4:
        i = random.randint(0, len(chars) - 2)
        choice = random.choice(["replace", "delete", "swap"])

        if choice == "replace":
            chars[i] = random.choice("abcdefghijklmnopqrstuvwxyz")
        elif choice == "delete":
            chars.pop(i)
        elif choice == "swap" and i < len(chars) - 1:
            chars[i], chars[i+1] = chars[i+1], chars[i]

    return "".join(chars)

# =========================
# 3. CREATE DATASET
# =========================
print("Preparing dataset...")

inputs = []
outputs = []

START_TOKEN = "\t"
END_TOKEN = "\n"

MAX_SAMPLES = 30000

# ✅ HuggingFace data
for wrong, correct in hf_pairs[:MAX_SAMPLES]:
    inputs.append(wrong.lower())
    outputs.append(START_TOKEN + correct.lower() + END_TOKEN)

# ✅ Custom dataset
for s in base_sentences[:MAX_SAMPLES]:
    for _ in range(3):
        inputs.append(add_noise(s))
        outputs.append(START_TOKEN + s + END_TOKEN)

# ✅ Identity training (ONLY ONCE)
for s in base_sentences:
    inputs.append(s)
    outputs.append(START_TOKEN + s + END_TOKEN)

print("Total samples:", len(inputs))

# =========================
# 4. TOKENIZER
# =========================
tokenizer = Tokenizer(char_level=True, lower=True, filters='')
tokenizer.fit_on_texts(inputs + outputs)

input_seq = tokenizer.texts_to_sequences(inputs)
output_seq = tokenizer.texts_to_sequences(outputs)

max_len = max(
    max(len(x) for x in input_seq),
    max(len(x) for x in output_seq)
)

input_seq = pad_sequences(input_seq, maxlen=max_len, padding='post')
output_seq = pad_sequences(output_seq, maxlen=max_len, padding='post')

vocab_size = len(tokenizer.word_index) + 1

print("Vocab size:", vocab_size)
print("Max length:", max_len)

# =========================
# 5. TEACHER FORCING
# =========================
decoder_input_seq = output_seq[:, :-1]
decoder_output_seq = output_seq[:, 1:]
decoder_output_seq = np.expand_dims(decoder_output_seq, -1)

# =========================
# 6. MODEL
# =========================
print("Building model...")

embedding_dim = 128
latent_dim = 256

encoder_inputs = Input(shape=(max_len,))
enc_emb = Embedding(vocab_size, embedding_dim)(encoder_inputs)
_, state_h, state_c = LSTM(latent_dim, return_state=True)(enc_emb)

decoder_inputs = Input(shape=(None,))
dec_emb = Embedding(vocab_size, embedding_dim)(decoder_inputs)

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=[state_h, state_c])

decoder_outputs = Dense(vocab_size, activation='softmax')(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy'
)

# =========================
# 7. TRAIN
# =========================
print("Training started...")

model.fit(
    [input_seq, decoder_input_seq],
    decoder_output_seq,
    epochs=100,
    batch_size=64
)

# =========================
# 8. SAVE MODEL + CONFIG
# =========================
print("Saving model...")

model.save("../model/spell_model.h5")

with open("../model/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# ✅ Save max_len
with open("../model/config.pkl", "wb") as f:
    pickle.dump({"max_len": max_len}, f)

print("✅ Training complete & model saved!")