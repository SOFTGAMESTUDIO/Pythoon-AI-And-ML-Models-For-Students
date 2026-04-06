
def save_to_csv(data, path="output/results.csv"):
    import os

    os.makedirs("output", exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write("word,lemma,pos\n")

        for row in data:
            f.write(f"{row['word']},{row['lemma']},{row['pos']}\n")