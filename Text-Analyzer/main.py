from src.cleaner import clean_text
from src.processor import process_text
from src.exporter import save_to_csv

def load_text(path):
    with open(path, "r") as f:
        return f.read()

def main():
    text = load_text("data/sample.txt")

    cleaned = clean_text(text)
    data, entities = process_text(cleaned)

    save_to_csv(data)

    print("\nProcessed Data:\n", data[:5])
    print("\nEntities:\n", entities)

if __name__ == "__main__":
    main()