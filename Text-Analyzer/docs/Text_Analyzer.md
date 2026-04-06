# Text Analyzer Internal Documentation

This document explains the internal modules and functions powering the Text Analyzer application. The application's core logic is cleanly separated into the `src/` directory, dividing its functionality into cleaning, processing, and exporting data.

## 1. `src/cleaner.py`
This module contains functionality for pre-processing text data to normalize it before Natural Language Processing.

### `clean_text(text: str) -> str`
- **Purpose**: Sanitizes and prepares raw text inputs.
- **Behavior**: 
  - Converts all characters to lowercase.
  - Applies a Regular Expression `[^a-zA-Z\s]` to strip away any numerical digits and punctuation marks, ensuring only alphabetical characters and spaces remain.
- **Returns**: The cleaned, normalized text string.

## 2. `src/processor.py`
This module houses the core Natural Language Processing logic utilizing the `spaCy` library. It loads the `en_core_web_sm` pre-trained NLP pipeline.

### `process_text(text: str) -> Tuple[List[dict], List[Tuple[str, str]]]`
- **Purpose**: Breaks down text into tokens and extracts meaningful metadata, such as lemmas, Parts-of-Speech, and Named Entities.
- **Behavior**:
  - Processes the text through the active SpaCy pipeline.
  - Iterates over each token. It specifically filters out stopwords and non-alphabetical tokens (using `token.is_stop` and `token.is_alpha`).
  - Constructs a list of parsed tokens with their origin word, lemma form (`token.lemma_`), and Part of Speech tag (`token.pos_`).
  - Independently loops through `doc.ents` to extract recognized Named Entities alongside their label categories.
- **Returns**: 
  - `data`: A list of dictionaries representing valid tokens.
  - `entities`: A list of tuples pairing respective entity names with their entity tags.

## 3. `src/exporter.py`
This module is responsible for persisting the analyzed output datasets directly to disk.

### `save_to_csv(data: List[dict], path: str = "output/results.csv")`
- **Purpose**: Saves the token data dictionaries into a formalized CSV file mapping for convenient distribution or reviewing.
- **Behavior**:
  - Automatically verifies and synthesizes the export output directory (e.g. `output/`) via `os.makedirs` if it does not yet exist.
  - Loops over the mapped dictionary lines to append structured columns (`word`, `lemma`, `pos`) encoded securely into CSV form.
- **Returns**: None (Outputs the artifact directly to `output/results.csv`).

## Integration Scripts
- **`app.py`**: A Streamlit user-interface that leverages the above modules. It dynamically inputs user text, invokes `clean_text()`, analyzes it via `process_text()`, writes metrics to the user screen, and persists to file via `save_to_csv()`.
- **`main.py`**: A secondary terminal pipeline. It imports string configurations strictly from `data/sample.txt` and drives the identical pre-processing loops directly to `.csv` export and stdout terminal display.
