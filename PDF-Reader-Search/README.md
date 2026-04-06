<div align="center">

# 🔎 Professional PDF Reader & Search Engine
### Developed by Soft Game Studio

*A robust, modular command-line utility designed for intelligent PDF text mapping, paragraph restructuring, and high-speed keyword retrieval.*

</div>

---

## 🚀 Overview

Standard PDF reading applications often extract raw strings fraught with fractured lines and broken spacing, making data query a tedious process. The **Soft Game Studio PDF Reader & Search Engine** transcends native limitations by structurally re-formatting text blocks based on punctuation syntax, and deploying highly responsive document-wide search queries via an integrated terminal interface.

## ✨ Core Features

- **Punctuation-Aware Formatter:** Rebuilds fragmented PDF string exports into continuous, human-readable paragraph structures by recognizing `.`, `!`, and `?` characters.
- **High-Speed Query Engine:** Instantly maps case-insensitive target keywords and returns exact page and line-level citations.
- **Interactive CLI Native Hub:** Easily toggle between document reading streams and deep search indices natively from your terminal.
- **Enterprise-Grade Modularity:** Clean separation of logic (`reader.py`, `formatter.py`, `search.py`) tailored for scalable pipeline integration.
- **100% Offline Local Security:** Guaranteed absolute data privacy. Zero external dependencies, utilizing in-memory caches exclusively.

## 📦 Installation

Ensure your Python environment is set up appropriately, then install the structural dependencies:

```bash
pip install -r requirements.txt
```

## 🛠️ Usage

1. Place your target `.pdf` files into the working directory.
2. Initialize the application hub:
   ```bash
   python src/main.py
   ```
3. A menu will appear in your console:
   - Provide the path to the loaded PDF (e.g., `sample.pdf`).
   - Select option `1` to stream a clean, reformatted text output per-page.
   - Select option `2` to execute rapid keyword indexing.
   - Select option `3` to exit the environment.

## 📂 Architecture Stack
For an in-depth breakdown of module responsibilities (`reader`, `formatter`, `search`), please check our [Technical Documentation](docs/PDF-Reader-Search.md).

---

<div align="center">

**[Soft Game Studio](https://github.com/SoftGameStudio)**
*Engineering state-of-the-art software systems for scalable enterprise ecosystems.*

</div>