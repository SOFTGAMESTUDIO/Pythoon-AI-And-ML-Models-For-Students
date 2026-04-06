# 🔍 Professional PDF Reader & Search Engine — Technical Documentation

> **Developed by Soft Game Studio**

---

## 1. System Overview

The **PDF Reader & Search Engine** is a high-performance, command-line utility built by *Soft Game Studio*. It is engineered to seamlessly parse PDF documents, structure raw text strings into readable paragraphs, and execute lightning-fast keyword queries across the entire document corpus.

This tool prioritizes speed, modularity, and clean formatting, avoiding the common pitfalls of disjointed PDF text extraction by actively monitoring sentence terminologies.

## 2. Core Architecture & Module Breakdown

The architecture follows a strict modular design pattern. Below is an in-depth explanation of every file and function within the `src/` directory.

### 2.1 `src/main.py`
The orchestration layer of the application. It provides an intuitive loop-driven command-line interface (CLI) for user interactions.

- **`main()` Function**: 
  - **Purpose**: Initializes the application loop and handles user navigation.
  - **Working Mechanism**: 
    - Prompts the user for a filesystem path to the target PDF, then triggers `read_pdf()`. 
    - It enters an infinite `while` loop offering a distinct menu of operations: 
      1. **Read PDF**: Iterates through the document pages. It calls `format_text()` to normalize the text and streams the processed output linearly to the console, neatly separated by standard page markers.
      2. **Search**: Prompts for a target string and calls `search_keyword()`. Evaluates the returned dict array and maps results sequentially, showing exactly what precise string line and structural page the keyword was discovered on.
      3. **Exit**: Halts application state and safely terminates execution.

### 2.2 `src/reader.py`
The ingestion engine. This isolates direct file I/O interactions representing best practices for modular stability.

- **`read_pdf(pdf_path)` Function**:
  - **Purpose**: To handle binary container mapping of the document using PyMuPDF.
  - **Working Mechanism**: Initializes a `fitz` document instance mapping locally. It iteratively loads each physical page sequentially and forces an aggressive `.get_text()` extraction. Captured plaintext is stacked onto an array logic stream where its index seamlessly relates directly to the parent page count parameters, building an optimized in-memory layout cache.

### 2.3 `src/formatter.py`
The natural language cleaner. A standard PDF usually exports sentences severed via arbitrary hardline breaks (`\n`) resulting from visual rendering widths.

- **`format_text(text)` Function**:
  - **Purpose**: To algorithmically rebuild fractured PDF strings into smooth, logical paragraphs.
  - **Working Mechanism**:
    - Divides raw paragraph text using newlines. 
    - Analyzes and trims white spaces sequentially.
    - Importantly, the logic probes string suffixes for terminal punctuation metrics (`.`, `!`, `?`). 
    - Unresolved string tokens are continually joined within a mutable cache container until a matching suffix trigger is flagged. Once a sentence is properly terminated functionally, it binds the assembled paragraph to a master array. 
    - Concluding the function, it connects these paragraphs intelligently using a wide-spaced double structural break (`\n\n`) creating optimal terminal legibility.

### 2.4 `src/search.py`
The high-speed analytical indexing module for query targeting. 

- **`search_keyword(pages, keyword)` Function**:
  - **Purpose**: Resolves rapid target string parameters across the active cache.
  - **Working Mechanism**: Bootstraps the `pages` string iteration engine leveraging the inputted keyword. Scanning procedurally layer-by-layer, it leverages native `\n` splits. By asserting a `.lower()` casting parameter against both the target token and the evaluation line string simultaneously, querying becomes flawlessly case-agnostic. Whenever successful matching intersects, it formulates and appends a structured python dictionary mapped conceptually as `{"page": integer_ref, "line": text_string}` returning it upstream to the UI.

## 3. Core Advantages & Stability
- **Engineered Formatting Execution**: Drastically outperforms regular PyMuPDF prints by repairing native human grammar flow syntaxes.
- **In-Memory Analytics Traversal**: Off-loads active workload from local hard drives placing data exclusively inside RAM for high velocity query capabilities.
- **Extensible Modularity Focus**: Rigidly implements single-responsibility concepts meaning future API or Interface expansions require zero structural breakage.

---

*(c) Copyright Soft Game Studio | Engineering state-of-the-art software intelligence utilities.*
