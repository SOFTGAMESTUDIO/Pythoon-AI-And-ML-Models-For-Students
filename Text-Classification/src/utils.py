import re
import string

def preprocess_text(text: str) -> str:
    """
    Cleans the input text by lowercasing, removing punctuation, 
    and stripping extra whitespace.
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove non-alphabetical characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Strip extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def label_to_int(label: str) -> int:
    """Maps string label to integer."""
    label = str(label).strip().lower()
    if label == 'spam':
        return 1
    return 0

def int_to_label(pred: int) -> str:
    """Maps integer prediction back to label."""
    if pred == 1:
        return 'spam'
    return 'ham'