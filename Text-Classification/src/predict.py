import torch
import pickle
import os

from src.utils import preprocess_text, int_to_label
from src.model import SpamClassifier

def load_system(model_path='models/saved_model.pth', vectorizer_path='models/vectorizer.pkl'):
    """Loads both the saved TF-IDF vectorizer and the PyTorch Model model."""
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Vectorizer not found at {vectorizer_path}. Please run train.py first.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please run train.py first.")
        
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
        
    input_dim = len(vectorizer.vocabulary_)
    model = SpamClassifier(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    return model, vectorizer

def predict(text: str, model, vectorizer):
    """Takes input string, processes it consistently and pushes to Neural Net logic."""
    # Ensure exact same preprocessing as Training Phase!
    cleaned_text = preprocess_text(text)
    
    # Handle completely unknown edge cases (empty strings post-process)
    if not cleaned_text.strip():
        # Fallback behaviour for completely unrecognized text
        return "ham", 0.0
        
    # Scale via trained vectorizer vocabulary limits
    vec = vectorizer.transform([cleaned_text])
    tensor_input = torch.tensor(vec.toarray(), dtype=torch.float32)
    
    # Run evaluation calculation
    with torch.no_grad():
        output = model(tensor_input).item()
        
    # Standard format metrics
    confidence = output if output > 0.5 else 1 - output
    predicted_label = int_to_label(1 if output > 0.5 else 0)
    
    return predicted_label, confidence

if __name__ == '__main__':
    try:
        model, vectorizer = load_system()
        print("\n--- Model Loaded Successfully ---")
        print("Type 'exit' or 'quit' to close the engine.\n")
        
        while True:
            text = input("Enter SMS/Email text: ")
            if text.strip().lower() in ['exit', 'quit']:
                break
                
            label, confidence = predict(text, model, vectorizer)
            print(f"Prediction: {label.upper()} (Confidence Score: {confidence:.2%})\n")
            
    except Exception as e:
        print(f"Error: {e}")