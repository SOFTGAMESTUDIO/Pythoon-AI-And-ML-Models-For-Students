import spacy

nlp = spacy.load("en_core_web_sm")

def process_text(text):
    doc = nlp(text)

    data = []

    for token in doc:
        if not token.is_stop and token.is_alpha:
            data.append({
                "word": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_
            })

    entities = [(ent.text, ent.label_) for ent in doc.ents]

    return data, entities