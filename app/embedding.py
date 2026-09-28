import spacy

nlp = spacy.load("en_core_web_md")


def get_embedding(word):
    doc = nlp(word)
    return doc.vector.tolist()

