from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import spacy
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

# Chargement des modèles
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")
nlp = spacy.load("fr_core_news_sm")
stop_words = set(stopwords.words("french"))

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Bienvenue sur l'API Rakuten"}

def clean_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_stopwords(text):
    mots = text.split()
    mots_filtre = [mot for mot in mots if mot not in stop_words]
    return " ".join(mots_filtre)

def lemmatize(text):
    doc = nlp(text)
    lemmes = [token.lemma_ for token in doc]
    return " ".join(lemmes)

@app.post("/predict")
def predict(input: TextInput):
    text = input.text
    text = clean_html(text)
    text = text.lower()
    text = remove_stopwords(text)
    text = lemmatize(text)
    text_tfidf = tfidf.transform([text])
    prediction = model.predict(text_tfidf)
    return {"categorie": int(prediction[0])}