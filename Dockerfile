FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download fr_core_news_sm

COPY rakuten_api.py .
COPY model.pkl .
COPY tfidf.pkl .

RUN python -c "import nltk; nltk.download('stopwords')"

EXPOSE 8000

CMD ["uvicorn", "rakuten_api:app", "--host", "0.0.0.0", "--port", "8000"]