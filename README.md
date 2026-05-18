# Rakuten MLOps — Classification Multimodale

## Description
Classification de produits e-commerce Rakuten à partir de données textuelles.
Prédit le code type d'un produit (prdtypecode) depuis sa désignation et description.

## Stack technique
- **Preprocessing** : BeautifulSoup, NLTK, spaCy
- **Modèle** : TF-IDF + Régression Logistique (Weighted F1 = 0.8056)
- **API** : FastAPI + Uvicorn
- **Tracking** : MLflow
- **Conteneurisation** : Docker + Docker Compose
- **CI/CD** : GitHub Actions

## Lancer le projet

### Avec Docker Compose
```bash
docker-compose up --build
```

### API disponible sur
- http://localhost:8000 — API FastAPI
- http://localhost:8080 — MLflow UI

## Résultats
| Modèle | Weighted F1 |
|--------|-------------|
| TF-IDF + LogisticRegression | 0.8056 |
| Benchmark Rakuten | 0.8100 |