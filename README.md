## Flipkart Review Sentiment Analysis
### Setup
- `python -m venv .venv && source .venv/bin/activate`
- `pip install -r requirements.txt`
- Download NLTK data once:
  ```python
  import nltk
  nltk.download("stopwords")
  nltk.download("wordnet")
  nltk.download("omw-1.4")
  ```

### Data
- Place raw CSV at `data/raw/reviews_data.csv` with columns including `Rating`, `Review Title`, `Review Text`, and optionally `Up Votes`.

### Notebook (EDA + Training)
- Open `notebooks/1_eda.ipynb` and run sequentially:
  - Build sentiment label (rating >= 4 positive, <= 2 negative; drop 3s).
  - EDA: rating distribution, pos/neg split, negative keywords, upvotes vs sentiment.
  - Preprocess text, create `clean_text`, and save `data/processed/clean_reviews.csv`.
  - Fit TF-IDF and Logistic Regression; note F1 from the classification report.
  - Artifacts saved to `models/model.pkl` and `models/vectorizer.pkl`.

### Streamlit App
- After training artifacts exist:
  - `streamlit run app.py`
  - Enter a review to see sentiment and confidence.

### Notes
- To try more models (Naive Bayes, Linear SVM, RandomForest, LSTM/BERT), add experiments in new notebooks and reuse the saved vectorizer/tokenizer paths in `models/`.

### MLflow (Experiment Tracking)
- Notebook: `notebooks/mlflow_runs.ipynb` (logs TF-IDF + Logistic Regression runs).
- Run the logging cells (or the provided script) after activating the venv.
- Launch UI locally to inspect runs/metrics/artifacts:
  ```
  mlflow ui --backend-store-uri ./mlruns --port 5000
  ```
  Then open http://127.0.0.1:5000 and look for experiment `flipkart-sentiment`.
