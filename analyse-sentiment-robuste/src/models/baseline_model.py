from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def build_baseline_pipeline(seed=42):
    """
    Construit et retourne un pipeline Sklearn (TF-IDF + Régression Logistique).
    """
    print("Construction du pipeline baseline (TF-IDF + LogReg)...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000, 
            ngram_range=(1, 2),
            min_df=5 # Ignorer les termes trop rares
        )),
        ('clf', LogisticRegression(
            C=1.0, 
            solver='liblinear', 
            random_state=seed
        ))
    ])
    return pipeline