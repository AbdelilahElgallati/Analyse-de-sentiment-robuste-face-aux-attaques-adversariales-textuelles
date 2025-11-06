import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from functools import lru_cache

@lru_cache(maxsize=None)
def get_nltk_resources():
    """
    Vérifie et télécharge les ressources NLTK requises de manière idempotente.
    """
    resources = {
        'corpora/stopwords': 'stopwords',
        'corpora/wordnet': 'wordnet',
        'corpora/omw-1.4': 'omw-1.4'
    }
    for path, pkg_id in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Ressource NLTK '{pkg_id}' non trouvée. Téléchargement...")
            nltk.download(pkg_id)
    
    return set(stopwords.words('english')), WordNetLemmatizer()

def clean_text(text):
    """
    Applique un nettoyage de base au texte.
    - Supprime HTML
    - Supprime URLs
    - Met en minuscule
    - Supprime la ponctuation et les chiffres
    """
    if not isinstance(text, str):
        return ""
        
    # Supprimer les balises HTML
    text = re.sub(r'<[^>]+>', '', text)
    # Supprimer les URLs
    text = re.sub(r'http\S+|www\S+', '', text, flags=re.MULTILINE)
    # Mettre en minuscule
    text = text.lower()
    # Supprimer la ponctuation et les chiffres (garder que les lettres et espaces)
    text = re.sub(r'[^a-z\s]', '', text)
    # Supprimer les espaces blancs excessifs
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_text(text):
    """
    Applique la lemmatisation et la suppression des stopwords.
    """
    stop_words, lemmatizer = get_nltk_resources()
    
    tokens = text.split()
    normalized_tokens = [
        lemmatizer.lemmatize(token) for token in tokens 
        if token not in stop_words and len(token) > 2
    ]
    return ' '.join(normalized_tokens)

def full_preprocessing_pipeline(text_series):
    """Applique le pipeline de nettoyage et normalisation complet à une Série pandas."""
    get_nltk_resources() # Assure que les ressources sont prêtes
    cleaned_series = text_series.astype(str).apply(clean_text)
    normalized_series = cleaned_series.apply(normalize_text)
    return normalized_series