# Utiliser une image de base Python
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les dépendances
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Télécharger les ressources NLTK nécessaires (optionnel, si 'clean_data' est exécuté dans le conteneur)
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader wordnet
RUN python -m nltk.downloader omw-1.4

# Copier le code source et les artefacts
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY artifacts/ ./artifacts/

# Point d'entrée par défaut (exemple : lancer une évaluation)
CMD ["python", "scripts/6_evaluate_robustness.py"]