import pandas as pd
import argparse
from pathlib import Path
import sys

# Ajouter le dossier parent (racine du projet) au sys.path
# Permet d'importer depuis 'src'
sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from src.preprocessing.text_cleaner import full_preprocessing_pipeline
except ImportError:
    print("Erreur: Impossible d'importer 'src.preprocessing.text_cleaner'.")
    print("Assurez-vous d'exécuter ce script depuis la racine du projet ou que 'src' est dans le PYTHONPATH.")
    sys.exit(1)

def main(args):
    """
    Script principal pour nettoyer les données brutes.
    Prend un CSV en entrée, applique le nettoyage, et sauvegarde en Parquet.
    """
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    
    if not input_path.exists():
        print(f"Erreur : Fichier d'entrée non trouvé à {input_path}")
        return
        
    print(f"Chargement des données brutes depuis {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Erreur lors de la lecture du CSV: {e}")
        return
    
    # Hypothèse : colonnes 'review' et 'sentiment' (basé sur le notebook IMDB)
    if 'review' not in df.columns or 'sentiment' not in df.columns:
        print(f"Erreur : Le CSV doit contenir les colonnes 'review' et 'sentiment'. Colonnes trouvées: {df.columns}")
        return
        
    print("Nettoyage et normalisation du texte... (cela peut prendre quelques minutes)")
    df['cleaned_text'] = full_preprocessing_pipeline(df['review'])
    
    # Mapper les sentiments en 0/1 (Négatif/Positif)
    df['label'] = df['sentiment'].map({'negative': 0, 'positive': 1})
    
    if df['label'].isnull().any():
        print("Avertissement : Certaines valeurs de 'sentiment' n'étaient ni 'positive' ni 'negative' et ont été converties en NaN.")
    
    # Garder uniquement les colonnes nécessaires
    final_df = df[['cleaned_text', 'label']].dropna(subset=['cleaned_text', 'label'])
    final_df['label'] = final_df['label'].astype(int)
    
    print(f"Sauvegarde des données nettoyées dans {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(output_path, index=False)
    
    print(f"Nettoyage terminé. {len(final_df)} lignes sauvegardées.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nettoyer les données textuelles brutes.")
    parser.add_argument(
        "--input_file", 
        type=str, 
        default="data/raw/IMDB Dataset.csv", 
        help="Chemin vers le fichier CSV brut."
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="data/processed/cleaned_data.parquet", 
        help="Chemin pour sauvegarder le fichier Parquet nettoyé."
    )
    args = parser.parse_args()
    main(args)