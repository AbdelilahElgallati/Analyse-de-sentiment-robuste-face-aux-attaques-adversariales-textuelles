import pandas as pd
import argparse
import joblib
from pathlib import Path
import sys
from sklearn.metrics import classification_report

# Ajouter la racine du projet pour importer 'src'
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.models.baseline_model import build_baseline_pipeline

def main(args):
    """
    Entraîne un modèle baseline (TF-IDF + Régression Logistique).
    """
    train_path = Path(args.train_data)
    val_path = Path(args.val_data)
    model_output_path = Path(args.output_model)
    
    print("Chargement des données train/val...")
    try:
        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path)
    except FileNotFoundError as e:
        print(f"Erreur: Fichier de données non trouvé. {e}")
        print("Veuillez exécuter les scripts 1_clean_data.py et 2_split_data.py d'abord.")
        return
    
    X_train, y_train = train_df['cleaned_text'], train_df['label']
    X_val, y_val = val_df['cleaned_text'], val_df['label']
    
    # Construire le pipeline depuis src/models/
    pipeline = build_baseline_pipeline(seed=args.seed)
    
    print("Entraînement du modèle...")
    pipeline.fit(X_train, y_train)
    
    print("\n" + "="*30)
    print("Évaluation sur le set de validation (Clean)")
    print("="*30)
    y_pred = pipeline.predict(X_val)
    print(classification_report(y_val, y_pred, target_names=['negative', 'positive']))
    
    print(f"Sauvegarde du pipeline modèle dans {model_output_path}...")
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_output_path)
    
    print("Entraînement baseline terminé.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîner un modèle baseline (TF-IDF + LogReg).")
    parser.add_argument("--train_data", type=str, default="data/processed/train.parquet")
    parser.add_argument("--val_data", type=str, default="data/processed/val.parquet")
    parser.add_argument(
        "--output_model", 
        type=str, 
        default="artifacts/models/baseline_logreg.joblib"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)