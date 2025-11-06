import pandas as pd
import argparse
import joblib
from pathlib import Path
import sys
from tqdm import tqdm

# Ajouter la racine du projet pour importer 'src'
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.models.baseline_model import build_baseline_pipeline
from src.evaluation.metrics import load_textattack_wrapper

from textattack.attack_recipes import TextFoolerJin2019
from textattack.augmentation import Augmenter
from sklearn.metrics import classification_report

def main(args):
    """
    Effectue un adversarial training pour le modèle baseline.
    1. Charge le modèle baseline.
    2. Génère des exemples adversariaux sur le set d'entraînement.
    3. Augmente le set d'entraînement avec ces exemples.
    4. Ré-entraîne un nouveau modèle sur les données augmentées.
    """
    train_path = Path(args.train_data)
    val_path = Path(args.val_data)
    baseline_model_path = Path(args.baseline_model)
    robust_model_output_path = Path(args.output_robust_model)
    
    print(f"Chargement du modèle baseline {baseline_model_path} pour l'augmentation...")
    try:
        model_wrapper = load_textattack_wrapper("sklearn", baseline_model_path)
    except FileNotFoundError:
        print(f"Erreur: Modèle baseline non trouvé à {baseline_model_path}")
        print("Veuillez d'abord exécuter '3_train_baseline.py'.")
        return
    
    # 1. Définir l'augmenter (basé sur une recette d'attaque)
    print("Configuration de l'augmenter (TextFooler)...")
    augmenter = Augmenter(
        transformation=TextFoolerJin2019.build(model_wrapper).transformation,
        constraints=TextFoolerJin2019.build(model_wrapper).constraints,
        pct_words_to_swap=0.1, # Augmenter légèrement (plus rapide)
        transformations_per_example=1 # 1 seul exemple augmenté par original
    )
    
    print(f"Chargement des données d'entraînement {train_path}...")
    train_df = pd.read_parquet(train_path)
    
    # Prendre un sous-échantillon pour l'augmentation
    if args.num_augment > 0 and args.num_augment < len(train_df):
        sample_df = train_df.sample(n=args.num_augment, random_state=args.seed)
    else:
        sample_df = train_df
        
    print(f"Génération de {len(sample_df)} exemples adversariaux pour l'augmentation...")
    augmented_texts = []
    
    # Utiliser tqdm pour la barre de progression
    for text in tqdm(sample_df['cleaned_text'], desc="Augmentation"):
        augmented_texts.append(augmenter.augment(text)[0])
        
    augmented_df = pd.DataFrame({
        'cleaned_text': augmented_texts,
        'label': sample_df['label'] # Les labels restent les mêmes
    })
    
    # 2. Combiner les données originales et augmentées
    print(f"Taille originale train: {len(train_df)}")
    robust_train_df = pd.concat([train_df, augmented_df]).drop_duplicates(subset=['cleaned_text'])
    print(f"Taille augmentée train: {len(robust_train_df)}")
    
    X_train_robust, y_train_robust = robust_train_df['cleaned_text'], robust_train_df['label']
    
    # 3. Définir et ré-entraîner le nouveau modèle robuste
    print("Ré-entraînement du pipeline sur les données augmentées...")
    robust_pipeline = build_baseline_pipeline(seed=args.seed)
    robust_pipeline.fit(X_train_robust, y_train_robust)
    
    # 4. Évaluation sur le set de validation (pour vérifier qu'on n'a pas sur-appris)
    print("\n" + "="*30)
    print("Évaluation du Modèle Robuste (sur Val Clean)")
    print("="*30)
    val_df = pd.read_parquet(val_path)
    X_val, y_val = val_df['cleaned_text'], val_df['label']
    y_pred_robust = robust_pipeline.predict(X_val)
    print(classification_report(y_val, y_pred_robust, target_names=['negative', 'positive']))
    
    # 5. Sauvegarde
    print(f"Sauvegarde du modèle robuste dans {robust_model_output_path}...")
    robust_model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(robust_pipeline, robust_model_output_path)
    
    print("Adversarial training terminé.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Effectuer un Adversarial Training (Défense).")
    parser.add_argument("--train_data", type=str, default="data/processed/train.parquet")
    parser.add_argument("--val_data", type=str, default="data/processed/val.parquet")
    parser.add_argument(
        "--baseline_model", 
        type=str, 
        default="artifacts/models/baseline_logreg.joblib"
    )
    parser.add_argument(
        "--output_robust_model", 
        type=str, 
        default="artifacts/models/baseline_logreg_robust.joblib"
    )
    parser.add_argument("--num_augment", type=int, default=5000, help="Nombre d'exemples à générer pour l'augmentation (0 = tout le train set).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)