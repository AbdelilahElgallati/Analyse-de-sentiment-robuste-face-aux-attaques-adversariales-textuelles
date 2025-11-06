import pandas as pd
import argparse
import joblib
from pathlib import Path
import sys

# Ajouter la racine du projet pour importer 'src'
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.evaluation.metrics import load_textattack_wrapper

from textattack.attack_recipes import (
    TextFoolerJin2019, 
    PWWSRen2019, 
    DeepWordBugGao2018,
    BERTAttackL2020
)
from textattack.attacker import Attacker
from textattack.attack_args import AttackArgs
from textattack.datasets import Dataset

# Mapper les noms d'attaques aux classes TextAttack
ATTACK_RECIPES = {
    "textfooler": TextFoolerJin2019,
    "pwws": PWWSRen2019,
    "deepwordbug": DeepWordBugGao2018,
    "bertattack": BERTAttackL2020
}

def main(args):
    """
    Lance une attaque adversariale sur un modèle sauvegardé.
    """
    model_path = Path(args.model_path)
    test_data_path = Path(args.test_data)
    
    # 1. Charger le modèle wrappé
    tokenizer_path = args.tokenizer_path if args.model_type == 'transformer' else None
    try:
        model_wrapper = load_textattack_wrapper(args.model_type, model_path, tokenizer_path)
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        return
    
    # 2. Charger le set de données
    print(f"Chargement des données de test depuis {test_data_path}...")
    try:
        test_df = pd.read_parquet(test_data_path)
    except FileNotFoundError:
        print(f"Erreur: Fichier de test non trouvé à {test_data_path}")
        return
        
    # Prendre un échantillon
    sample_df = test_df.sample(n=min(args.num_examples, len(test_df)), random_state=42)
    dataset_list = list(zip(sample_df['cleaned_text'], sample_df['label']))
    dataset = Dataset(dataset_list)
    
    # 3. Définir l'attaque
    if args.attack not in ATTACK_RECIPES:
        print(f"Erreur : Attaque '{args.attack}' non reconnue. Choisir parmi {list(ATTACK_RECIPES.keys())}")
        return
    
    print(f"Construction de la recette d'attaque : {args.attack}...")
    attack_recipe = ATTACK_RECIPES[args.attack].build(model_wrapper)
    
    # 4. Configurer l'attaquant
    log_filename = f"attack_{args.model_type}_{args.attack}.csv"
    log_path = Path("artifacts/reports") / log_filename
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    attack_args = AttackArgs(
        num_examples=args.num_examples,
        log_to_csv=str(log_path),
        checkpoint_interval=max(1, args.num_examples // 10),
        disable_stdout=False,
        enable_advance_metrics=True # Donne plus de détails
    )
    
    attacker = Attacker(attack_recipe, dataset, attack_args)
    
    print(f"\nLancement de l'attaque {args.attack} sur {len(dataset_list)} exemples...")
    results = attacker.attack_dataset()
    
    print("\n" + "="*30)
    print("Résultats de l'attaque")
    print("="*30)
    print(results)
    print(f"\nRapport CSV sauvegardé dans {log_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lancer une attaque adversariale.")
    parser.add_argument("--model_type", type=str, required=True, choices=['sklearn', 'transformer'], help="Type de modèle à attaquer.")
    parser.add_argument("--model_path", type=str, required=True, help="Chemin vers le modèle (.joblib ou dossier HF).")
    parser.add_argument("--tokenizer_path", type=str, help="Chemin vers le tokenizer (requis si model_type='transformer').")
    parser.add_argument("--test_data", type=str, default="data/processed/test.parquet", help="Données de test pour générer des exemples.")
    parser.add_argument("--attack", type=str, default="textfooler", choices=ATTACK_RECIPES.keys(), help="Recette d'attaque TextAttack.")
    parser.add_argument("--num_examples", type=int, default=100, help="Nombre d'exemples à attaquer.")
    
    args = parser.parse_args()
    main(args)