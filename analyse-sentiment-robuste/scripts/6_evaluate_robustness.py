import pandas as pd
import argparse
import joblib
import json
import sys
from pathlib import Path
from tqdm import tqdm

# Ajouter la racine du projet pour importer 'src'
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.evaluation.metrics import load_textattack_wrapper

from textattack.attack_recipes import (
    TextFoolerJin2019, 
    PWWSRen2019,
    DeepWordBugGao2018
)
from textattack.attacker import Attacker
from textattack.attack_args import AttackArgs
from textattack.datasets import Dataset
from textattack.models.wrappers import SklearnModelWrapper
from sklearn.metrics import accuracy_score

def evaluate_model(model_wrapper, dataset, attack_recipe_class, num_examples):
    """
    Évalue un modèle sur données clean et attaquées.
    Retourne un dictionnaire de métriques.
    """
    
    dataset_list = dataset.dataset[:num_examples]
    texts = [d[0] for d in dataset_list]
    labels = [d[1] for d in dataset_list]
    
    # 1. Évaluation Clean
    print("  Évaluation (Clean)...")
    if isinstance(model_wrapper, SklearnModelWrapper):
        preds_clean = model_wrapper.model.predict(texts)
    else: # Transformer
        logits = model_wrapper(texts)
        preds_clean = logits.argmax(axis=1)
        
    acc_clean = accuracy_score(labels, preds_clean)
    
    # 2. Évaluation Adversariale
    print(f"  Évaluation (Adversariale - {attack_recipe_class.__name__})...")
    attack = attack_recipe_class.build(model_wrapper)
    attack_args = AttackArgs(
        num_examples=num_examples,
        disable_stdout=True,
        enable_advance_metrics=True,
        log_to_stdout=False
    )
    
    # Utiliser un sous-ensemble du dataset pour l'attaque
    attacker = Attacker(attack, Dataset(dataset_list), attack_args)
    results = attacker.attack_dataset()
    
    # Extraire les métriques de TextAttack
    # 'Attack Success Rate' de TextAttack est (attaques réussies / total attaques)
    # Nous voulons l'accuracy *après* attaque.
    # original_accuracy = (attaques réussies + échouées) / total
    # accuracy_after_attack = (attaques échouées) / total
    
    num_results = len(results)
    num_failures = results.num_failures
    num_successes = results.num_successes
    
    if num_results == 0:
        return {"Accuracy (Clean)": acc_clean, "Accuracy (Adversarial)": 0, "Attack Success Rate": 0, "Avg. Words Perturbed": 0}

    # Accuracy sur les exemples clean (devrait être proche de acc_clean)
    original_acc = (num_failures + num_successes) / num_results
    
    # Accuracy après attaque (robustesse)
    acc_adv = num_failures / num_results
    
    # Attack Success Rate (parmi ceux initialement corrects)
    if original_acc > 0:
        asr = num_successes / (num_failures + num_successes)
    else:
        asr = 0

    return {
        "Accuracy (Clean)": original_acc, # Acc sur les N exemples testés
        "Accuracy (Adversarial)": acc_adv,
        "Attack Success Rate": asr,
        "Avg. Words Perturbed": results.average_perturbation
    }

def main(args):
    """
    Script d'évaluation final comparant tous les modèles.
    """
    test_data_path = Path(args.test_data)
    num_examples = args.num_examples
    
    print(f"Chargement du set de test (max {num_examples} exemples)...")
    try:
        test_df = pd.read_parquet(test_data_path)
    except FileNotFoundError:
        print(f"Erreur: Fichier de test non trouvé à {test_data_path}")
        return
        
    # Utiliser un échantillon fixe pour la comparabilité
    sample_df = test_df.sample(n=min(num_examples, len(test_df)), random_state=42)
    dataset = Dataset(list(zip(sample_df['cleaned_text'], sample_df['label'])))
    
    # Définir les modèles à tester
    models_to_evaluate = {
        "Baseline (LogReg)": {
            "type": "sklearn",
            "path": "artifacts/models/baseline_logreg.joblib",
            "tokenizer": None
        },
        "Robust (LogReg)": {
            "type": "sklearn",
            "path": "artifacts/models/baseline_logreg_robust.joblib",
            "tokenizer": None
        },
        "Baseline (DistilBERT)": {
            "type": "transformer",
            "path": "artifacts/models/distilbert_baseline",
            "tokenizer": "artifacts/tokenizers/distilbert_baseline"
        }
        # Ajoutez ici un 'DistilBERT Robuste' si vous l'implémentez
    }
    
    # Définir les attaques à tester
    attacks_to_run = {
        "TextFooler": TextFoolerJin2019,
        "PWWS": PWWSRen2019,
        "DeepWordBug": DeepWordBugGao2018
    }
    
    final_results = {}

    for model_name, config in models_to_evaluate.items():
        print(f"\n--- Évaluation du modèle : {model_name} ---")
        try:
            wrapper = load_textattack_wrapper(config["type"], config["path"], config["tokenizer"])
        except (FileNotFoundError, OSError):
            print(f"Avertissement: Modèle {model_name} non trouvé. Saut...")
            continue
            
        final_results[model_name] = {}
        
        for attack_name, recipe in attacks_to_run.items():
            print(f"  > Lancement attaque : {attack_name}...")
            metrics = evaluate_model(wrapper, dataset, recipe, num_examples)
            final_results[model_name][attack_name] = metrics

    # Affichage et sauvegarde des résultats
    print("\n\n" + "="*50)
    print(" RÉSULTATS COMPLETS DE ROBUSTESSE")
    print("="*50)
    
    # Formater pour affichage tableau
    header = f"{'Modèle':<25} | {'Attaque':<12} | {'Acc (Clean)':<12} | {'Acc (Advers.)':<13} | {'ASR (%)':<10}"
    print(header)
    print("-" * len(header))
    
    for model_name, attacks in final_results.items():
        for attack_name, metrics in attacks.items():
            print(f"{model_name:<25} | {attack_name:<12} | {metrics['Accuracy (Clean)']*100:12.2f}% | {metrics['Accuracy (Adversarial)']*100:13.2f}% | {metrics['Attack Success Rate']*100:10.2f}%")

    output_path = Path(args.output_report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=4)
        
    print(f"\nRapport détaillé sauvegardé dans {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation finale de la robustesse des modèles.")
    parser.add_argument("--test_data", type=str, default="data/processed/test.parquet")
    parser.add_argument("--num_examples", type=int, default=100, help="Nombre d'exemples de test à utiliser.")
    parser.add_argument("--output_report", type=str, default="artifacts/reports/robustness_summary.json")
    args = parser.parse_args()
    main(args)