import pandas as pd
import argparse
import tensorflow as tf
from pathlib import Path
import sys

# Ajouter la racine du projet pour importer 'src'
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.models.transformer_model import (
    load_transformer_components, 
    create_tf_datasets
)
from src.evaluation.metrics import compute_tf_metrics

# Choix de modèles :
# 1. Rapide : 'distilbert-base-uncased' (bon équilibre performance/vitesse)
# 2. Performant : 'bert-base-uncased' (plus lourd, souvent plus précis)

def main(args):
    """
    Fine-tune un modèle Transformer (ex: DistilBERT) pour l'analyse de sentiment.
    """
    train_path = Path(args.train_data)
    val_path = Path(args.val_data)
    model_output_dir = Path(args.output_dir)
    tokenizer_output_dir = Path(args.tokenizer_dir)
    
    print("Chargement des données train/val...")
    try:
        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path)
    except FileNotFoundError as e:
        print(f"Erreur: Fichier de données non trouvé. {e}")
        print("Veuillez exécuter les scripts 1_clean_data.py et 2_split_data.py d'abord.")
        return

    # 1. Charger les composants du modèle
    tokenizer, model = load_transformer_components(args.model_name)
    
    # 2. Créer les tf.data.Dataset
    tf_train_dataset, tf_val_dataset = create_tf_datasets(
        train_df, 
        val_df, 
        tokenizer, 
        args.max_length, 
        args.batch_size
    )

    # 3. Compiler le modèle
    print("Compilation du modèle...")
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    # 4. Définir les Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=2, 
        restore_best_weights=True
    )
    
    print(f"\nDébut de l'entraînement ({args.epochs} époques)...")
    history = model.fit(
        tf_train_dataset,
        validation_data=tf_val_dataset,
        epochs=args.epochs,
        callbacks=[early_stopping]
    )
    
    print("Entraînement terminé.")
    
    print(f"Sauvegarde du modèle fine-tuné dans {model_output_dir}")
    model_output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_output_dir)
    
    print(f"Sauvegarde du tokenizer dans {tokenizer_output_dir}")
    tokenizer_output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_output_dir)
    
    print("Fine-tuning du Transformer terminé.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuner un modèle Transformer.")
    parser.add_argument("--train_data", type=str, default="data/processed/train.parquet")
    parser.add_argument("--val_data", type=str, default="data/processed/val.parquet")
    parser.add_argument("--output_dir", type=str, default="artifacts/models/distilbert_baseline")
    parser.add_argument("--tokenizer_dir", type=str, default="artifacts/tokenizers/distilbert_baseline")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="Modèle HF (rapide: distilbert-base-uncased, performant: bert-base-uncased)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=256)
    
    args = parser.parse_args()
    main(args)