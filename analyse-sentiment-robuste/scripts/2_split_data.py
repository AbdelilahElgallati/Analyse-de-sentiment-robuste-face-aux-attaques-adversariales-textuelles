import pandas as pd
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split

def main(args):
    """
    Script pour splitter les données nettoyées en train, validation et test.
    """
    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    
    if not input_path.exists():
        print(f"Erreur : Fichier d'entrée non trouvé à {input_path}")
        return
        
    print(f"Chargement des données nettoyées depuis {input_path}...")
    df = pd.read_parquet(input_path)
    
    if 'label' not in df.columns:
        print("Erreur: Colonne 'label' non trouvée dans le fichier Parquet.")
        return
        
    print("Séparation des données (train, val, test)...")
    
    # S'assurer que les proportions sont valides
    if args.val_size + args.test_size >= 1.0:
        print("Erreur: La somme de val_size et test_size doit être < 1.0")
        return
        
    # Premier split : (1 - val_size - test_size) % train, (val_size + test_size) % temp (val + test)
    train_df, temp_df = train_test_split(
        df, 
        test_size=(args.val_size + args.test_size), 
        random_state=args.seed, 
        stratify=df['label']
    )
    
    # Deuxième split : val_size, test_size (calculé sur les % restants)
    # Ex: 0.15 / (0.15 + 0.15) = 0.5
    val_test_ratio = args.test_size / (args.val_size + args.test_size)
    
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=val_test_ratio, 
        random_state=args.seed, 
        stratify=temp_df['label']
    )
    
    print(f"Taille Train: {len(train_df)}")
    print(f"Taille Val:   {len(val_df)}")
    print(f"Taille Test:  {len(test_df)}")
    
    # Sauvegarde
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(output_dir / "train.parquet", index=False)
    val_df.to_parquet(output_dir / "val.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)
    
    print(f"Données sauvegardées dans {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Splitter les données en train/val/test.")
    parser.add_argument(
        "--input_file", 
        type=str, 
        default="data/processed/cleaned_data.parquet", 
        help="Chemin vers le fichier Parquet nettoyé."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/processed", 
        help="Dossier pour sauvegarder les splits."
    )
    parser.add_argument("--val_size", type=float, default=0.15, help="Proportion du set de validation (ex: 0.15).")
    parser.add_argument("--test_size", type=float, default=0.15, help="Proportion du set de test (ex: 0.15).")
    parser.add_argument("--seed", type=int, default=42, help="Seed pour la reproductibilité.")
    
    args = parser.parse_args()
    main(args)