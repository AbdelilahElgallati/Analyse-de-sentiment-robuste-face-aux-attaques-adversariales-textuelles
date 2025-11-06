import tensorflow as tf
from transformers import (
    AutoTokenizer, 
    TFAutoModelForSequenceClassification, 
    DataCollatorWithPadding
)
from datasets import Dataset

def load_transformer_components(model_name):
    """
    Charge le tokenizer et le modèle de base pour la classification de séquence.
    """
    print(f"Chargement du tokenizer et du modèle de base pour {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, from_pt=False)    
    return tokenizer, model

def create_tf_datasets(train_df, val_df, tokenizer, max_length, batch_size):
    """
    Convertit les DataFrames pandas en tf.data.Dataset prêts pour l'entraînement.
    """
    print("Conversion des DataFrames en Datasets Hugging Face...")
    train_dataset = Dataset.from_pandas(train_df.rename(columns={'cleaned_text': 'text'}))
    val_dataset = Dataset.from_pandas(val_df.rename(columns={'cleaned_text': 'text'}))
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=max_length)
        
    print("Tokenisation des données...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)
    
    # Data collator pour padding dynamique
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
    
    print("Préparation des tf.data.Dataset...")
    
    def remove_unused_columns(dataset, cols_to_keep):
        """Helper pour ne garder que ce dont TF a besoin."""
        all_cols = dataset.column_names
        cols_to_remove = [c for c in all_cols if c not in cols_to_keep]
        return dataset.remove_columns(cols_to_remove)

    cols_needed = ['input_ids', 'attention_mask', 'label']
    
    # Définir expliciement les colonnes de features
    feature_cols = ['input_ids', 'attention_mask']
    
    tf_train_dataset = remove_unused_columns(tokenized_train, cols_needed).to_tf_dataset(
        columns=feature_cols, 
        label_cols="label",
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    
    tf_val_dataset = remove_unused_columns(tokenized_val, cols_needed).to_tf_dataset(
        columns=feature_cols, 
        label_cols="label",
        shuffle=False,
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    
    return tf_train_dataset, tf_val_dataset