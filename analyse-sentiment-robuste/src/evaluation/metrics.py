import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from textattack.models.wrappers import SklearnModelWrapper, HuggingFaceModelWrapper
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import joblib

class TFKerasModelWrapper(HuggingFaceModelWrapper):
    """ 
    Wrapper TextAttack pour les modèles Keras TFAutoModelForSequenceClassification.
    Le wrapper par défaut de HF est pour PyTorch.
    """
    def __call__(self, text_input_list):
        """
        Prend une liste de strings et retourne les logits.
        """
        inputs = self.tokenizer(
            text_input_list, 
            return_tensors="tf", 
            padding=True, 
            truncation=True,
            max_length=self.tokenizer.model_max_length
        )
        # S'assurer que les inputs sont sur le bon device si GPU est utilisé
        # (TensorFlow gère souvent cela automatiquement)
        
        # Le modèle retourne un objet TFSequenceClassifierOutput
        model_output = self.model(inputs)
        logits = model_output.logits
        
        # Retourner en numpy comme attendu par TextAttack
        return logits.numpy()

def compute_tf_metrics(eval_pred):
    """
    Calcule les métriques (accuracy, f1) pour l'évaluation
    pendant l'entraînement du Transformer.
    """
    logits, labels = eval_pred
    predictions = tf.math.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='binary') # Assumer binaire
    return {"accuracy": acc, "f1": f1}

def load_textattack_wrapper(model_type, model_path, tokenizer_path=None):
    """
    Charge le bon wrapper TextAttack pour un modèle sauvegardé.
    """
    if model_type == "sklearn":
        print(f"Chargement du pipeline Sklearn depuis {model_path}...")
        pipeline = joblib.load(model_path)
        # Note : SklearnModelWrapper attend un modèle qui a 'predict_proba'
        return SklearnModelWrapper(pipeline)
        
    elif model_type == "transformer":
        if not tokenizer_path:
            raise ValueError("tokenizer_path est requis pour les modèles transformer")
            
        print(f"Chargement du modèle Transformer depuis {model_path}...")
        model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # S'assurer que max_length est défini
        if tokenizer.model_max_length > 1e10 or tokenizer.model_max_length is None:
             tokenizer.model_max_length = 256 # Définir une valeur par défaut
             
        return TFKerasModelWrapper(model, tokenizer)
        
    else:
        raise ValueError("model_type doit être 'sklearn' ou 'transformer'")