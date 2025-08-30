import pytest
import torch
from src.wrool_ai.modules.nlp.text_classifier import TextClassifier

def test_model_initialization():
    """Test that model initializes correctly"""
    config = {
        "model_name": "bert-base-uncased",
        "num_classes": 2,
        "use_gpu": False
    }
    
    model = TextClassifier(config)
    assert model is not None
    assert hasattr(model, 'transformer')
    assert hasattr(model, 'classifier')

def test_model_prediction():
    """Test model prediction functionality"""
    config = {
        "model_name": "bert-base-uncased",
        "num_classes": 2,
        "use_gpu": False
    }
    
    model = TextClassifier(config)
    result = model.predict("This is a test sentence.")
    
    assert "prediction" in result
    assert "confidence" in result
    assert "probabilities" in result
    assert isinstance(result["prediction"], int)
    assert isinstance(result["confidence"], float)