import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Any
from ...core.base_model import BaseAIModel

class TextClassifier(BaseAIModel):
    """Text classification module for Wrool-AI"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_classes = config.get("num_classes", 2)
        self._initialize_tokenizer()
        
    def _initialize_model(self):
        """Initialize transformer model for classification"""
        model_name = self.config.get("model_name", "bert-base-uncased")
        self.transformer = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(
            self.transformer.config.hidden_size, 
            self.num_classes
        )
        self.dropout = nn.Dropout(self.config.get("dropout", 0.1))
        
    def _initialize_tokenizer(self):
        """Initialize tokenizer"""
        model_name = self.config.get("model_name", "bert-base-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass for text classification"""
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)
    
    def _process_input(self, text: str) -> Dict[str, Any]:
        """Process text input for prediction"""
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.config.get("max_length", 512),
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self(**inputs)
        probabilities = torch.softmax(outputs, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1)
        
        return {
            "prediction": prediction.item(),
            "confidence": probabilities.max().item(),
            "probabilities": probabilities.tolist()
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict on a batch of texts"""
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results