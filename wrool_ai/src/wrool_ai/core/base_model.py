import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from loguru import logger

class BaseAIModel(ABC, nn.Module):
    """Abstract base class for all Wrool-AI models"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.device = self._setup_device()
        self._initialize_model()
        
    def _setup_device(self) -> torch.device:
        """Setup device (GPU/CPU) based on availability"""
        if torch.cuda.is_available() and self.config.get("use_gpu", True):
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
        return device
    
    @abstractmethod
    def _initialize_model(self):
        """Initialize model architecture"""
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass implementation"""
        pass
    
    def save(self, path: str):
        """Save model weights and configuration"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model weights and configuration"""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint['config']
        logger.info(f"Model loaded from {path}")
    
    def predict(self, input_data: Any) -> Any:
        """Make prediction on input data"""
        self.eval()
        with torch.no_grad():
            return self._process_input(input_data)
    
    @abstractmethod
    def _process_input(self, input_data: Any) -> Any:
        """Process input data for prediction"""
        pass