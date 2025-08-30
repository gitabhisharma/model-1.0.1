from src.wrool_ai.modules.nlp.text_classifier import TextClassifier
from src.wrool_ai.utils.config_loader import ConfigLoader

# Load configuration
config_loader = ConfigLoader()
config = config_loader.load_config()

# Initialize model
model = TextClassifier(config["model"])

# Make predictions
texts = [
    "I love this product! It's amazing!",
    "This is terrible and I want my money back.",
    "The product is okay, nothing special."
]

for text in texts:
    result = model.predict(text)
    print(f"Text: {text}")
    print(f"Prediction: {result['prediction']}, Confidence: {result['confidence']:.4f}")
    print()