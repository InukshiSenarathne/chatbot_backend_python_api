from flask import current_app
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
class TextModel:
    @staticmethod
    def get_init():
        # Implement your data retrieval logic here
        return {'message': 'text model resources init completed!'}

    def predict_emotion(text):
        # Get the model path from the configuration
        model_path = current_app.config['TEXT_MODEL_PATH']
        class_names = current_app.config['TEXT_CALSSES']

        # Load the tokenizer associated with the model
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        # Load the model
        loaded_model = AutoModelForSequenceClassification.from_pretrained(model_path)

        # Make predictions
        with torch.no_grad():
            outputs = loaded_model(**inputs)
            logits = outputs.logits

        # Get the predicted class (assuming it's a classification model)
        predicted_class = torch.argmax(logits, dim=1).item()

        # Get the emotion label name
        predicted_label = class_names[predicted_class]

        # Get the confidence score (softmax probability) and round it to two decimal points
        confidence_score = round(torch.softmax(logits, dim=1)[0][predicted_class].item() * 100, 2)

        return predicted_label, confidence_score
