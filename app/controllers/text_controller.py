from flask import Blueprint, jsonify, request
from app.models.text_model import TextModel

bp = Blueprint('text_controller', __name__)

@bp.route('/api/text/resource', methods=['GET'])
def get_resource():
    data = TextModel.get_init()  # Call a method from your model
    return jsonify(data)

@bp.route('/api/text/predict', methods=['POST'])
def predict_text():
    try:
        # Get the text input from the request body
        request_data = request.get_json()
        text = request_data.get('text')

        if text is None:
            return jsonify({'error': 'Text input is missing in the request body'}), 400

        # Call a method from your model to make predictions
        predicted_label, confidence_score = TextModel.predict_emotion(text)

        return jsonify({"predicted_label": predicted_label, "confidence_score": confidence_score})
    except Exception as e:
        return jsonify({'error': str(e)}), 500