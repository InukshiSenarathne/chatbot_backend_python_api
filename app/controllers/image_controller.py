from flask import Blueprint, jsonify, request
from app.models.image_model import ImageModel

bp = Blueprint('image_controller', __name__)

@bp.route('/api/image/resource', methods=['GET'])
def get_resource():
    data = ImageModel.get_init()  # Call a method from your model
    return jsonify(data)
@bp.route('/api/image/predict', methods=['POST'])
def predict_voice():
    try:
        # Check if the 'audio_file' key is in the request files
        if 'image_file' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        image_file = request.files['image_file']

        # Check if the file has a valid extension (e.g., .wav)
        if image_file:
            # Process the audio data
            predicted_emotion, accuracy = ImageModel.predict_image(image_file)

            return jsonify({
                'predicted_emotion': predicted_emotion,
                'accuracy': round(accuracy, 2)
            })

        else:
            return jsonify({'error': 'Invalid image file format.'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500