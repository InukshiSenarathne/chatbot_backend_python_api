from flask import Blueprint, jsonify, request
from app.models.voice_model import VoiceModel
from pydub import AudioSegment
import librosa

bp = Blueprint('voice_controller', __name__)

@bp.route('/api/voice/resource', methods=['GET'])
def get_resource():
    data = VoiceModel.get_init()  # Call a method from your model
    return jsonify(data)

@bp.route('/api/voice/predict', methods=['POST'])
def predict_voice():
    try:
        # Check if the 'audio_file' key is in the request files
        if 'audio_file' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio_file']

        # Check if the file has a valid extension (e.g., .wav)
        if audio_file and audio_file.filename.endswith('.wav'):
            # Process the audio data
            predicted_emotion, accuracy = VoiceModel.predict_emotion(audio_file)

            return jsonify({
                'predicted_emotion': predicted_emotion,
                'accuracy': round(accuracy, 2)
            })

        else:
            return jsonify({'error': 'Invalid audio file format. Supported format is .wav'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500