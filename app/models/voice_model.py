from io import BytesIO
from flask import current_app
import numpy as np
import librosa
from keras.models import load_model
from pydub import AudioSegment

def preprocess_audio(audio_data):
    samples = np.array(audio_data.get_array_of_samples(), dtype='float32')
    trimmed, _ = librosa.effects.trim(samples, top_db=25)
    padded = np.pad(trimmed, (0, 180000 - len(trimmed)), 'constant')
    return padded

class VoiceModel:
    @staticmethod
    def get_init():
        # Implement your data retrieval logic here
        return {'message': 'voice model resources init completed!'}

    def predict_emotion(audio_file):
        FRAME_LENGTH = current_app.config['VOICE_FRAME_LENGTH']
        HOP_LENGTH = current_app.config['VOICE_HOP_LENGTH']
        MODEL_PATH = current_app.config['VOICE_MODEL_PATH']
        VOICE_CALSSES = current_app.config['VOICE_CALSSES']

        model = load_model(MODEL_PATH)
        audio_data = AudioSegment.from_file(BytesIO(audio_file.read()))
        y = preprocess_audio(audio_data)

        # Dynamically determine the sampling rate (sr)
        sr = audio_data.frame_rate

        zcr = librosa.feature.zero_crossing_rate(y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
        rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=HOP_LENGTH)

        # Ensure that all feature arrays have the same shape
        max_len = min(zcr.shape[1], rms.shape[1], mfccs.shape[1])
        zcr = zcr[:, :max_len]
        rms = rms[:, :max_len]
        mfccs = mfccs[:, :max_len]

        X_test = np.concatenate((zcr, rms, mfccs), axis=0)  # Concatenate along axis 0
        X_test = X_test.T  # Transpose to match the shape
        X_test = X_test.astype('float32')

        y_pred = model.predict(np.expand_dims(X_test, axis=0))
        emotion_labels = VOICE_CALSSES
        predicted_emotion = emotion_labels[np.argmax(y_pred)]
        accuracy = np.amax(y_pred) * 100

        return predicted_emotion, accuracy
