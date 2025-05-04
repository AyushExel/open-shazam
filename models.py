import wav2clip
import librosa
import io
import numpy as np
import os
import uuid
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
import io
from typing import Union
import laion_clap

WAV2CLIP = None
CLAP = None
TEMP_WAV_FOLDER = "temp_wav_files"

def get_model_wav2clip():
    global WAV2CLIP
    if WAV2CLIP is not None:
        return WAV2CLIP
    model = wav2clip.get_model()
    WAV2CLIP = model
    return model

def get_model_clap():
    global CLAP
    if CLAP is not None:
        return CLAP
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel= 'HTSAT-base')
    model.load_ckpt("music_audioset_epoch_15_esc_90.14.pt")
    CLAP = model
    return model


def convert_to_wav(input_path: str) -> bytes | None:
    try:
        if not os.path.exists(input_path):
            print(f"Error: Input file not found at {input_path}")
            return None

        audio = AudioSegment.from_file(input_path)

        buffer = io.BytesIO()
        audio.export(buffer, format="wav")
        buffer.seek(0)
        wav_data = buffer.read()
        return wav_data

    except CouldntDecodeError:
        print(f"Error: Could not decode audio file: {input_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def get_wav2clip_features(file: Union[str|bytes]):
    model = get_model_wav2clip()
    temp_file_path = None

    try:
        if not os.path.exists(TEMP_WAV_FOLDER):
            os.makedirs(TEMP_WAV_FOLDER)
            print(f"Created temporary directory: {TEMP_WAV_FOLDER}")

        temp_filename = f"{uuid.uuid4()}.wav"
        temp_file_path = os.path.join(TEMP_WAV_FOLDER, temp_filename)
        if isinstance(file, str):
            with open(temp_file_path, 'wb') as f:
                f.write(convert_to_wav(file))
        else:
            with open(temp_file_path, 'wb') as f:
                f.write(file)
        print(f"Saved temporary WAV file to: {temp_file_path}")
        audio, _ = librosa.load(temp_file_path, sr=44100)
        embeddings = wav2clip.embed_audio(audio, model)

        return embeddings
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                print(f"Removed temporary file: {temp_file_path}")
            except OSError as oe:
                print(f"Error removing temporary file {temp_file_path}: {oe}")
    
def get_clap_features(file: Union[str|bytes]):
    model = get_model_clap()
    temp_file_path = None
    try:
        if not os.path.exists(TEMP_WAV_FOLDER):
            os.makedirs(TEMP_WAV_FOLDER)
            print(f"Created temporary directory: {TEMP_WAV_FOLDER}")
        temp_filename = f"{uuid.uuid4()}.wav"
        temp_file_path = os.path.join(TEMP_WAV_FOLDER, temp_filename)
        if isinstance(file, str):
            with open(temp_file_path, 'wb') as f:
                f.write(convert_to_wav(file))
        else:
            with open(temp_file_path, 'wb') as f:
                f.write(file)
        print(f"Saved temporary WAV file to: {temp_file_path}")
        audio, _ = librosa.load(temp_file_path, sr=44100)
        audio = audio.reshape(1, -1) # Make it (1,T) or (N,T)
        embedding = model.get_audio_embedding_from_data(x = audio, use_tensor=False)
        return embedding
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None