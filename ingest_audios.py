import lancedb
from lancedb.pydantic import Vector, LanceModel
import os
import numpy as np
import io
from scipy.io.wavfile import read as read_wav
from scipy.io.wavfile import write as write_wav
import math

from models import get_wav2clip_features, convert_to_wav, get_clap_features

class Schema(LanceModel):
    vector: Vector(512)
    name: str
    chunk_number: int

db = lancedb.connect("db")
TABLE_NAME = "audio_5s" #"audio"
if TABLE_NAME in db.table_names():
    table = db.open_table(TABLE_NAME)
else:
    table = db.create_table(TABLE_NAME, schema=Schema)


def ingest(audio_folder="audios", model="wav2clip", chunk_duration_s=10):
    data_to_add = []

    for filename in os.listdir(audio_folder):
        file_path = os.path.join(audio_folder, filename)
        if not os.path.isfile(file_path):
            continue

        print(f"\nProcessing file: {file_path}")
        wav_bytes = convert_to_wav(file_path)
        if wav_bytes is None:
            print(f"Skipping file due to conversion error: {filename}")
            continue

        samplerate, audio_data = read_wav(io.BytesIO(wav_bytes))
        print(f"Samplerate: {samplerate}, audio file {filename}")
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        samples_per_chunk = int(chunk_duration_s * samplerate)
        num_chunks = math.ceil(len(audio_data) / samples_per_chunk)
        file_basename = os.path.splitext(filename)[0]

        print(f"Splitting into {num_chunks} chunks of {chunk_duration_s}s...")

        for i in range(num_chunks):
            start_sample = i * samples_per_chunk
            end_sample = start_sample + samples_per_chunk
            chunk_data = audio_data[start_sample:end_sample]

            if len(chunk_data) == 0:
                continue

            chunk_wav_buffer = io.BytesIO()
            write_wav(chunk_wav_buffer, samplerate, chunk_data.astype(np.int16))
            chunk_wav_bytes = chunk_wav_buffer.getvalue()

            print(f"  Generating features for chunk {i+1}/{num_chunks} of {filename}")
            vector = None
            if model == "wav2clip":
                vector = get_wav2clip_features(chunk_wav_bytes)
            elif model == "clap":
                vector = get_clap_features(chunk_wav_bytes)

            if vector is None:
                print(f"  Skipping chunk {i+1} due to feature extraction error.")
                continue

            record = {
                "vector": vector.flatten().tolist(),
                "name": file_basename,
                "chunk_number": i
            }
            data_to_add.append(record)



    if data_to_add:
        print(f"\nAdding {len(data_to_add)} chunk records to LanceDB table 'audio'...")
        table.add(data_to_add)
        print("Ingestion complete.")
    else:
        print("No data to add.")


if __name__ == "__main__":
    ingest(model="clap", chunk_duration_s=5)