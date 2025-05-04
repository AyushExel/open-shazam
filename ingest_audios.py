import lancedb
from lancedb.pydantic import Vector, LanceModel
import os


from models import get_wav2clip_features, convert_to_wav, get_clap_features

class Schema(LanceModel):
    vector: Vector(512)
    audio: bytes
    name: str

db = lancedb.connect("db")
if "audio" in db.table_names():
    table = db.open_table("audio")
else:
    table = db.create_table("audio", schema=Schema)


def ingest(audio_folder="audios", model="wav2clip"):
    data_to_add = []

    for filename in os.listdir(audio_folder):
        file_path = os.path.join(audio_folder, filename)
        if os.path.isfile(file_path):
            print(f"\nProcessing file: {file_path}")
            wav_data = convert_to_wav(file_path)
            if wav_data is None:
                print(f"Skipping file due to conversion error: {filename}")
                continue

            print(f"Generating features for: {filename}")
            if model == "wav2clip":
                vector = get_wav2clip_features(wav_data)
            elif model == "clap":
                vector = get_clap_features(wav_data)
            
            if vector is None:
                print(f"Skipping file due to feature extraction error: {filename}")
                continue

            file_basename = os.path.splitext(filename)[0]
            record = {
                "vector": vector.flatten().tolist(),
                "audio": wav_data,
                "name": file_basename
            }
            data_to_add.append(record)

    if data_to_add:
        print(f"\nAdding {len(data_to_add)} records to LanceDB table 'audio'...")
        table.add(data_to_add)
        print("Ingestion complete.")



if __name__ == "__main__":
    ingest(model="wav2clip")
    #up_and_up = get_wav2clip_features("test.mp4")
    #print(table.search(up_and_up.flatten().tolist()).to_pandas())
    