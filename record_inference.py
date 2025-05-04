import sounddevice as sd
import numpy as np
import time
import queue
import sys
import io
from scipy.io.wavfile import write as write_wav
import lancedb
from models import get_wav2clip_features, get_clap_features
import os
import collections


def record_audio_generator(interval_t=1, max_duration=30, samplerate=16000, channels=1, dtype='float32'):
    q = queue.Queue()
    cumulative_audio_chunks = []
    recording_start_time = None
    stream = None

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"Stream status: {status}", file=sys.stderr, flush=True)
        q.put(indata.copy())

    try:
        actual_samplerate = samplerate
        print(f"Attempting to start recording: SR={actual_samplerate}, Channels={channels}, Interval={interval_t}s, Max Duration={max_duration}s, Dtype={dtype}", flush=True)

        stream = sd.InputStream(
            samplerate=actual_samplerate,
            channels=channels,
            callback=audio_callback,
            dtype=dtype
        )
        with stream:
            recording_start_time = time.time()
            last_yield_time = recording_start_time
            print("Recording started...", flush=True)

            while True:
                current_time = time.time()
                elapsed_time = current_time - recording_start_time

                if elapsed_time >= max_duration:
                    print(f"\nMaximum recording duration ({max_duration}s) reached.", flush=True)
                    break

                processed_new_data = False
                while not q.empty():
                    chunk = q.get_nowait()
                    cumulative_audio_chunks.append(chunk)
                    processed_new_data = True

                if current_time - last_yield_time >= interval_t:
                    if cumulative_audio_chunks:
                        full_audio_np = np.concatenate(cumulative_audio_chunks, axis=0)
                        print(f"Yielding audio at {elapsed_time:.2f}s (Shape: {full_audio_np.shape})", flush=True)
                        yield full_audio_np
                    last_yield_time = current_time

                sleep_duration = min(0.05, interval_t / 10)
                time.sleep(sleep_duration)

        print("Stream closed.", flush=True)

        print("Processing any remaining audio chunks...", flush=True)
        while not q.empty():
            try:
                chunk = q.get_nowait()
                cumulative_audio_chunks.append(chunk)
            except queue.Empty:
                break

        if cumulative_audio_chunks:
            full_audio_np = np.concatenate(cumulative_audio_chunks, axis=0)
            final_duration = len(full_audio_np) / actual_samplerate
            print(f"Final yield of complete audio (Duration: {final_duration:.2f}s, Shape: {full_audio_np.shape})", flush=True)
            yield full_audio_np
        else:
            print("No audio was recorded.", flush=True)
            yield None

    except sd.PortAudioError as pae:
         print(f"\nPortAudioError: {pae}", file=sys.stderr, flush=True)
         print("This might indicate an issue with the selected audio device or sample rate.", file=sys.stderr, flush=True)
         print("Please check your microphone connection and system audio settings.", file=sys.stderr, flush=True)
         yield None
    except Exception as e:
        print(f"\nAn error occurred during recording: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        yield None
    finally:
        if stream and not stream.closed:
             print("Force closing stream in finally block.", flush=True)
             stream.abort(ignore_errors=True)
             stream.close(ignore_errors=True)
        print("Recording function finished.", flush=True)


def run_live_inference(interval_t=2, max_duration=30, samplerate=16000, save_final_audio=False, last_n_intervals=3, model="wav2clip"):
    try:
        db = lancedb.connect("db")
        table_name = "audio"
        if table_name not in db.table_names():
            print(f"Error: LanceDB table '{table_name}' not found. Please run ingestion first.")
            return
        table = db.open_table(table_name)
        print(f"Connected to LanceDB table '{table_name}'.")
    except Exception as e:
        print(f"Error connecting to LanceDB: {e}")
        return

    generator = record_audio_generator(
        interval_t=interval_t,
        max_duration=max_duration,
        samplerate=samplerate,
        channels=1,
        dtype='float32'
    )

    print("\nStarting live inference loop...")
    final_audio_data = None
    previous_cumulative_audio_len = 0 
    recent_interval_chunks = collections.deque(maxlen=last_n_intervals)

    for i, cumulative_audio_np in enumerate(generator):
        if cumulative_audio_np is None:
            print("Generator returned None, stopping inference.")
            break

        final_audio_data = cumulative_audio_np

        current_cumulative_len = len(cumulative_audio_np)
        # Extract the audio added in this specific interval
        current_interval_chunk = cumulative_audio_np[previous_cumulative_audio_len:]
        previous_cumulative_audio_len = current_cumulative_len 

        if len(current_interval_chunk) > 0:
            recent_interval_chunks.append(current_interval_chunk)

        if recent_interval_chunks:
            recent_n_audio_np = np.concatenate(list(recent_interval_chunks), axis=0)
        else:
            recent_n_audio_np = None 

        current_duration = len(cumulative_audio_np) / samplerate
        print(f"\n--- Interval {i+1} (Cumulative Duration: {current_duration:.2f}s) ---")

        if recent_n_audio_np is not None and len(recent_n_audio_np) > 0:
            recent_duration = len(recent_n_audio_np) / samplerate
            print(f"\nProcessing Last {len(recent_interval_chunks)} Intervals (Duration: {recent_duration:.2f}s)...")
            try:
                wav_buffer_recent = io.BytesIO()
                write_wav(wav_buffer_recent, samplerate, recent_n_audio_np)
                wav_bytes_recent = wav_buffer_recent.getvalue()
                print("  Converted recent audio chunk to WAV bytes.")

                print("  Extracting recent features...")
                if model == "wav2clip":
                    features_recent = get_wav2clip_features(wav_bytes_recent)
                elif model == "clap":
                    features_recent = get_clap_features(wav_bytes_recent)

                if features_recent is not None:
                    search_vector_recent = features_recent.flatten().tolist()
                    results_recent = table.search(search_vector_recent).limit(5).to_pandas()
                    print("  Search Results (Recent):")
                    if not results_recent.empty:
                        print(results_recent[['name', '_distance']])

                else:
                    print("  Feature extraction failed for recent chunk.")

            except Exception as e:
                print(f"  Error processing recent audio chunk: {e}")
        else:
             print("\nNot enough data yet for recent interval search.")


    print("\nLive inference finished.")

    if save_final_audio and final_audio_data is not None:
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_filename = f"recorded_audio_{timestamp}.wav"
            output_path = os.path.join(os.getcwd(), output_filename)
            write_wav(output_path, samplerate, final_audio_data)
            print(f"Saved complete recording to: {output_path}")
        except Exception as e:
            print(f"Error saving final audio: {e}")
    elif save_final_audio:
        print("No audio data was captured, skipping save.")


if __name__ == "__main__":
    run_live_inference(
        interval_t=1, # Check every 1 second
        max_duration=20,
        samplerate=44100, 
        last_n_intervals=7,
        n_last=4,
        model="wav2clip",
    )