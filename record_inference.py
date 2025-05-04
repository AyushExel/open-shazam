import sounddevice as sd
import numpy as np
import time
import queue
import sys
import io
from scipy.io.wavfile import write as write_wav
import lancedb
from models import get_wav2clip_features, get_clap_features, get_model_clap
import os
import collections
from collections import Counter

import argparse
import webbrowser 
from googlesearch import search

TABLE_NAME = "audio_5s" #"audio"
get_model_clap()
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


def run_live_inference(interval_t=2, max_duration=30, samplerate=16000, save_final_audio=False, 
                      last_n_intervals=3, model="wav2clip", n_last=1, consecutive_threshold=None):
    try:
        db = lancedb.connect("db")
        if TABLE_NAME not in db.table_names():
            print(f"Error: LanceDB table '{TABLE_NAME}' not found. Please run ingestion first.")
            return
        table = db.open_table(TABLE_NAME)
        print(f"Connected to LanceDB table '{TABLE_NAME}'.")
    except Exception as e:
        print(f"Error connecting to LanceDB: {e}")
        return None

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
    global_search_results = []

    previous_top_match = None
    consecutive_matches = 0
    seen_atleast_one_full_length_chunk = False

    final_top_2_matches = (None, None)

    for i, cumulative_audio_np in enumerate(generator):
        if cumulative_audio_np is None:
            print("Generator returned None, stopping inference.")
            break

        final_audio_data = cumulative_audio_np
        current_cumulative_len = len(cumulative_audio_np)
        current_interval_chunk = cumulative_audio_np[previous_cumulative_audio_len:]
        previous_cumulative_audio_len = current_cumulative_len

        if len(current_interval_chunk) > 0:
            recent_interval_chunks.append(current_interval_chunk)

        current_duration = len(cumulative_audio_np) / samplerate
        print(f"\n--- Interval {i+1} (Cumulative Duration: {current_duration:.2f}s) ---")

        # Print current global rankings if we have any matches
        if global_search_results:
            global_counts = Counter(global_search_results)
            current_top = global_counts.most_common(2)  # Get top 2 results
            
            # Early stopping logic
            if consecutive_threshold is not None and seen_atleast_one_full_length_chunk:
                if current_top and current_top[0][0] == previous_top_match:
                    consecutive_matches += 1
                    if consecutive_matches >= consecutive_threshold:
                        final_top_2_matches = (current_top[0][0], current_top[1][0] if len(current_top) > 1 else None)
                        print(f"\nEarly stopping: Match '{final_top_2_matches[0]}' consistent for {consecutive_threshold} intervals")
                        break
                else:
                    consecutive_matches = 1
                    previous_top_match = current_top[0][0] if current_top else None

            if current_top:
                print(f"Current best global match: '{current_top[0][0]}' (Matched {current_top[0][1]} times so far)")


        print("\n--- Sliding Window Search Results (Matching against DB Chunks) ---")
        top_results_names = []

        for k in range(n_last):
            current_window_size = last_n_intervals - k
            if current_window_size < 1:
                continue

            if len(recent_interval_chunks) < current_window_size:
                continue
                print(f"\n  Not enough chunks for window size {current_window_size} (have {len(recent_interval_chunks)}), searching available chunks")
                window_chunks = list(recent_interval_chunks)
            else:
                seen_atleast_one_full_length_chunk = True
                window_chunks = list(recent_interval_chunks)[-current_window_size:]

            window_audio_np = np.concatenate(window_chunks, axis=0)
            window_duration = len(window_audio_np) / samplerate

            try:
                wav_buffer_window = io.BytesIO()
                write_wav(wav_buffer_window, samplerate, window_audio_np.astype(np.float32))
                wav_bytes_window = wav_buffer_window.getvalue()

                features_window = None
                if model == "wav2clip":
                    features_window = get_wav2clip_features(wav_bytes_window)
                elif model == "clap":
                    features_window = get_clap_features(wav_bytes_window)

                if features_window is not None:
                    search_vector_window = features_window.flatten().tolist()
                    results_window = table.search(search_vector_window)\
                                      .select(["name", "chunk_number"])\
                                      .limit(1)\
                                      .to_pandas()

                    if not results_window.empty:
                        top_name = results_window.iloc[0]['name']
                        top_chunk = results_window.iloc[0]['chunk_number']
                        top_distance = results_window.iloc[0]['_distance']
                        #print(f"    Top Match (Window {len(window_chunks)}): Name='{top_name}', Chunk={top_chunk}, Dist={top_distance:.4f}")
                        top_results_names.append(top_name)
                    else:
                        print(f"    No matching chunk found for window {len(window_chunks)}.")
                else:
                    print(f"    Feature extraction failed for window {len(window_chunks)}.")

            except Exception as e:
                print(f"    Error processing window {len(window_chunks)}: {type(e).__name__}: {e}")

        if top_results_names:
            name_counts = Counter(top_results_names)
            final_top_2 = name_counts.most_common(2)

            print("\n--- Final Aggregated Top Results for Interval (Sliding Window vs DB Chunks) ---")
            if len(final_top_2) >= 1:
                global_search_results.append(final_top_2[0][0])
                print(f"  1st Match: '{final_top_2[0][0]}' (Matched {final_top_2[0][1]} times across {len(top_results_names)} window searches)")
            else:
                print("  No consistent match found.")
            if len(final_top_2) >= 2:
                global_search_results.append(final_top_2[1][0])
                print(f"  2nd Match: '{final_top_2[1][0]}' (Matched {final_top_2[1][1]} times across {len(top_results_names)} window searches)")

    print("\nLive inference finished.")

    # New: Print final aggregated results
    if global_search_results:
        global_counts = Counter(global_search_results)
        most_common = global_counts.most_common(2)  # Get top 2 results
        if most_common:
            final_top_2_matches = (most_common[0][0], most_common[1][0] if len(most_common) > 1 else None)
            print("\n--- Final Global Results ---")
            print(f"Most probable match: '{final_top_2_matches[0]}' (Matched {most_common[0][1]} times total)")
            if final_top_2_matches[1]:
                print(f"2nd probable match: '{final_top_2_matches[1]}' (Matched {most_common[1][1]} times total)")
    else:
        print("\nNo matches found during inference.")

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
    
    return final_top_2_matches


def interactive_shazam(enable_guitar_tabs=False):
    """Interactive terminal app that waits for input to start listening"""
    print("Interactive Open Shazam")
    print("Press ENTER to start listening, 'q' to quit")

    while True:
        print("\nPress ENTER to start listening or 'q' to quit...")
        user_input = input()
        
        if user_input.lower() == 'q':
            break
            
        print("\nListening for audio...")
        result = run_live_inference(
            interval_t=1,
            max_duration=20,
            samplerate=44100,
            last_n_intervals=5,
            n_last=3,
            model="clap",
            consecutive_threshold=4,
            save_final_audio=False
        )
        
        if result[0] is not None:
            print(f"\nTop match: {result[0]}")
            if enable_guitar_tabs:
                search_query = f"{result[0]} guitar tabs"
                print(f"Searching for: {search_query}")
                try:
                    # Get first result from Google Search API
                    for url in search(search_query, num_results=1):
                        print(f"Opening: {url}")
                        webbrowser.open(url)
                        break
                    else:
                        print("No search results found")
                except Exception as e:
                    print(f"Search error: {e}")
            
            if result[1] is not None:
                print(f"2nd match: {result[1]}")
                if enable_guitar_tabs:
                    search_query = f"{result[1]} guitar tabs"
                    print(f"Searching for: {search_query}")
                    try:
                        for url in search(search_query, num_results=1):
                            print(f"Opening: {url}")
                            webbrowser.open(url)
                            break
                        else:
                            print("No search results found")
                    except Exception as e:
                        print(f"Search error: {e}")
        else:
            print("\nNo match found")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--guitar-tabs', action='store_true',
                       help='Enable automatic guitar tabs lookup')
    args = parser.parse_args()
    
    interactive_shazam(enable_guitar_tabs=args.guitar_tabs)