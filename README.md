### Guitar Shazam

1. download music_audioset_epoch_15_esc_90.14.pt checkpoint in cwd from laion
2. Add bunch of music files in audio folder.
3. run ingest_audios.py file
4. run record_inference.py file to run inference from your mic.
5. run `record_inference.py --guitar-tabs` to run inference from your mic and directly open guitar tabs for the that's playing or you want to play.


### Demo on isolated Guitar recorded via mic
https://github.com/user-attachments/assets/dcb1788e-3fec-4579-917e-fceeb846b345


### Demo on music from various sources
https://github.com/user-attachments/assets/3b4867c6-08fb-4e96-bb64-f1d970d0b983


### How this works:

👉 Used existing smaller audio embedding models (like wav2clip or clap). Clap worked better for me
⚠️ The main limitation of these models is that they only work best for smaller audio files of few seconds. 



**Augment using VectorDB**

👉 I used LanceDB to store smaller chunks (experimented with many but settled to 5sec chunks) with their original song id.
⚠️ But simply retrieving the most common chunk id across all chunks didn't do the trick. In fact, the results were pretty bad.


**Inference Phase - Sliding window avg-ing**

👉 Just avg across chunk searches didn't show great results due to many factors (sampling rate, quality loss due to mic, stereo vs mono channels) 

👉 So, each chunk was retrieved from t to t-n (n=3) timestamps and the most common top1 was chosen as the best result for that window to smooth the noise.


**Track matches across multiple segments**

👉 Global best match is maintained for each chunk. If the global top1 doesn't change for early_stopping (=4) timestamps, the execution is stopped and either the result is displayed or its guitar tabs page is opened if tabs mode is enabled.

