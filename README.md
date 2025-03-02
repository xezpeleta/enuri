# ENUR̈I

ENUR̈I is a real-time speech transcription **_experimental_** system based on the Whisper. It is designed to transcribe speech in real-time, with low latency. ENUR̈I uses a combination of state-of-the-art speech recognition models to achieve high-quality results in real-time. The system is designed to be used in video recording/streaming applications like OBS, as an overlay for live streams.

It has two main components: a server and a client:

- The server is responsible for transcribing and translating the speech, and should be run on a machine with a GPU.
- The client is responsible for capturing the audio and sending it to the server, as well as showing the transcriptions using a web interface that can be accessed as an overlay in OBS. The client can be run on any machine with a microphone and network access to the server.

**Where does the name come from?**
ENUR̈I is a compound word derived from ENTZUN (Basque for "listen") and IRAKURRI (Basque for "read").

## Quickstart with Docker Compose

Customize the `--command` argument in the `docker-compose.yml` file:
```yaml
...
command: --model=xezpeleta/whisper-tiny-eu-ct2 --language eu --warmup-file samples/basque.wav --web-port 5000
...
```

Start the server:

```bash
docker compose up
```

On the client:

```bash
arecord -f S16_LE -c1 -r 16000 -t raw -D default | nc <ip-address> 43007
```

[On the browser](https://<ip-address>:5000) you will see the transcription in real-time.


## Acknowledgements

This project has been based on the [whisper_streaming](https://github.com/ufal/whisper_streaming) project, which is a real-time speech transcription system based on the Whisper model.