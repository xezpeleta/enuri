#!/usr/bin/env python3
from whisper_online import *
from flask import Flask, render_template_string
from flask_socketio import SocketIO
import sys
import argparse
import os
import logging
import numpy as np
import threading
import socket
import soundfile
import io

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()

# server options
parser.add_argument("--host", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, default=43007)
parser.add_argument("--web-port", type=int, default=5000)
parser.add_argument("--warmup-file", type=str, dest="warmup_file",
        help="The path to a speech audio wav file to warm up Whisper.")

# options from whisper_online
add_shared_args(parser)
args = parser.parse_args()

set_logging(args, logger, other="")

SAMPLING_RATE = 16000

# Initialize Flask and SocketIO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Live Transcription</title>
    <style>
        #transcription {
            width: 80%;
            height: 400px;
            margin: 20px auto;
            padding: 20px;
            border: 1px solid #ccc;
            overflow-y: auto;
            font-family: Arial, sans-serif;
        }
        .transcription-segment {
            margin-bottom: 10px;
            padding: 5px;
            background-color: #f9f9f9;
        }
    </style>
</head>
<body>
    <div id="transcription"></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script>
        const socket = io.connect('http://' + document.domain + ':5001');
        const transcriptionDiv = document.getElementById('transcription');

        socket.on('transcription', function(data) {
            const segment = document.createElement('div');
            segment.className = 'transcription-segment';
            segment.textContent = `[${Math.floor(data.start/1000)}s - ${Math.floor(data.end/1000)}s] ${data.text}`;
            transcriptionDiv.appendChild(segment);
            transcriptionDiv.scrollTop = transcriptionDiv.scrollHeight;
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

class Connection:
    '''it wraps conn object'''
    PACKET_SIZE = 32000*5*60 # 5 minutes

    def __init__(self, conn):
        self.conn = conn
        self.conn.setblocking(True)

    def non_blocking_receive_audio(self):
        try:
            r = self.conn.recv(self.PACKET_SIZE)
            return r
        except ConnectionResetError:
            return None

class ServerProcessor:
    def __init__(self, c, online_asr_proc, min_chunk):
        self.connection = c
        self.online_asr_proc = online_asr_proc
        self.min_chunk = min_chunk
        self.last_end = None
        self.is_first = True

    def receive_audio_chunk(self):
        out = []
        minlimit = self.min_chunk*SAMPLING_RATE
        while sum(len(x) for x in out) < minlimit:
            raw_bytes = self.connection.non_blocking_receive_audio()
            if not raw_bytes:
                break
            sf = soundfile.SoundFile(io.BytesIO(raw_bytes), channels=1, endian="LITTLE", 
                                   samplerate=SAMPLING_RATE, subtype="PCM_16", format="RAW")
            audio, _ = librosa.load(sf, sr=SAMPLING_RATE, dtype=np.float32)
            out.append(audio)
        if not out:
            return None
        conc = np.concatenate(out)
        if self.is_first and len(conc) < minlimit:
            return None
        self.is_first = False
        return np.concatenate(out)

    def format_output_transcript(self, o):
        if o[0] is not None:
            beg, end = o[0]*1000, o[1]*1000
            if self.last_end is not None:
                beg = max(beg, self.last_end)
            self.last_end = end
            return {"start": beg, "end": end, "text": o[2]}
        return None

    def send_result(self, o):
        result = self.format_output_transcript(o)
        if result is not None:
            print(f"{result['start']} {result['end']} {result['text']}", 
                  flush=True, file=sys.stderr)
            socketio.emit('transcription', result)

    def process(self):
        self.online_asr_proc.init()
        while True:
            a = self.receive_audio_chunk()
            if a is None:
                break
            self.online_asr_proc.insert_audio_chunk(a)
            o = online.process_iter()
            try:
                self.send_result(o)
            except Exception as e:
                logger.error(f"Error sending result: {e}")
                break

def run_audio_server():
    # Initialize Whisper
    asr, online = asr_factory(args)
    min_chunk = args.min_chunk_size

    # Warm up Whisper if specified
    if args.warmup_file and os.path.isfile(args.warmup_file):
        a = load_audio_chunk(args.warmup_file, 0, 1)
        asr.transcribe(a)
        logger.info("Whisper is warmed up.")

    # Start audio server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((args.host, args.port))
        s.listen(1)
        logger.info(f'Listening for audio on {args.host}:{args.port}')
        while True:
            conn, addr = s.accept()
            logger.info(f'Connected to client on {addr}')
            connection = Connection(conn)
            proc = ServerProcessor(connection, online, args.min_chunk_size)
            proc.process()
            conn.close()
            logger.info('Connection to client closed')

if __name__ == '__main__':
    # Start the audio server in a separate thread
    audio_thread = threading.Thread(target=run_audio_server)
    audio_thread.daemon = True
    audio_thread.start()

    # Start the web server
    logger.info(f'Starting web server on port {args.web_port}')
    socketio.run(app, host=args.host, port=args.web_port)