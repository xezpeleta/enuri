#!/usr/bin/env python3
from whisper_online import *
from flask import Flask, render_template_string, request
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
import re  # Add import for regular expressions

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
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True)

@socketio.on('connect')
def handle_connect():
    logger.info(f'Client connected from {request.remote_addr}')

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')

@socketio.on_error()
def error_handler(e):
    logger.error(f'SocketIO error: {str(e)}')

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Live Transcription</title>
    <style>
        body {
            margin: 0;
            padding: 20px;
            background-color: transparent;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            justify-content: flex-end;
            min-height: 100vh;
            box-sizing: border-box;
        }
        #transcription {
            font-family: Arial, sans-serif;
            font-size: 24px;
            color: white;
            text-shadow: 2px 2px 2px rgba(0,0,0,0.8);
        }
        .transcription-segment {
            background-color: rgba(0,0,0,0.5);
            padding: 10px 20px;
            border-radius: 5px;
            margin: 5px 0;
            text-align: center;
        }
        .previous-line {
            opacity: 0.8;
        }
        .current-line {
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div id="transcription">
        <div id="previous-segment" class="transcription-segment previous-line"></div>
        <div id="current-segment" class="transcription-segment current-line"></div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script>
        const socket = io({
            reconnection: true,
            reconnectionDelay: 1000,
            reconnectionDelayMax: 5000,
            reconnectionAttempts: Infinity,
            transports: ['websocket']
        });

        const previousSegment = document.getElementById('previous-segment');
        const currentSegment = document.getElementById('current-segment');

        socket.on('connect', () => {
            console.log('Connected to server');
            currentSegment.textContent = 'Connected to server...';
        });

        socket.on('connect_error', (error) => {
            console.error('Connection error:', error);
            currentSegment.textContent = 'Connection error, retrying...';
        });

        socket.on('disconnect', () => {
            console.log('Disconnected from server');
            currentSegment.textContent = 'Disconnected from server...';
        });

        socket.on('transcription', function(data) {
            console.log('Received transcription:', data);
            const text = data.text.trim();
            if (text) {
                previousSegment.textContent = currentSegment.textContent;
                currentSegment.textContent = text;
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, web_port=args.web_port)

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
        if o[2]:  # if there's text
            # Just send the text, without timestamps
            text = o[2].replace('[', '').replace(']', '')  # Remove any remaining brackets
            # Remove any timestamp patterns at the start of the text
            text = re.sub(r'^\d+s\s*-\s*\d+s\s*', '', text)
            return {"text": text.strip()}
        return None

    def send_result(self, o):
        result = self.format_output_transcript(o)
        if result is not None:
            socketio.emit('transcription', result)

    def process(self):
        self.online_asr_proc.init()
        while True:
            a = self.receive_audio_chunk()
            if a is None:
                break
            self.online_asr_proc.insert_audio_chunk(a)
            o = self.online_asr_proc.process_iter()
            try:
                if o and o[2]:
                    # Remove anything inside brackets including the brackets
                    text = re.sub(r'\[.*?\]', '', o[2])
                    # Clean up any double spaces that might result
                    text = ' '.join(text.split())
                    if text:
                        socketio.emit('transcription', {"text": text.strip()})
            except Exception as e:
                logger.error(f"Error sending result: {e}")
                break

def run_audio_server():
    # Initialize Whisper with timestamps disabled
    args.show_timestamps = False  # Force timestamps off for web interface
    asr, online = asr_factory(args)
    min_chunk = args.min_chunk_size

    # Warm up Whisper if specified
    if args.warmup_file and os.path.isfile(args.warmup_file):
        a = load_audio_chunk(args.warmup_file, 0, 1)
        asr.transcribe(a)
        logger.info("Whisper is warmed up.")

    # Start audio server
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                # Add socket reuse option
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((args.host, args.port))
                s.listen(1)
                logger.info(f'Listening for audio on {args.host}:{args.port}')
                
                while True:
                    try:
                        conn, addr = s.accept()
                        logger.info(f'Connected to client on {addr}')
                        connection = Connection(conn)
                        proc = ServerProcessor(connection, online, args.min_chunk_size)
                        proc.process()
                    except Exception as e:
                        logger.error(f'Error processing connection: {e}')
                    finally:
                        conn.close()
                        logger.info('Connection to client closed')
        except Exception as e:
            logger.error(f'Server error: {e}')
            import time
            time.sleep(1)  # Wait before attempting to restart

if __name__ == '__main__':
    # Start the audio server in a separate thread
    audio_thread = threading.Thread(target=run_audio_server)
    audio_thread.daemon = True
    audio_thread.start()

    # Start the web server with improved settings
    logger.info(f'Starting web server on port {args.web_port}')
    socketio.run(app, 
                host=args.host, 
                port=args.web_port, 
                debug=True,
                allow_unsafe_werkzeug=True,  # Required for debug mode
                use_reloader=False)  # Disable reloader to prevent duplicate threads