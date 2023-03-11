
import sys

import torch
from PyQt6.QtCore import pyqtSlot, pyqtSignal, QRunnable, QObject, QThreadPool
from PyQt6.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget
import openai
from pydub.playback import play
from pydub import AudioSegment
import pyaudio

with open("openai_api_key", 'r') as f:
    api_key = f.read()[:-1]  # remove new line
openai.api_key = api_key


class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc() )

    result
        object data returned from processing, anything

    progress
        int indicating % progress

    '''
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)


class RecordingWorker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, format, channels, rate, chunk, verbose=False):
        self.verbose = verbose
        if self.verbose:
            print("initializing recorder!")
        super(RecordingWorker, self).__init__()
        self.FORMAT = format
        self.CHANNELS = channels
        self.RATE = rate
        self.CHUNK = chunk
        self.st = 0
        self.p = pyaudio.PyAudio()
        self.frames = []

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''
        if self.verbose:
            print("starting to record!")
        self.st = 1
        # Retrieve args/kwargs here; and fire processing using them
        self.frames = []
        stream = self.p.open(format=self.FORMAT, channels=self.CHANNELS,
                             rate=self.RATE, input=True,
                             frames_per_buffer=self.CHUNK)
        c = 0
        while self.st == 1:
            data = stream.read(self.CHUNK)
            self.frames.append(data)
            if self.verbose:
                if c % 10 == 1:
                    print(c, "* recording")
            c += 1
        sound = AudioSegment(
            # raw audio data (bytes)
            data=b''.join(self.frames),
            # 2 byte (16 bit) samples
            sample_width=self.CHANNELS,  # 2,
            # 44.1 kHz frame rate
            frame_rate=self.RATE,  # 44100,
            # stereo
            channels=2
        )
        if self.verbose:
            print("finished recording!")
        # TODO(nik): remove saving to disk, this should stay in memory
        sound.export("test_recording.mp3", format="mp3")


class Window(QWidget):
    def __init__(self, parent=None, verbose=False):
        super().__init__(parent)
        self.verbose = verbose
        self.st = 0
        self.CHUNK = 3024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 2
        self.RATE = 44100
        self.threadpool = QThreadPool()
        self.setup()

        # init Gui
        self.button = QPushButton("Record !")
        self.button.setFixedSize(120, 60)
        self.button.clicked.connect(self.record)
        layout = QVBoxLayout()
        layout.addWidget(self.button)
        self.setLayout(layout)

    def setup(self):
        # setup voice syntheszer

        print("loading tacotron2")
        self.tacotron2 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math='fp16')
        self.tacotron2 = self.tacotron2.to('cuda')
        self.tacotron2.eval()

        print("loading waveglow")
        self.waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp16')
        self.waveglow = self.waveglow.remove_weightnorm(self.waveglow)
        self.waveglow = self.waveglow.to('cuda')
        self.waveglow.eval()

        self.tacotron2_utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')

        self.messages = [
            {"role": "system", "content": "You are an helpful asistant, who wont mention that its created by openai."},
            {"role": "user", "content": "Who are you?"}]

        message = {'role': 'assistant',
                   "content": 'I am an Artificial Intelligence, here to assist you with any questions or tasks you may have. How can I help you today?'}
        self.say(message)
        self.messages.append(message)

    def say(self, message):
        text_long = message['content']
        role = message['role']
        text_split = text_long.split('.')
        for text in text_split:
            print(role, ":", text)
            text += '.'

            sequences, lengths = self.tacotron2_utils.prepare_input_sequence([text])
            with torch.no_grad():
                mel, _, _ = self.tacotron2.infer(sequences, lengths)
                audio = self.waveglow.infer(mel)
            audio_numpy = audio[0].data.cpu().numpy()
            rate = 22050
            audio_segment = AudioSegment(
                audio_numpy.tobytes(),
                frame_rate=rate,
                sample_width=audio_numpy.dtype.itemsize,
                channels=1
            )
            play(audio_segment)

    def transcribe_audio(self, filename="test_recording.mp3"):
        audio_file = open(filename, "rb")
        if self.verbose:
            print("sending transcription request")
        transcript = openai.Audio.transcribe("whisper-1", audio_file)

        print("transcribed:", transcript['text'])

        self.messages.append({"role": "user", "content": transcript['text']})

    def ask_chatgpt(self):
        if self.verbose:
            print("sending request to gpt-3")
        r = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.messages
        )
        if self.verbose:
            print("answer received")
        for c in r['choices']:
            self.messages.append(c['message'])

    def stop_recording(self):
        # stop recording
        self.recorder.st = 0

        self.button.disconnect()
        self.button.clicked.connect(self.record)
        self.button.setText("Record !")
        # force update gui
        QApplication.processEvents()

        # Execute
        all_finished_correctly = self.threadpool.waitForDone()
        if self.verbose:
            print("all threads exited gracefully:", all_finished_correctly)

        self.transcribe_audio()

        self.ask_chatgpt()

        self.say(self.messages[-1])

    def record(self):
        self.button.disconnect()
        self.button.clicked.connect(self.stop_recording)
        self.button.setText("Stop recording !")
        # force update gui
        QApplication.processEvents()
        # start recording thread
        self.recorder = RecordingWorker(format=self.FORMAT, channels=self.CHANNELS,
                                        rate=self.RATE, chunk=self.CHUNK)
        # Execute
        self.threadpool.start(self.recorder)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec())
