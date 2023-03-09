# This is a sample Python script.
from time import sleep

from tkinter import *
import openai

with open("openai_api_key", 'r') as f:
    api_key = f.read()[:-1] # remove new line
openai.api_key = api_key
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# def voice_rec():


# Import the necessary modules.
import tkinter
import tkinter as tk
import tkinter.messagebox
import pyaudio
import wave
from pydub import AudioSegment
import os
import torch
from pydub.playback import play


class RecAUD:

    def __init__(self, chunk=3024, frmat=pyaudio.paInt16, channels=2, rate=44100, py=pyaudio.PyAudio()):

        # Start Tkinter and set Title
        self.main = tkinter.Tk()
        self.collections = []
        self.main.geometry('300x200')
        self.main.title('Recorder')
        self.CHUNK = chunk
        self.FORMAT = frmat
        self.CHANNELS = channels
        self.RATE = rate
        self.p = py
        self.frames = []
        self.st = 1
        self.stream = self.p.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, input=True,
                                  frames_per_buffer=self.CHUNK)

        # Set Frames
        self.buttons = tkinter.Frame(self.main, padx=120, pady=20)

        # Pack Frame
        self.buttons.pack(fill=tk.BOTH)

        # Start and Stop buttons
        self.strt_rec = tkinter.Button(self.buttons, width=10, padx=10, pady=5, text='Start Recording',
                                       command=lambda: self.start_record())
        self.strt_rec.grid(row=0, column=0, padx=50, pady=5)
        self.stop_rec = tkinter.Button(self.buttons, width=10, padx=10, pady=5, text='Stop Recording',
                                       command=lambda: self.stop())
        self.stop_rec.grid(row=1, column=0, columnspan=1, padx=50, pady=5)

        self.messages = [
            {"role": "system", "content": "You are an helpful asistant, who wont mention that its created by openai."},
            {"role": "user", "content": "Who are you?"}]
        # r = openai.ChatCompletion.create(
        #       model="gpt-3.5-turbo",
        #       messages=self.messages
        #     )
        # for c in r['choices']:
        #     self.messages.append(c['message'])
        message = {'role': 'assistant',
                   "content": 'I am an Artificial Intelligence, here to assist you with any questions or tasks you may have. How can I help you today?'}

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

        self.say(message)
        self.messages.append(message)
        tkinter.mainloop()

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
        # sound = AudioSegment(
        #     # raw audio data (bytes)
        #     data=b''.join(self.frames),
        #     # 2 byte (16 bit) samples
        #     sample_width=self.CHANNELS,  # 2,
        #     # 44.1 kHz frame rate
        #     frame_rate=rate,  # 44100,
        #     # stereo
        #     channels=2
        # )
        # play(sound)

    def start_record(self):
        self.st = 1
        self.frames = []
        stream = self.p.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, input=True,
                             frames_per_buffer=self.CHUNK)
        c = 0
        while self.st == 1:
            data = stream.read(self.CHUNK)
            self.frames.append(data)
            if c % 10 == 1:
                print(c, "* recording")
            self.main.update()
            c += 1

        # stream.close()
        #
        # wf = wave.open('test_recording.wav', 'wb')
        # wf.setnchannels(self.CHANNELS)
        # wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
        # wf.setframerate(self.RATE)
        # wf.writeframes(b''.join(self.frames))
        # Advanced usage, if you have raw audio data:
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

        sound.export("test_recording.mp3", format="mp3")
        # wf.close()
        self.transcribe_audio()

        self.ask_chatgpt()

        self.say(self.messages[-1])

    def transcribe_audio(self, filename="test_recording.mp3"):
        audio_file = open(filename, "rb")
        print("sending transcription request")
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        # transcript = {"text":"How are you today?"}
        print("transcribed:", transcript['text'])

        self.messages.append({"role": "user", "content": transcript['text']})

    def ask_chatgpt(self):
        print("sending request to gpt-3")
        r = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.messages
        )
        print("answer received")
        for c in r['choices']:
            self.messages.append(c['message'])

    def stop(self):
        self.st = 0


# Create an object of the ProgramGUI class to begin the program.
guiAUD = RecAUD()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
