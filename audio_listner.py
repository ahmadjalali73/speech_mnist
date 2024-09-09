import streamlit as st
from audio_recorder_streamlit import audio_recorder
import tensorflow as tf
import numpy as np
import pandas as pd
from pydub import AudioSegment
from pydub.silence import detect_nonsilent, split_on_silence
import matplotlib.pyplot as plt
from os.path import join, dirname, abspath, isdir
from os import unlink, listdir, makedirs

class AudioPredictor:
    def __init__(self, audio_path=""):
        self.current_dir = dirname(abspath(__file__))
        self.chunk_path = join(self.current_dir, "chuncks")
        self.model_path = join(self.current_dir, "speech_mnist_model.h5")
        self.speech_predictor_model = tf.keras.models.load_model(self.model_path)
        self.audio_path = audio_path

    def set_audio_path(self, audio_path):
        self.audio_path = audio_path

    def word_prechecks(self):
        if isdir(self.chunk_path):
            for chunk in listdir(self.chunk_path):
                unlink(join(self.chunk_path, chunk))
        else:
            makedirs(self.chunk_path)
         
    # def decode_audio(self, audio):
    # # audio = tf.io.read_file(audio_file_path)
    #     audio, rate = tf.audio.decode_wav(contents=audio)
    #     rate = tf.cast(rate, tf.float32)
    #     # print(rate)
    #     # return tf.squeeze(audio, axis=-1)
    #     return audio, rate

    def decode_audio(self, audio_file_path):
        audio = tf.io.read_file(audio_file_path)
        audio, rate = tf.audio.decode_wav(contents=audio)
        # print(rate)
        return tf.squeeze(audio, axis=-1)

    def get_spectrogram(self, waveform):
        spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
        spectrogram = tf.abs(spectrogram)
        spectrogram = tf.expand_dims(spectrogram, -1)
        return spectrogram

    def add_noise(self, waveform, noise_factor=0.005):
        noise = tf.random.normal(shape=tf.shape(waveform), mean=0.0, stddev=1.0)
        return waveform + noise_factor * noise

    def time_stretch(self, waveform, rate=1.25):
        return tf.image.resize(waveform, (int(waveform.shape[0] / rate),waveform.shape[1]))

    def normalize_spectrogram(self, spectrogram):
        mean = tf.math.reduce_mean(spectrogram)
        std = tf.math.reduce_std(spectrogram)
        return (spectrogram - mean) / std
    
    def pad_audio(self, tensor):
        target_size = 298
        padding = target_size - tf.shape(tensor)[0]
        if padding > 0:
            tensor = tf.pad(tensor, [[0, padding], [0, 0], [0, 0]])
        return tensor

    def _model_predict(self, audio_path):
        padded_tensors = []
        audio = self.decode_audio(audio_path)
        audio = self.get_spectrogram(audio)
        audio = self.add_noise(audio)
        audio = self.time_stretch(audio)
        audio = self.normalize_spectrogram(audio)
        padded_tensors.append(self.pad_audio(audio))
        padded_tensors = np.array(padded_tensors)
        result = self.speech_predictor_model.predict(padded_tensors)
        return np.argmax(result)

    def word_splitter(self):
        self.word_prechecks()
        sound_file = AudioSegment.from_wav(self.audio_path)
        audio_chunks = split_on_silence(sound_file, min_silence_len=200, silence_thresh=-32)
        print(f"{len(audio_chunks)} Words has been detected!")
        for i, chunk in enumerate(audio_chunks):
            out_file = join(self.chunk_path, "chunk_{}.wav".format(i))
            # print("exporting", out_file)
            channels = chunk.split_to_mono()
            channels[0].export(out_file, format="wav")
    
    def predict(self):
        result_number = []
        self.word_splitter()
        for audio in listdir(self.chunk_path):
            if audio.endswith(".wav"):
                audio_path = join(self.chunk_path, audio)
                predicted_number = self._model_predict(audio_path)
                result_number.append(str(predicted_number))
        return " - ".join(result_number)


def main():
    current_dir = dirname(abspath(__file__))
    audio_path = join(current_dir, "audio_placeholder")
    if not isdir(audio_path):
        makedirs(audio_path)
    audio_file_path = join(audio_path, "saved_file.wav")
    audio_predictor = AudioPredictor()

    st.title("Audio Recorder")
    audio_bytes = audio_recorder()

    if audio_bytes:
        st.audio(audio_bytes, format='audio/wav')
        with open(audio_file_path, 'wb') as f:
            f.write(audio_bytes)
        audio_predictor.set_audio_path(audio_file_path)
        result = audio_predictor.predict()
        st.header(result)

if __name__ == "__main__":
    main()
