#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import shlex
import subprocess
import sys
import wave
from scipy.io import wavfile
import scipy.signal as sps

from deepspeech import Model, version

try:
    from shhlex import quote
except ImportError:
    from pipes import quote

d_model = "./asr_models/deepspeech-0.9.3-models.pbmm"
d_scorer = "./asr_models/deepspeech-0.9.3-models.scorer"

def convert_samplerate(audio_path, desired_sample_rate):
    sox_cmd = 'sox {} --type raw --bits 16 --channels 1 --rate {} --encoding signed-integer --endian little --compression 0.0 --no-dither - '.format(quote(audio_path), desired_sample_rate)
    try:
        output = subprocess.check_output(shlex.split(sox_cmd), stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError('SoX returned non-zero status: {}'.format(e.stderr))
    except OSError as e:
        raise OSError(e.errno, 'SoX not found, use {}hz files or install it: {}'.format(desired_sample_rate, e.strerror))

    return desired_sample_rate, np.frombuffer(output, np.int16)

def speech_to_text(audio_filepath: str) -> str:
    """
    Runs the deepspeech framework on the audio stored in the above file_path.
    :param audio_filepath: Path to the audio file
    :return: The transcript for the given audio file
    """
    # Loading model from file
    ds = Model(d_model)

    desired_sample_rate = ds.sampleRate()

    if d_scorer:
        # Loading scorer from files
        ds.enableExternalScorer(d_scorer)

    fin = wave.open(audio_filepath, 'rb')
    fs_orig = fin.getframerate()
    if fs_orig != desired_sample_rate:
        # print('Warning: original sample rate ({}) is different than {}hz. Resampling might produce erratic speech recognition.'.format(fs_orig, desired_sample_rate), file=sys.stderr)
        fs_new, audio = convert_samplerate(audio_filepath, desired_sample_rate)
    else:
        audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)
    
    audio_length = fin.getnframes() * (1/fs_orig)
    fin.close()

    # Running inference
    asr_output = ds.stt(audio)
    print(asr_output)
    return asr_output

# speech_to_text('audio_files/261ff450-e435-4410-a21c-ce1eaf5004f6.wav')