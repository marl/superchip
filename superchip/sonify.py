from __future__ import print_function

import numpy as np
import mir_eval
import scipy
import librosa


def remove_salience_noise(salience):
    """Remove noise from the salience representation by removing non-peaks

    Parameters
    ----------
    salience : np.ndarray
        Salience representation

    Returns
    -------
    clean_salience : np.ndarray
        Cleaned salience representation

    """
    clean_salience = np.zeros(salience.shape)
    peaks = scipy.signal.argrelmax(salience, axis=0)
    clean_salience[peaks] = salience[peaks]
    return clean_salience


def get_single_f0(salience, time_grid, freq_grid):
    """Get single-f0 curve from a salience representation by taking
    the argmax
    
    Parameters
    ----------
    salience : np.ndarray
        Salience representation
    time_grid : np.array
        
    """
    max_idx = np.argmax(salience, axis=0)
    est_freqs = []
    amps = []
    for i, f in enumerate(max_idx):
        est_freqs.append(freq_grid[f])
        amps.append(salience[f, i])
    est_freqs = np.array(est_freqs)
    return time_grid, est_freqs, amps


def triangle(*args, **kwargs):
    '''Synthesize a triangle wave'''
    v = scipy.signal.sawtooth(*args, **kwargs)

    return 2 * np.abs(v) - 1.


def sonify(time_grid, freq_grid, salience_dictionary, use_contours=True):

    bass = salience_dictionary['bass']
    bass = bass / np.max(np.max(bass))

    print("melody...")
    if use_contours:
        _, mel_f0, mel_f0_amp = get_single_f0(
            salience_dictionary['melody'], time_grid, freq_grid)
        y_mel = mir_eval.sonify.pitch_contour(
            time_grid, mel_f0, 8000, amplitudes=mel_f0_amp,
            function=scipy.signal.square)
    else:
        melody = remove_salience_noise(salience_dictionary['melody'])
        y_mel = mir_eval.sonify.time_frequency(
            melody[:, :], freq_grid[:], time_grid, 8000,
            function=scipy.signal.square)

    print("bass...")
    if use_contours:
        _, bass_f0, bass_f0_amp = get_single_f0(bass, time_grid, freq_grid)
        y_bass = mir_eval.sonify.pitch_contour(
            time_grid, bass_f0, 8000, amplitudes=bass_f0_amp, function=triangle)
    else:
        y_bass = mir_eval.sonify.time_frequency(
            bass, freq_grid, time_grid, 8000, function=triangle)

    y_chip = np.zeros((np.max([len(y_mel), len(y_bass)]), ))
    y_chip[:len(y_mel)] += y_mel
    y_chip[:len(y_bass)] += y_bass

    return y_mel, y_bass, y_chip


def save_audio(signal, sr, save_path):
    librosa.output.write_wav(save_path, signal, sr, norm=True)

