"""Script to predict deepsalience output from audio"""
from __future__ import print_function

import librosa
import numpy as np
import os

from keras.models import Model
from keras.layers import Input, Lambda
from keras.layers.merge import Concatenate, Multiply
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.models import load_model

TASKS = ['multif0', 'melody', 'bass', 'vocal', 'piano', 'guitar']
TASK_INDICES = {
    'multif0': 0, 'melody': 1, 'bass': 2, 'vocal': 3,
    'piano': 4, 'guitar': 5
}
BINS_PER_OCTAVE = 60
N_OCTAVES = 6
HARMONICS = [1, 2, 3, 4, 5]
SR = 22050
FMIN = 32.7
HOP_LENGTH = 256


def compute_hcqt(audio_fpath):
    """Compute the harmonic CQT from a given audio file

    Parameters
    ----------
    audio_fpath : str
        path to audio file

    Returns
    -------
    hcqt : np.ndarray
        Harmonic cqt
    time_grid : np.ndarray
        List of time stamps in seconds
    freq_grid : np.ndarray
        List of frequency values in Hz

    """
    y, fs = librosa.load(audio_fpath, sr=SR)

    cqt_list = []
    shapes = []
    for h in HARMONICS:
        cqt = librosa.cqt(
            y, sr=fs, hop_length=HOP_LENGTH, fmin=FMIN*float(h),
            n_bins=BINS_PER_OCTAVE*N_OCTAVES,
            bins_per_octave=BINS_PER_OCTAVE
        )
        cqt_list.append(cqt)
        shapes.append(cqt.shape)

    shapes_equal = [s == shapes[0] for s in shapes]
    if not all(shapes_equal):
        min_time = np.min([s[1] for s in shapes])
        new_cqt_list = []
        for i in range(len(cqt_list)):
            new_cqt_list.append(cqt_list[i][:, :min_time])
        cqt_list = new_cqt_list

    log_hcqt = ((1.0/80.0) * librosa.core.amplitude_to_db(
        np.abs(np.array(cqt_list)), ref=np.max)) + 1.0

    freq_grid = librosa.cqt_frequencies(
        BINS_PER_OCTAVE*N_OCTAVES, FMIN, bins_per_octave=BINS_PER_OCTAVE
    )

    time_grid = librosa.core.frames_to_time(
        range(log_hcqt.shape[2]), sr=SR, hop_length=HOP_LENGTH
    )

    return log_hcqt, freq_grid, time_grid


def bkld(y_true, y_pred):
    """KL Divergence where both y_true an y_pred are probabilities
    """
    y_true = K.clip(y_true, K.epsilon(), 1.0 - K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    return K.mean(K.mean(
        -1.0*y_true* K.log(y_pred) - (1.0 - y_true) * K.log(1.0 - y_pred),
        axis=-1), axis=-1)


def get_model():
    """ Get model structure.

    Returns
    -------
    model : keras model
        Precompiled multitask model

    """
    input_shape = (None, None, 5)
    y0 = Input(shape=input_shape)

    y1_pitch = Conv2D(
        32, (5, 5), padding='same', activation='relu', name='pitch_layer1')(y0)
    y1a_pitch = BatchNormalization()(y1_pitch)
    y2_pitch = Conv2D(
        32, (5, 5), padding='same', activation='relu', name='pitch_layer2')(y1a_pitch)
    y2a_pitch = BatchNormalization()(y2_pitch)
    y3_pitch = Conv2D(32, (3, 3), padding='same', activation='relu', name='smoothy2')(y2a_pitch)
    y3a_pitch = BatchNormalization()(y3_pitch)
    y4_pitch = Conv2D(8, (70, 3), padding='same', activation='relu', name='distribute')(y3a_pitch)
    y4a_pitch = BatchNormalization()(y4_pitch)

    y_multif0 = Conv2D(
        1, (1, 1), padding='same', activation='sigmoid', name='multif0_presqueeze')(y4a_pitch)
    multif0 = Lambda(lambda x: K.squeeze(x, axis=3), name='multif0')(y_multif0)

    y_mask = Multiply(name='mask')([y_multif0, y0])
    y1_timbre = Conv2D(
        512, (2, 3), padding='same', activation='relu', name='timbre_layer1')(y_mask)
    y1a_timbre = BatchNormalization()(y1_timbre)

    y_concat = Concatenate(name='timbre_and_pitch')([y_multif0, y1a_timbre])
    ya_concat = BatchNormalization()(y_concat)

    y_mel_feat = Conv2D(
        32, (3, 3), padding='same', activation='relu', name='melody_filters')(ya_concat) #32
    ya_mel_feat = BatchNormalization()(y_mel_feat)
    y_mel_feat2 = Conv2D(
        32, (3, 3), padding='same', activation='relu', name='melody_filters2')(ya_mel_feat)#32
    ya_mel_feat2 = BatchNormalization()(y_mel_feat2)
    y_mel_feat3 = Conv2D(
        8, (240, 1), padding='same', activation='relu', name='melody_filters3')(ya_mel_feat2) # 8
    ya_mel_feat3 = BatchNormalization()(y_mel_feat3)
    y_mel_feat4 = Conv2D(
        16, (7, 7), padding='same', activation='relu', name='melody_filters4')(ya_mel_feat3) # 16
    ya_mel_feat4 = BatchNormalization()(y_mel_feat4)
    y_mel_feat5 = Conv2D(
        16, (7, 7), padding='same', activation='relu', name='melody_filters5')(ya_mel_feat4) #16
    ya_mel_feat5 = BatchNormalization()(y_mel_feat5)

    y_bass_feat = Conv2D(
        32, (3, 3), padding='same', activation='relu', name='bass_filters')(ya_concat) #32
    ya_bass_feat = BatchNormalization()(y_bass_feat)
    y_bass_feat2 = Conv2D(
        32, (3, 3), padding='same', activation='relu', name='bass_filters2')(ya_bass_feat) #32
    ya_bass_feat2 = BatchNormalization()(y_bass_feat2)
    y_bass_feat3 = Conv2D(
        8, (240, 1), padding='same', activation='relu', name='bass_filters3')(ya_bass_feat2) #8
    ya_bass_feat3 = BatchNormalization()(y_bass_feat3)
    y_bass_feat4 = Conv2D(
        16, (7, 7), padding='same', activation='relu', name='bass_filters4')(ya_bass_feat3) #16
    ya_bass_feat4 = BatchNormalization()(y_bass_feat4)
    y_bass_feat5 = Conv2D(
        16, (7, 7), padding='same', activation='relu', name='bass_filters5')(ya_bass_feat4) #16
    ya_bass_feat5 = BatchNormalization()(y_bass_feat5)

    y_vocal_feat = Conv2D(
        32, (3, 3), padding='same', activation='relu', name='vocal_filters')(ya_concat) #32
    ya_vocal_feat = BatchNormalization()(y_vocal_feat)
    y_vocal_feat2 = Conv2D(
        32, (3, 3), padding='same', activation='relu', name='vocal_filters2')(ya_vocal_feat) #32
    ya_vocal_feat2 = BatchNormalization()(y_vocal_feat2)
    y_vocal_feat3 = Conv2D(
        8, (240, 1), padding='same', activation='relu', name='vocal_filters3')(ya_vocal_feat2) #8
    ya_vocal_feat3 = BatchNormalization()(y_vocal_feat3)
    y_vocal_feat4 = Conv2D(
        16, (7, 7), padding='same', activation='relu', name='vocal_filters4')(ya_vocal_feat3) # 16
    ya_vocal_feat4 = BatchNormalization()(y_vocal_feat4)
    y_vocal_feat5 = Conv2D(
        16, (7, 7), padding='same', activation='relu', name='vocal_filters5')(ya_vocal_feat4) #16
    ya_vocal_feat5 = BatchNormalization()(y_vocal_feat5)

    y_piano_feat = Conv2D(
        32, (3, 3), padding='same', activation='relu', name='piano_filters')(ya_concat) #32
    ya_piano_feat = BatchNormalization()(y_piano_feat)
    y_piano_feat2 = Conv2D(
        32, (3, 3), padding='same', activation='relu', name='piano_filters2')(ya_piano_feat) #32
    ya_piano_feat2 = BatchNormalization()(y_piano_feat2)
    y_piano_feat3 = Conv2D(
        8, (240, 1), padding='same', activation='relu', name='piano_filters3')(ya_piano_feat2) #8
    ya_piano_feat3 = BatchNormalization()(y_piano_feat3)
    y_piano_feat4 = Conv2D(
        16, (7, 7), padding='same', activation='relu', name='piano_filters4')(ya_piano_feat3) # 16
    ya_piano_feat4 = BatchNormalization()(y_piano_feat4)
    y_piano_feat5 = Conv2D(
        16, (7, 7), padding='same', activation='relu', name='piano_filters5')(ya_piano_feat4) #16
    ya_piano_feat5 = BatchNormalization()(y_piano_feat5)

    y_guitar_feat = Conv2D(
        32, (3, 3), padding='same', activation='relu', name='guitar_filters')(ya_concat) #32
    ya_guitar_feat = BatchNormalization()(y_guitar_feat)
    y_guitar_feat2 = Conv2D(
        32, (3, 3), padding='same', activation='relu', name='guitar_filters2')(ya_guitar_feat) #32
    ya_guitar_feat2 = BatchNormalization()(y_guitar_feat2)
    y_guitar_feat3 = Conv2D(
        8, (240, 1), padding='same', activation='relu', name='guitar_filters3')(ya_guitar_feat2) #8
    ya_guitar_feat3 = BatchNormalization()(y_guitar_feat3)
    y_guitar_feat4 = Conv2D(
        16, (7, 7), padding='same', activation='relu', name='guitar_filters4')(ya_guitar_feat3) # 16
    ya_guitar_feat4 = BatchNormalization()(y_guitar_feat4)
    y_guitar_feat5 = Conv2D(
        16, (7, 7), padding='same', activation='relu', name='guitar_filters5')(ya_guitar_feat4) #16
    ya_guitar_feat5 = BatchNormalization()(y_guitar_feat5)

    y_melody = Conv2D(
        1, (1, 1), padding='same', activation='sigmoid', name='melody_presqueeze')(ya_mel_feat5)
    melody = Lambda(lambda x: K.squeeze(x, axis=3), name='melody')(y_melody)

    y_bass = Conv2D(
        1, (1, 1), padding='same', activation='sigmoid', name='bass_presqueeze')(ya_bass_feat5)
    bass = Lambda(lambda x: K.squeeze(x, axis=3), name='bass')(y_bass)

    y_vocal = Conv2D(
        1, (1, 1), padding='same', activation='sigmoid', name='vocal_presqueeze')(ya_vocal_feat5)
    vocal = Lambda(lambda x: K.squeeze(x, axis=3), name='vocal')(y_vocal)

    y_piano = Conv2D(
        1, (1, 1), padding='same', activation='sigmoid', name='piano_presqueeze')(ya_piano_feat5)
    piano = Lambda(lambda x: K.squeeze(x, axis=3), name='piano')(y_piano)

    y_guitar = Conv2D(
        1, (1, 1), padding='same', activation='sigmoid', name='guitar_presqueeze')(ya_guitar_feat5)
    guitar = Lambda(lambda x: K.squeeze(x, axis=3), name='guitar')(y_guitar)

    model = Model(inputs=y0, outputs=[multif0, melody, bass, vocal, piano, guitar])

    model.compile(
        loss=bkld, metrics=['mse'], optimizer='adam'
    )

    return model


def load_model():
    """Load the precompiled, pretrained model

    Returns
    -------
    model : Model
        Pretrained, precompiled Keras model
    """
    model = get_model()
    weights_path = os.path.join('weights', 'model_weights.h5')
    if not os.path.exists(weights_path):
        raise IOError(
            "Cannot find weights path {}".format(weights_path))

    model.load_weights(weights_path)
    return model


def get_single_test_prediction(model, input_hcqt, max_frames=None):
    """Generate output from a model given an input numpy file

    Parameters
    ----------
    model : Model
        Pretrained model
    input_hcqt : np.ndarray
        HCQT
    max_frames : int or None
        Maximum number of frames to compute over input
        If None, computes over all frames

    Returns
    -------
    predicted_output : np.ndarray
        list of numpy arrays of predictions
    """
    x = input_hcqt.transpose(1, 2, 0)[np.newaxis, :, :, :]

    if max_frames is not None:
        x = x[:, :, :max_frames, :]

    n_t = x.shape[2]
    n_slices = 1000
    t_slices = list(np.arange(0, n_t, n_slices))
    model_output = model.output
    if isinstance(model_output, list):
        output_list = [[] for i in range(len(model_output))]
        predicted_output = [[] for i in range(len(model_output))]
    else:
        output_list = [[]]
        predicted_output = [[]]

    for j, t in enumerate(t_slices):
        print("{} / {}".format(j, len(t_slices)))
        x_slice = x[:, :, t:t+n_slices, :]

        prediction = model.predict(x_slice)

        if isinstance(prediction, list):
            for i, pred in enumerate(prediction):
                output_list[i].append(pred[0, :, :])
        else:
            output_list[0].append(prediction[0, :, :])

    for i in range(len(output_list)):
        predicted_output[i] = np.hstack(output_list[i])

    return predicted_output


def compute_output(hcqt, time_grid, freq_grid, max_frames=None):
    """Comput output for a given task

    Parameters
    ----------
    hcqt : np.ndarray
        harmonic cqt
    time_grid : np.ndarray
        array of times
    freq_grid : np.ndarray
        array of frequencies
    max_frames : int or None
        Maximum number of frames to compute over input
        If None, computes over all frames

    Returns
    -------
    salience_dict : dict
        Dictionary with task labels as keys and
        salience numpy arrays as values.
    """
    model = load_model()

    print("Computing salience...")
    output_array = get_single_test_prediction(
        model, hcqt, max_frames=max_frames)

    output_dict = {}
    for task, i in TASK_INDICES.items():
        output_dict[task] = output_array[i]

    return output_dict


def run(audio_fpath):
    """ Compute f0 salience from an audio file

    Parameters
    ----------
    audio_fpath : str
        Path to input audio file

    Returns
    -------
    salience_dict : dict
        Dictionary with task labels as keys and
        salience numpy arrays as values.
    """
    print("Computing HCQT...")
    hcqt, freq_grid, time_grid = compute_hcqt(audio_fpath)

    print("Predicting output...")
    salience_dict = compute_output(hcqt, time_grid, freq_grid)

    return salience_dict
