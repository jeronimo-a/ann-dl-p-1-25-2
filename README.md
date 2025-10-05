# Artificial Neural Networks and Deep Learning
**Implementation of a Multi-Layer Perceptron from scratch to identify snoring from audio.**

Jerônimo de Abreu Afrange

## Data Selection

Private dataset of polysomnography exams. To access the dataset contact me.

It contains polysomnography data of a few hundred patients, divided into over a thousand files, totalling over a thousand hours of examination.

The exam data consists of multiple time-series, including room audio and a snore signal, which are the two time-series of interest. The features will be extracted from the audio and the labels from the snore signal.

This dataset was selected because it can be a significant challenge.

## Dataset Explanation

The features and labels must be extracted from the raw data. To do that, a lot of pre-processing is needed. In short, the audio and snore signal are divided into frames of a given size and a given hop from the start of one frame to the start of another. For each audio frame, several audio metrics are calculated, and for each snore signal frame, the RMS intensity. After that, every sequence of frames of a given number of frames is combined into one big frame by averaging the values of each extracted metric. At that point, the target variable will be split into two categories using a threshold. Ultimately, there will be 46 numerical features and one categorical target.

### Potential Issues

For it being an audio signal and for the nature of the problem, imbalance and noise will be the biggest issues with the dataset. Imbalance because people don't snore half the time they're sleeping and noise due to the nature of the data, that is raw audio.

Also, because of how the audio was taken, it is only suitable for snore detection in controlled environments (no mundane noises such as motorcylces passing in the street or dogs barking in the samples).

### Data Sampling

The complete dataset is way too large for this project, because of that, only a small part of the data will be used. Futher details later.

## Pre-processing

The snore signal is a filtered audio signal extracted from secondary audio recorded by a tracheal contact microphone during the examination. The resulting signal is an audio signal with 500Hz sample rate. The intensity of the signal represents the intensity of the snore, for that, the intensity in dB will be computed. The intensity threshold will be based on human hearing thresholds.

### Data Cleaning

Every audio and snore signal in every file is hashed to identify repeated data. Repeated data is then removed from the dataset.

### Feature Extraction

**Parameters:**
- Frame size: 52ms -> 2496 audio samples @ 48kHz sample rate -> 26 snore samples @ 500Hz
- Frame hop: 26ms -> 1248 audio samples @ 48kHz sample rate -> 13 snore samples @ 500Hz
- Frame sequence size: 10 frames
- Frame sequence hop: 1 frame
- Audio intensity FS to SPL: +50dB
- Intensity threshold: 20dB -> -30dBFS - fairly faint sound level

There will be considerable frame and sequence overlap, meaning, information will be repeated, which sounds like a problem but is standard practice for audio processing.

#### Logic

1. **Audio and Snore Signal Extraction**
    - Read all EDF files of a given patient;
    - Create an audio file for each complete audio interval for each patient (EDF files in the middle may be corrupted or missing, ideally, one audio file per patient);
    - Repeat for the snore signal.

2. **Frame Extraction**
    - Go through every input audio calculating the metrics below at every frame:
        - RMS Intensity (intensity of the audio);
        - Zero-Crossing Rate (rate at which the audio signal crosses zero);
        - Spectral Centroid (main frequency of the audio);
        - Spectral Bandwidth (variety of the frequencies of the audio);
        - Spectral Rolloff (frequency below which most of the energy is);
        - Spectral Flatness (high indicates noise-like audio, low indicates tone-like audio);
        - Spectral Flux (measure how much the spectrum is changing between frames);
        - 13 Mel-Frequency Cepstral Coefficients (MFCCs);
        - 13 MFCCs First Derivative;
        - 13 MFCCs Second Derivative.
    - Store the results in a separate parquet;
    - Go through every snore signal calculating the signal intensity in dB at every frame;
    - Store the results in a separate parquet.

3. **Frame Combination**
    - Go through every sequence of $n$ frames (defined in the pre-processing parameters) aggregating the metrics by taking the mean of each feature, both for audio and snore signal;
    - Store the results in a separate parquet.

4. **Category Splitting:** apply the threshold to the intensity of the snore signal;

See deatailed implementation in ```src/process.py```.

## Data Sampling

The data for the four patients with the most balanced snore occurence were used. Two of them will be used for training, one for validating and one for testing.

**Patient IDs:**
- 1593 - 46.6% positive @ -30 dBFS threshold
- 1193 - 42.6% positive @ -30 dBFS threshold
- 1402 - 42.2% positive @ -30 dBFS threshold
- 1480 - 42.2% positive @ -30 dBFS threshold

### Normalization

After the samples are picked, the data of the two training patients is combined and the z-score of each feature calculated. After that, the features of the validating and testing normalized using the training data's mean and standar deviation.

## Model

### Implementation

See implementation in the source code: ```src/mlp.py```

### Hyperparameters and Settings

- **Input Layer:** 46 neurons
- **Hidden Layers:**
    - 69 neurons
    - 34 neurons
    - 10 neurons
- **Output Layer:** 1 neuron
- **Optimizer:** Stochastic Gradient Descent
- **Activation:** sigmoid in every layer to keep activations between 0 and 1
- **Loss:** Binary Cross-Entropy

