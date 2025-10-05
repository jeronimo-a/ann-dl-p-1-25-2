import os
import torch
import numpy as np
import pandas as pd
import librosa as lr
import soundfile as sf

from tqdm import tqdm
from toml import load as toml_load
from threading import Lock
from concurrent.futures import ThreadPoolExecutor

MAX_THREADS_CIF = 1
MAX_THREADS_COF = 8
MAX_THREADS_CIS = 2
MAX_THREADS_COS = 12

FILEPATH = os.path.abspath(__file__)
PBAR_LOCK = Lock()

class Process:

    files: list[str]
    settings: dict
    raw_input_dir: str
    raw_output_dir: str
    input_frames_dir: str
    output_frames_dir: str

    def __init__(self, raw_input_dir: str, raw_output_dir: str, input_frames_dir: str, output_frames_dir: str):
        
        # checks if the input and output directories exist
        if not os.path.isdir(raw_input_dir): raise NotADirectoryError(f"Raw input data directory '{raw_input_dir}' is not a directory.")
        if not os.path.isdir(raw_output_dir): raise NotADirectoryError(f"Raw output data directory '{raw_output_dir}' is not a directory.")

        # assigns input and output directories
        self.raw_input_dir = raw_input_dir
        self.raw_output_dir = raw_output_dir
        self.input_frames_dir = input_frames_dir
        self.output_frames_dir = output_frames_dir

        try:
            # loads settings
            self.settings = toml_load(os.path.join(os.path.dirname(FILEPATH), '..', 'settings.toml'))['process']

            # calculates derived settings
            self.settings["frame_size_ms"] = int(1000 * self.settings['audio_frame_size_len'] / self.settings['audio_sample_rate'])
            self.settings["frame_hop_ms"] = int(1000 * self.settings['audio_frame_hop_len'] / self.settings['audio_sample_rate'])
            snore_frame_size_len = self.settings['snore_sample_rate'] * self.settings['frame_size_ms'] / 1000
            self.settings["snore_frame_size_len"] = int(snore_frame_size_len)
            snore_frame_hop_len = self.settings['snore_sample_rate'] * self.settings['frame_hop_ms'] / 1000
            self.settings["snore_frame_hop_len"] = int(snore_frame_hop_len)
            if snore_frame_hop_len != int(snore_frame_hop_len):
                raise ValueError(f"Snore frame hop length {snore_frame_hop_len} is not an integer. Adjust settings.")
            if snore_frame_size_len != int(snore_frame_size_len):
                raise ValueError(f"Snore frame size length {snore_frame_size_len} is not an integer. Adjust settings.")
            
            input_files = set([f.split(".")[0] for f in os.listdir(raw_input_dir) if f.endswith('.wav')])
            output_files = set([f.split(".")[0] for f in os.listdir(raw_output_dir) if f.endswith('.npy')])
            if output_files != input_files:
                unmatched_input = list(map(lambda x: x + '.wav', output_files - input_files))
                unmatched_output = list(map(lambda x: x + '.npy', input_files - output_files))
                raise FileNotFoundError(f"Unmatched files. Each input file must have a matching output file.\n\nUnmatched input file(s): {unmatched_input}.\n\nUnmatched output file(s): {unmatched_output}.")
            self.files = input_files
        
        except Exception as e:
            raise e
        
        
    def compute_input_frames(self):

        # checks files that need processing
        all_patients = set([int(f.split(".")[0].split("_")[0]) for f in self.files])
        processed_patients = set([int(f.split(".")[0]) for f in os.listdir(self.input_frames_dir) if f.endswith('.parquet')])
        patients_to_process = all_patients - processed_patients

        # process patients in parallel
        pbar = tqdm(total=len(patients_to_process), desc="Computing input frames", unit="patient")
        with ThreadPoolExecutor(max_workers=MAX_THREADS_CIF) as executor:
            futures = [executor.submit(self.process_patient_input_frames, patient_id, pbar) for patient_id in patients_to_process]
            try: 
                for future in futures:
                    future.result()
            except KeyboardInterrupt:
                executor.shutdown(cancel_futures=True)
                raise
            except Exception as e:
                executor.shutdown(cancel_futures=True)
                raise e
        pbar.close()

    def compute_output_frames(self):

        # checks files that need processing
        all_patients = set([int(f.split(".")[0].split("_")[0]) for f in self.files])
        processed_patients = set([int(f.split(".")[0]) for f in os.listdir(self.output_frames_dir) if f.endswith('.parquet')])
        patients_to_process = all_patients - processed_patients

        # process patients in parallel
        pbar = tqdm(total=len(patients_to_process), desc="Computing output frames", unit="patient")
        with ThreadPoolExecutor(max_workers=MAX_THREADS_COF) as executor:
            futures = [executor.submit(self.process_patient_output_frames, patient_id, pbar) for patient_id in patients_to_process]
            try: 
                for future in futures:
                    future.result()
            except KeyboardInterrupt:
                executor.shutdown(cancel_futures=True)
                raise
            except Exception as e:
                executor.shutdown(cancel_futures=True)
                raise e
        pbar.close()

    def process_patient_output_frames(self, patient_id:int, pbar:tqdm=None):

        patient_files = [f for f in self.files if f.startswith(f"{patient_id}_")]

        frame_size_len = self.settings['snore_frame_size_len']
        frame_hop_len = self.settings['snore_frame_hop_len']

        # process each file for the patient
        dfs: list[pd.DataFrame] = list()
        for filename in patient_files:

            filepath = os.path.join(self.raw_output_dir, filename + '.npy')
            snore_signal = np.load(filepath)
            snore_signal = snore_signal.astype(np.float32) / 32768.0

            num_frames = (len(snore_signal) - frame_size_len) // frame_hop_len + ((len(snore_signal) - frame_size_len) % 2)

            df = pd.DataFrame()
            df['frame'] = range(num_frames)
            df['file_index'] = int(filename.split('_')[1])
            df['rms'] = lr.feature.rms(y=snore_signal, frame_length=frame_size_len, hop_length=frame_hop_len, center=False).flatten()[:-1]
            df["dbfs"] = df["rms"].apply(lambda x: 20 * np.log10(x) if x > 0 else -100)
            dfs.append(df)
        
        # concatenates all dataframes and saves to parquet
        patient_df = pd.concat(dfs, ignore_index=True)
        output_path = os.path.join(self.output_frames_dir, f"{patient_id}.parquet")
        patient_df.attrs['patient_id'] = patient_id
        patient_df.attrs['frame_size_len'] = frame_size_len
        patient_df.attrs['frame_hop_len'] = frame_hop_len
        patient_df.attrs['snore_sample_rate'] = self.settings['snore_sample_rate']
        patient_df.to_parquet(output_path, index=False)

        if pbar:
            with PBAR_LOCK:
                pbar.update(1)

    def process_patient_input_frames(self, patient_id:int, pbar:tqdm=None):

        patient_files = [f for f in self.files if f.startswith(f"{patient_id}_")]

        frame_size_len = self.settings['audio_frame_size_len']
        frame_hop_len = self.settings['audio_frame_hop_len']

        # process each file for the patient
        dfs: list[pd.DataFrame] = list()
        for filename in patient_files:
            

            filepath = os.path.join(self.raw_input_dir, filename + '.wav')
            waveform, sample_rate = sf.read(filepath)

            if sample_rate != self.settings['audio_sample_rate']:
                raise ValueError(f"Sample rate {sample_rate} does not match expected {self.settings['audio_sample_rate']}.")
            
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            
            num_frames = (len(waveform) - frame_size_len) // frame_hop_len + ((len(waveform) - frame_size_len) % 2)

            df = pd.DataFrame()
            df['frame'] = range(num_frames)
            df['file_index'] = int(filename.split('_')[1])
            
            Process.compute_time_domain_features(df, waveform, frame_size_len, frame_hop_len)
            Process.compute_frequency_domain_features(df, waveform, sample_rate, frame_size_len, frame_hop_len)
            Process.compute_mfccs(df, waveform, sample_rate, frame_size_len, frame_hop_len, n_mfcc=self.settings["n_mfcc"])
            
            dfs.append(df)

        # concatenates all dataframes and saves to parquet
        patient_df = pd.concat(dfs, ignore_index=True)
        output_path = os.path.join(self.input_frames_dir, f"{patient_id}.parquet")
        patient_df.attrs['patient_id'] = patient_id
        patient_df.attrs['frame_size_len'] = frame_size_len
        patient_df.attrs['frame_hop_len'] = frame_hop_len
        patient_df.attrs['audio_sample_rate'] = self.settings['audio_sample_rate']
        patient_df.to_parquet(output_path, index=False)

        if pbar:
            with PBAR_LOCK:
                pbar.update(1)

    def compute_input_sequences(self):
        
        # checks files that need processing
        all_patients = set([int(f.split(".")[0].split("_")[0]) for f in self.files])
        processed_patients = set([int(f.split(".")[0]) for f in os.listdir(self.input_sequences_dir) if f.endswith('.parquet')])
        patients_to_process = all_patients - processed_patients

        # process patients in parallel
        pbar = tqdm(total=len(patients_to_process), desc="Computing input sequences", unit="patient")
        with ThreadPoolExecutor(max_workers=MAX_THREADS_CIS) as executor:
            futures = [executor.submit(self.process_patient_sequences, patient_id, pbar, self.input_frames_dir, self.input_sequences_dir) for patient_id in patients_to_process]
            try: 
                for future in futures:
                    future.result()
            except KeyboardInterrupt:
                executor.shutdown(cancel_futures=True)
                raise
            except Exception as e:
                executor.shutdown(cancel_futures=True)
                raise e
        pbar.close()

    def compute_output_sequences(self):

        # checks files that need processing
        all_patients = set([int(f.split(".")[0].split("_")[0]) for f in self.files])
        processed_patients = set([int(f.split(".")[0]) for f in os.listdir(self.output_sequences_dir) if f.endswith('.parquet')])
        patients_to_process = all_patients - processed_patients

        # process patients in parallel
        pbar = tqdm(total=len(patients_to_process), desc="Computing output sequences", unit="patient")
        with ThreadPoolExecutor(max_workers=MAX_THREADS_COS) as executor:
            futures = [executor.submit(self.process_patient_sequences, patient_id, pbar, self.output_frames_dir, self.output_sequences_dir) for patient_id in patients_to_process]
            try: 
                for future in futures:
                    future.result()
            except KeyboardInterrupt:
                executor.shutdown(cancel_futures=True)
                raise
            except Exception as e:
                executor.shutdown(cancel_futures=True)
                raise e
        pbar.close()
    
    def process_patient_sequences(self, patient_id:int, pbar:tqdm=None, frames_dir:str=None, sequences_dir:str=None):

        patient_file = os.path.join(frames_dir, f"{patient_id}.parquet")
        if not os.path.isfile(patient_file):
            return
        
        df = pd.read_parquet(patient_file)
        sequence_size_len = self.settings['sequence_size_len']
        sequence_hop_len = self.settings['sequence_hop_len']

        tensor = torch.tensor(df.values, device="cuda")
        tensor = tensor.T
        tensor = tensor.unfold(dimension=1, size=sequence_size_len, step=sequence_hop_len)
        tensor = tensor.permute(1,0,2).contiguous()
        tensor = tensor.mean(dim=-1)
        columns = ["sequence"] + [col for col in df.columns if col != "frame"]
        df_sequences = pd.DataFrame(tensor.cpu().numpy(), columns=columns)
        df_sequences["sequence"] = df_sequences.index

        output_filepath = os.path.join(sequences_dir, f"{patient_id}.parquet")
        df_sequences.to_parquet(output_filepath, index=False)

        with PBAR_LOCK:
            pbar.update(1)

    @staticmethod
    def compute_time_domain_features(df:pd.DataFrame, waveform:np.ndarray, frame_size_len:int, frame_hop_len:int):
        df["rms"] = lr.feature.rms(y=waveform, frame_length=frame_size_len, hop_length=frame_hop_len, center=False).flatten()[:-1]
        n = df["rms"].isna().sum() - len(df)
        df["zcr"] = lr.feature.zero_crossing_rate(y=waveform, frame_length=frame_size_len, hop_length=frame_hop_len, center=False).flatten()[:-1]
        n = df["zcr"].isna().sum() - len(df)

    @staticmethod
    def compute_frequency_domain_features(df:pd.DataFrame, filename:str, waveform:np.ndarray, sample_rate:int, frame_size_len:int, frame_hop_len:int):
        df["scr"] = lr.feature.spectral_centroid(y=waveform, sr=sample_rate, n_fft=frame_size_len, hop_length=frame_hop_len, center=False).flatten()[:-1]
        n = df["scr"].isna().sum() - len(df)
        df["sbw"] = lr.feature.spectral_bandwidth(y=waveform, sr=sample_rate, n_fft=frame_size_len, hop_length=frame_hop_len, center=False).flatten()[:-1]
        n = df["sbw"].isna().sum() - len(df)
        df["srf"] = lr.feature.spectral_rolloff(y=waveform, sr=sample_rate, n_fft=frame_size_len, hop_length=frame_hop_len, center=False).flatten()[:-1]
        n = df["srf"].isna().sum() - len(df)
        df["sfl"] = lr.feature.spectral_flatness(y=waveform, n_fft=frame_size_len, hop_length=frame_hop_len, center=False).flatten()[:-1]
        n = df["sfl"].isna().sum() - len(df)
        tmp = np.abs(lr.stft(y=waveform, n_fft=frame_size_len, hop_length=frame_hop_len, center=False))
        tmp = tmp / (np.sum(tmp, axis=0, keepdims=True) + 1e-10)
        tmp = np.sqrt(np.sum((tmp[:, 1:] - tmp[:, :-1])**2, axis=0))[:-1]
        df["sfx"] = np.concatenate(([0], tmp))
        n = df["sfx"].isna().sum() - len(df)

    @staticmethod
    def compute_mfccs(df:pd.DataFrame, waveform:np.ndarray, sample_rate:int, frame_size_len:int, frame_hop_len:int, n_mfcc:int=13):
        mfccs = lr.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=n_mfcc, n_fft=frame_size_len, hop_length=frame_hop_len, center=False)[:,:-1]
        del waveform
        mfccs_delta = lr.feature.delta(mfccs)
        mfccs_delta2 = lr.feature.delta(mfccs, order=2)
        for i in range(n_mfcc):
            df[f"mfcc_{i+1}"] = mfccs[i]
            df[f"mfcc_d1_{i+1}"] = mfccs_delta[i]
            df[f"mfcc_d2_{i+1}"] = mfccs_delta2[i]