from typing import List, Tuple, Union
import numpy as np
import torch
import matplotlib.pyplot as plt
from pyannote.audio.pipelines.voice_activity_detection import VoiceActivityDetection
from pyannote.audio import Pipeline

from scipy.stats import norm
from numpy.lib.stride_tricks import as_strided
from sklearn.cluster import KMeans

from scipy.ndimage import label
from itertools import permutations
import os

# Get token required for model acess
HUGGING_FACE_TOKEN = os.environ.get('HUGGING_FACE_TOKEN')

def create_overlapping_windows(arr: Union[np.ndarray, torch.tensor], window_size: int, window_offset: int):
    
    if isinstance(arr, torch.Tensor):
        arr = np.array(arr)
    
    overlap = window_size - window_offset
    
    shape = ((arr.shape[-1] - overlap) // window_offset, window_size)
    strides = (arr.strides[0] * window_offset, arr.strides[0])

    return as_strided(arr, shape=shape, strides=strides)

class EnergyBasedExMaxVAD:

    """https://www.isca-archive.org/interspeech_2021/ichikawa21_interspeech.html"""

    def __init__(self, 
                 frame_len: int = 128, 
                 frame_offset: int = 64,
                 speech_thr: float = 0.5
                 ):

        # Single-channel reference model to detect silent segments
        self.reference_vad_model = VoiceActivityDetection()
        
        # Single-channel diarization model
        self.far_field_diarization_model = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.0",
                use_auth_token=HUGGING_FACE_TOKEN
            )

        self.frame_len = frame_len
        self.frame_offset = frame_offset
        self.speech_thr = speech_thr

    def _compute_non_speech_intervals(self, file):

        sample_rate = file['sample_rate']

        # Sum audio to mono to detect silent segments
        summed_audio = file['waveform'].sum(0, keepdims=True)
        summed_audio = summed_audio/torch.max(summed_audio)
        summed_file = {
            "waveform": summed_audio,
            "sample_rate": sample_rate
                        }
        
        summed_vad = self.reference_vad_model(summed_file)

        total_len_s = summed_audio.shape[-1]/sample_rate
        speech_intervals = [ (float(seg.start), float(seg.end)) for seg in summed_vad.itersegments()]
        non_speech_intervals = self._compute_complement_intervals(speech_intervals, total_len_s)

        return non_speech_intervals
    
    def _compute_complement_intervals(self, intervals: List[Tuple], total_len_s: float):

        intervals = sorted(intervals, key=lambda elem: elem[0]) # Sort by start time

        first_interval_start = intervals[0][0]
        last_interval_end = intervals[-1][1]

        complement_intervals = []
        if first_interval_start > 0.0:
            complement_intervals.append( (0.00, first_interval_start) )

        for i in range(len(intervals) - 1):

            assert intervals[i][1] < intervals[i+1][0], 'intervals must be disjoint'
            complement_intervals.append( (intervals[i][1], intervals[i+1][0]) )

        if last_interval_end < total_len_s:
            complement_intervals.append( (last_interval_end, total_len_s) )

        return complement_intervals

    
    def _normalize_per_channel(self, file):

        n_channels = file['waveform'].shape[0]
        sample_rate = file['sample_rate']

        # Compute intervals without speech on any channel
        non_speech_intervals = self._compute_non_speech_intervals(file)
        non_speech_intervals_samp = [ (int(intval[0]*sample_rate), int(intval[1]*sample_rate)) for intval in non_speech_intervals ] # Convert to samples

        # Compute energy of each channel and normalize
        for chan_idx in range(n_channels):
            
            channel_audio = file['waveform'][chan_idx]
            non_speech_audio = np.concatenate([channel_audio[s:e] for (s,e) in non_speech_intervals_samp])

            energy = np.sqrt( np.mean( non_speech_audio**2 ) ) # RMS energy
            file['waveform'][chan_idx] /= energy # Apply energy normalization

        return file
    
    def _diarize_from_params(self, file, frame_energy, theta):

        sample_rate = file['sample_rate']
        waveform = file['waveform']

        N = frame_energy.shape[0]

        Pz = self._compute_pz(frame_energy, theta) # Compute speech probabilities per frame

        # Resample to original time resolution
        t_orig = np.arange(waveform.shape[-1])
        t_windowed = create_overlapping_windows(t_orig, self.frame_len, self.frame_offset)
        t_frame = t_windowed[:, self.frame_len//2]

        Pz_interp = [
            np.interp(t_orig, t_frame, Pz[n]).clip(0,1)
            for n in range(N)
        ]

        speech_regions = []
        for n in range(N):
            mask = Pz_interp[n] > self.speech_thr
            labeled_array, num_features = label(mask)
            regions = [np.where(labeled_array == i)[0] for i in range(1, num_features+1)]
            region_bounds = [(np.min(region), np.max(region)) for region in regions]
            speech_regions.append(region_bounds)

        # TODO add post-processing
        diarizization_outputs = []
        for speaker_idx, regions in enumerate(speech_regions):
            for (start, end) in regions:
                diarizization_outputs.append( (str(speaker_idx), start/sample_rate, end/sample_rate) )

        return diarizization_outputs

    def __call__(self, lav_file, array_file):
        
        n_channels = lav_file['waveform'].shape[0]

        lav_file = self._normalize_per_channel(lav_file) # Normalize each channel using silent portions

        windowed_channels = np.stack([
            create_overlapping_windows(lav_file['waveform'][i], self.frame_len, self.frame_offset) 
            for i in range(n_channels)
        ], 0) # shape = (channels, n_frames, frame_len)

        frame_energy = np.sqrt( np.mean( windowed_channels**2, -1 ) ) # shape = (channels, n_frames)

        theta_init = self.initalize_theta(array_file, frame_energy, num_speakers=n_channels)
        theta = self.em(frame_energy, theta_init)

        diarizization_outputs = self._diarize_from_params(lav_file, frame_energy, theta)

        return diarizization_outputs
    
    def initalize_theta(self, array_file, frame_energy: np.ndarray, num_speakers: int):
        
        file_len = array_file['waveform'].shape[-1]
        sample_rate = array_file['sample_rate']
        T = frame_energy.shape[-1]

        def _locate_active_frames(intervals):

            intervals_sampl = [(int(sample_rate * start_s), int(sample_rate * end_s))
                        for start_s, end_s in intervals
                    ]
            
            is_active = np.zeros(file_len, dtype=np.int32)
            for (s,e) in intervals_sampl:
                is_active[s:e] = 1

            is_active_windowed = create_overlapping_windows(is_active, self.frame_len, self.frame_offset)
            is_active_frame = np.sum(is_active_windowed, -1) >= self.frame_len//2

            return is_active_frame


        # Get predictions from array diarization model
        ff_diarization = self.far_field_diarization_model(array_file, num_speakers=num_speakers)
        results = [ (spkr, speech_turn.start, speech_turn.end)
                for speech_turn, spkr, _ in ff_diarization.itertracks(yield_label=True)
            ]

        liklihoods = []
        thetas = []
        
        # Test each possible channel alignment 
        for channel_perm in permutations(range(num_speakers)):
            
            # Apply channel mapping
            channel_mapping = dict(zip(range(num_speakers), channel_perm))
            results_remapped = [ (channel_mapping[spkr], start, end)
                for spkr, start, end in results
            ]

            pi = np.zeros(num_speakers)
            mu_a = np.zeros(num_speakers)
            mu_s = np.zeros(num_speakers)
            sigma_a = np.zeros(num_speakers)
            sigma_s = np.zeros(num_speakers)

            for n in range(num_speakers): # For each channel

                speech_intervals = [(s,e) for spkr, s, e in results_remapped if spkr == n]
                active_frames = _locate_active_frames(speech_intervals)
                 
                if active_frames.sum() == 0: # No active frames identified for this speaker

                    pi[n] = 0.01
                    mu_s[n] = np.mean(frame_energy[n])
                    sigma_s[n] = np.sqrt( 1/(T - 1) * np.sum( (frame_energy[n] - mu_s[n])**2 ) )

                    mu_a[n] = 10 * mu_s[n]
                    sigma_a[n] = sigma_s[n]

                else:
                    x_a = frame_energy[n][active_frames == 1]
                    x_s = frame_energy[n][active_frames == 0]

                    pi[n] = np.sum(active_frames)/T
                    mu_a[n] = np.mean(x_a)
                    sigma_a[n] = np.sqrt( 1/(len(x_a) - 1) * np.sum( (x_a - mu_a[n])**2 ) )

                    mu_s[n] = np.mean(x_s)
                    sigma_s[n] = np.sqrt( 1/(len(x_s) - 1) * np.sum( (x_s - mu_s[n])**2 ) )
            
            theta = (pi, mu_a, sigma_a, mu_s, sigma_s)

            thetas.append(theta)
            liklihoods.append(np.mean(self.liklihood(frame_energy, theta)))

        theta_init = thetas[ np.argmax(liklihoods) ]

        return theta_init

    def _compute_pz(self, frame_energy: np.ndarray, theta: Tuple):

        N = frame_energy.shape[0]

        (pi, mu_a, sigma_a, mu_s, sigma_s) = theta

        Pz = np.zeros_like(frame_energy) # Shape (N,T)
        for n in range(N): # For each channel
            
            xn = frame_energy[n]

            Pz[n] = (pi[n] * norm.pdf(xn, mu_a[n], sigma_a[n]) )/( 
                    pi[n] * norm.pdf(xn, mu_a[n], sigma_a[n]) 
                    + 
                    (1-pi[n]) * norm.pdf(xn, mu_s[n], sigma_s[n]) 
                )
        
        return Pz
    
    def liklihood(self, frame_energy: np.ndarray, theta: Tuple):

        N = frame_energy.shape[0]

        (pi, mu_a, sigma_a, mu_s, sigma_s) = theta
        
        X = frame_energy

        L = np.array(
            [
                np.sum( 
                        np.log( 
                            pi[n] * norm.pdf(X[n], mu_a[n], sigma_a[n]) + 
                            (1-pi[n]) * norm.pdf(X[n], mu_s[n], sigma_s[n]) 
                        ) 
                    ) 
             for n in range(N)]
        )

        return L

    def em_step(self, frame_energy: np.ndarray, theta: Tuple):

        # Compute P( z = 1 ) for each frame and channel
        Pz = self._compute_pz(frame_energy, theta)
                
        # Compute updated parameters to maximize expected log-likihood
        X = frame_energy
        
        pi = np.mean(Pz, -1)
       
        mu_a = np.sum(X * Pz, -1)/np.sum(Pz, -1)
        mu_s = np.sum(X * (1-Pz), -1)/np.sum((1-Pz), -1)

        sigma_a = np.sqrt( 
            np.sum( ((X - mu_a.reshape(-1,1))**2) * Pz, -1)
            /
            np.sum( Pz, -1) 
        )
        sigma_s = np.sqrt( 
            np.sum(((X - mu_s.reshape(-1,1))**2) * (1-Pz), -1)
            /
            np.sum( (1-Pz), -1) 
        )

        theta_new = (pi, mu_a, sigma_a, mu_s, sigma_s)

        assert np.isfinite( theta_new ).all()
        assert np.min(self.liklihood(frame_energy, theta_new) - self.liklihood(frame_energy, theta)) > -1e-3

        return theta_new

    def em(self, frame_energy: np.ndarray, theta_init, converge_thr: float = 0.01, max_steps: int = 1000):

        theta = theta_init
        for i in range(max_steps):
            
            theta_new = self.em_step(frame_energy, theta)

            delta = self.liklihood(frame_energy, theta_new) - self.liklihood(frame_energy, theta)

            if np.max(delta) < converge_thr:
                print(f'converged on step {i+1}')
                return theta_new
            
            else:
                theta = theta_new

        # If we reach here, converge thr is not met
        print('WARNING: unable to converge')
        return theta
    

if __name__ == "__main__":

    from multiparty_diarization.datasets.ami_multichannel import AMIMultiChannel
    from multiparty_diarization.eval.metrics import compute_sample_diarization_metrics

    model = EnergyBasedExMaxVAD(speech_thr=0.5, frame_len=64)

    dset = AMIMultiChannel(
                    array_root_path = '/proj/disney/amicorpus_array',
                    lav_root_path = '/proj/disney/amicorpus_lav',
                    sample_len_s = 50
                    )
    
    (array_audio, lav_audio), sample, sample_info = dset[ np.random.randint(0, len(dset)) ]

    lav_file = {"waveform": lav_audio, "sample_rate": 16000}
    array_file = {"waveform": array_audio, "sample_rate": 16000}

    preds = model(lav_file, array_file)

    array_preds = model.far_field_diarization_model(array_file)
    array_preds = [ (str(spkr), speech_turn.start, speech_turn.end)
                for speech_turn, spkr, _ in array_preds.itertracks(yield_label=True)
            ]

    metrics = compute_sample_diarization_metrics(sample, preds)
    metrics_ff = compute_sample_diarization_metrics(sample, array_preds)

    for metric, val in metrics.items():
        print(f"{metric}: {val:.4f} \ {metrics_ff[metric]:.4f}")

