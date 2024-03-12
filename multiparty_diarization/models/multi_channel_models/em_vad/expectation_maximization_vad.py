from typing import List, Tuple, Union
import numpy as np
import torch
from pyannote.audio.pipelines import VoiceActivityDetection
from multiparty_diarization.models.model_wrapper import DiarizationModel
from multiparty_diarization.models.multi_channel_models.em_vad.pyannote_diarization_probs import SpeakerDiarizationProbs

from scipy.stats import multivariate_normal
from numpy.lib.stride_tricks import as_strided
from pyannote.audio.core.io import AudioFile

from scipy.ndimage import label
from itertools import permutations

def create_overlapping_windows(arr: Union[np.ndarray, torch.tensor], window_size: int, window_offset: int) -> np.ndarray:
    
    """Breaks signal into overlapped frames
    
    Args:
        arr (np.ndarray, torch.tensor): 1D array to be segmented
        window_size (int): length of each window
        window_offset (int): step size between adjacent windows

    Returns:
        windowed_arr (np.ndarray): 2D array with shape (num_frames, window_size)
    """

    if isinstance(arr, torch.Tensor):
        arr = np.array(arr)
    
    overlap = window_size - window_offset
    
    shape = ((arr.shape[-1] - overlap) // window_offset, window_size)
    strides = (arr.strides[0] * window_offset, arr.strides[0])

    return as_strided(arr, shape=shape, strides=strides)

class EMCloseTalkVAD(DiarizationModel):

    """Expectation Maximization based VAD using close talk channels"""

    def __init__(self, 
                window_size: int = 128, 
                window_offset: int = 64,
                speech_thr: float = 0.5,
                convergence_thr: float = 1e-5,
                max_steps: int = 100,
                max_agg: bool = False,
                sample_rate: int = 16000
            ):
        
        """Creates instance of EMCloseTalkVAD model
        
        Args:
            window_size (int): length of analysis window
            window_offset (int): offset between adjacent analysis windows
            speech_thr (float): threshold for binarization of speech activity, used only if max_agg == False
            convergence_thr (float): increase in liklihood sufficient for convergence of EM
            max_steps (int): maximum number of EM steps
            max_agg (bool): use most probable z for diarazation rather then expectation of Z
            sample_rate (int): audio sampling rate
        """

        super().__init__()

        # Single-channel reference model to detect silent segments
        self.reference_vad_model = VoiceActivityDetection()
        
        # Single-channel diarization model for EM warm-start initialization
        self.far_field_diarization_model = SpeakerDiarizationProbs()

        # Save params
        self.window_size = window_size
        self.window_offset = window_offset
        self.speech_thr = speech_thr
        self.convergence_thr = convergence_thr
        self.max_steps = max_steps
        self.max_agg = max_agg
        self.sample_rate = sample_rate

    def _compute_non_speech_intervals(self, lav_audio: np.ndarray) -> List[Tuple]:

        """Identifies non-speech segments in audio file for normalization purposes

        Args:
            lav_audio (np.ndarray): lav audio samples with shape (n_channels, time)

        Returns:
            intervals (List[Tuple]): list of silent intervals of form (start,end)        
        """

        # Sum audio to mono to detect silent segments
        summed_audio = lav_audio.sum(0, keepdims=True)
        summed_audio = summed_audio/torch.max(summed_audio)
        summed_file = {
            "waveform": summed_audio,
            "sample_rate": self.sample_rate
                    }
        
        # Call VAD model
        summed_vad = self.reference_vad_model(summed_file)

        # Locate non-speech intervals
        total_len_s = summed_audio.shape[-1]/self.sample_rate
        speech_intervals = [ (float(seg.start), float(seg.end)) for seg in summed_vad.itersegments()]

        non_speech_intervals = self._compute_complement_intervals(speech_intervals, total_len_s)

        return non_speech_intervals
    
    def _compute_complement_intervals(self, intervals: List[Tuple], total_len_s: float) -> List[Tuple]:

        """Utility function to compute complement to a set of intervals
        
        Args:
            intervals (List[Tuple]): original intervals
            total_len_s (float): total length of file in seconds
        """

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

    
    def _normalize_per_channel(self, lav_audio: np.ndarray) -> np.ndarray:

        """Performs silence based per-channel normalization using the method from
        https://www.isca-archive.org/interspeech_2021/ichikawa21_interspeech.html
        
        Args:
            lav_audio (np.ndarray): lav audio samples with shape (n_channels, time)
        
        Returns:
            normalized_audio (np.ndarray)    
        """

        n_channels = lav_audio.shape[0]

        # Compute intervals without speech on any channel
        non_speech_intervals = self._compute_non_speech_intervals(lav_audio)
        non_speech_intervals_samp = [ (int(interval[0]*self.sample_rate), int(interval[1]*self.sample_rate)) 
                                     for interval in non_speech_intervals ] # Convert to samples

        # Compute energy of each channel and normalize
        for chan_idx in range(n_channels):
            
            channel_audio = lav_audio[chan_idx]
            non_speech_audio = np.concatenate([channel_audio[s:e] for (s,e) in non_speech_intervals_samp])

            energy = np.sqrt( np.mean( non_speech_audio**2 ) ) # RMS energy
            lav_audio[chan_idx] /= energy # Apply energy normalization

        return lav_audio
    
    def _diarize_from_params(self, frame_energy: np.ndarray, theta: tuple, file_len_samp: int) -> List[Tuple[str, float, float]]:

        """Generates diarization using EM predictions
        
        Args:
            frame_energy (np.ndarray): per-channel energy at each time step
            theta (tuple): EM params
            file_len_samp (int): length of original audio file in samples 
        
        Returns:
            results (list): list of tuples for each segment of the form (speaker, start, end)
        """

        N = frame_energy.shape[0]

        Z, P_z = self._compute_P_z(frame_energy, theta) # Compute latent probabilities per frame
        Nz, T = P_z.shape
        Z = np.array(Z)
        Z = Z.T.reshape(N, Nz, 1)

        # Resample to original time resolution
        t_orig = np.arange(file_len_samp)
        t_windowed = create_overlapping_windows(t_orig, self.window_size, self.window_offset)
        t_frame = t_windowed[:, self.window_size//2]
        P_z_interp = np.array([
            np.interp(t_orig, t_frame, P_z[n]).clip(0,1)
            for n in range(Nz)
        ])

        if self.max_agg: # Select most-likley z for each frame
            Z_agg = np.squeeze(Z[:, np.argmax(P_z_interp, 0)]) # (N, file_len)
        else: # Take average of Z
            Z_agg = np.sum( Z * P_z_interp.reshape(1, Nz, file_len_samp), axis=(1) ) # (N, file_len)
        

        # Locate speech regions per-channel
        speech_regions = []
        for n in range(N):
            
            if self.max_agg:
                mask = Z_agg[n]
            else:
                mask = Z_agg[n] > self.speech_thr
            
            labeled_array, num_features = label(mask)
            regions = [np.where(labeled_array == i)[0] for i in range(1, num_features+1)]
            region_bounds = [(np.min(region), np.max(region)) for region in regions]
            speech_regions.append(region_bounds)

        # Convert to diarization format: (speaker, start_s, end_s)
        diarizization_outputs = []
        for speaker_idx, regions in enumerate(speech_regions):
            for (start, end) in regions:
                diarizization_outputs.append( (str(speaker_idx), start/self.sample_rate, end/self.sample_rate) )

        return diarizization_outputs

    def __call__(self, audio: tuple, **kwargs) -> List[Tuple[str, float, float]]:

        """Applies diariazation to audio signal
        
        Args:
            waveform (tuple): tuple of array_audio, lav_audio
            kwargs (dict, optional): additional kwargs to support model-specific needs

        Returns:
            results (list): list of tuples for each segment of the form (speaker, start, end)
        """

        array_audio, lav_audio = audio
        n_channels = lav_audio.shape[0]

        lav_audio = self._normalize_per_channel(lav_audio) # Normalize each channel using silent portions

        array_file = {"waveform": array_audio, "sample_rate": self.sample_rate}

        windowed_channels = np.stack([
            create_overlapping_windows(lav_audio[i], self.window_size, self.window_offset) 
            for i in range(n_channels)
        ], 0) # shape = (channels, n_frames, window_size)

        frame_energy = np.sqrt( np.mean( windowed_channels**2, -1 ) ) # RMS energy shape = (channels, n_frames)

        # Initialize theta using array audio diarization
        theta_init = self.initalize_theta(array_file, frame_energy, num_speakers=n_channels)

        # Run EM
        theta, liklihoods = self.em(frame_energy, theta_init)

        # Compute diarization
        diarizization_outputs = self._diarize_from_params(frame_energy, theta, lav_audio.shape[-1])

        return diarizization_outputs
    
    def initalize_theta(self, 
                        array_file: AudioFile, 
                        frame_energy: np.ndarray, 
                        num_speakers: int, 
                        smoothing: float = 0.1
                    ) -> tuple:

        """Initializes EM parameters using array audio diarization results
        
        Args:
            array_file (AudioFile): array audio file in dict form
            frame_energy (np.ndarray): per-channel energy at each time step
            num_speakers (int): number of speakers
            smoothing (float): amount by which to smooth array speaker probabilities

        Returns:
            theta_init (tuple): initial params
        """

        N, T = frame_energy.shape

        # Get predictions from array diarization model (N,T')
        P_s = self.far_field_diarization_model(array_file, num_speakers=num_speakers)
        Np, Tp = P_s.shape

        if Np < N: # If array model predicts fewer speakers, add zero-probability channels to match number of lav channels
            P_s = np.concatenate((np.zeros((N - Np, Tp)), P_s), 0)

        # Interpolate to same timeline
        t_new = np.arange(T)
        t_old = np.arange(Tp)

        P_s = np.array(
                [np.interp(t_new, t_old, p_s) for p_s in P_s]
            )
        
        # Smoothing
        P_s = (1-smoothing) * P_s + smoothing * 0.5 * np.ones_like(P_s)

        Z = np.array( self._list_all_z(N) )

        liklihoods = []
        thetas = []
        
        # Test each possible channel alignment 
        for channel_perm in permutations(range(num_speakers)):
            
            Ps_perm = P_s[np.array(channel_perm)]

            P_z = np.array(
                    [ np.prod( (Ps_perm**z.reshape(-1,1)) * ( (1-Ps_perm)**(1-z.reshape(-1,1)) ), axis=0) for z in Z ]
                )
            
            pi = self._compute_pi(P_z, Z)
            M = self._compute_M(frame_energy, P_z, Z)
            Sigma = self._compute_Sigma(frame_energy, P_z, Z, M)

            theta = (pi, M, Sigma)

            thetas.append( theta )
            liklihoods.append( self.liklihood(frame_energy, theta) )


        theta_init = thetas[ np.argmax(liklihoods) ]
        
        return theta_init

    def _list_all_z(self, N):
        if N == 1:
            return [[1], [0]]
        else: 
            return [[1] + x for x in self._list_all_z(N-1)] + [[0] + x for x in self._list_all_z(N-1)]

    def _compute_P_zx(self, frame_energy: np.ndarray, theta: Tuple):
        """Computes a matrix P_zx with shape (Nz,T) where element i,j = P(x_j,z_i|theta)"""

        pi, M, Sigma = theta

        N = frame_energy.shape[0]
        Z = self._list_all_z(N)

        P_zx = np.array( 
                [ 
                    multivariate_normal.pdf( frame_energy.T, M@z, Sigma, allow_singular=True )
                    * np.prod(( pi ** z) * ((1 - pi)**(1 - z)) ) 
                    for z in np.array(Z) 
                ]
            )
               
        return Z, P_zx
    
    def _compute_P_z(self, frame_energy: np.ndarray, theta: Tuple):
        """Generate matrix P_z with shape (Nz, T) where element i,j = P(z_i|x_j, theta)"""

        Z, P_zx = self._compute_P_zx(frame_energy, theta)

        Pz = P_zx/(np.sum(P_zx, 0, keepdims=True) + 1e-30)

        return Z, Pz
    
    def liklihood(self, frame_energy: np.ndarray, theta: Tuple):

        _, P_zx = self._compute_P_zx(frame_energy, theta)
        _, P_z = self._compute_P_z(frame_energy, theta)
        L = np.mean( np.log(np.sum( P_zx, 0 ) + 1e-40))

        return L

    def _compute_pi(self, P_z: np.ndarray, Z: list):
        
        Z = np.array(Z)
        
        Nz, N = Z.shape
        _, T = P_z.shape # (Nz, T)

        Z = Z.T.reshape(N, Nz, 1)

        num = np.sum( Z * P_z.reshape(1, Nz, T), axis=(1,2) ) # (N,)
        den = np.sum(P_z)

        pi = num/den

        return pi

    
    def _compute_M(self, frame_energy: np.ndarray, P_z: np.ndarray, Z: list):

        Z = np.array(Z) # (Nz, N)
        X = frame_energy # (N, T)

        Nz, N = Z.shape
        _, T = P_z.shape # (Nz, T)

        ZZ_T = Z.reshape(Nz, N, 1) @ Z.reshape(Nz, 1, N) # (Nz, N, N)
        Mz = P_z.reshape(Nz, T, 1, 1) * ZZ_T.reshape(Nz, 1, N, N) # (Nz, T, N, N)

        XZ_T = X.T.reshape(1, T, N, 1) @ Z.reshape(Nz, 1, 1, N) # (Nz, T, N, N)
        Mx = P_z.reshape(Nz, T, 1, 1) *  XZ_T # (Nz, T, N, N)

        M = np.sum(Mx, (0, 1) ) @  np.linalg.inv( np.sum(Mz, (0, 1)) )

        return M
    
    def _compute_Sigma(self, frame_energy: np.ndarray, P_z: np.ndarray, Z: list, M: np.ndarray):

        Z = np.array(Z) # (Nz, N)
        X = frame_energy # (N, T)

        Nz, N = Z.shape
        _, T = P_z.shape # (Nz, T)
 
        MZ = (M @ Z.T).T # (Nz, N)
        x_mu = X.T.reshape(1, T, N) - MZ.reshape(Nz, 1, N) # (Nz, T, N)

        Sigma_raw = x_mu.reshape(Nz, T, N, 1,) @ x_mu.reshape(Nz, T, 1, N) # (Nz, T, N, N)

        Sigma = 1/T * np.sum( P_z.reshape(Nz, T, 1, 1) * Sigma_raw, axis=(0,1) ) # (N, N)

        return Sigma

    def em_step(self, frame_energy: np.ndarray, theta: tuple):

        Z, P_z = self._compute_P_z(frame_energy, theta)

        pi_star = self._compute_pi(P_z, Z)
        M_star = self._compute_M(frame_energy, P_z, Z)
        Sigma_star = self._compute_Sigma(frame_energy, P_z, Z, M_star)

        return (pi_star, M_star, Sigma_star)

    def em(self, frame_energy: np.ndarray, theta_init):

        liklihoods = []

        theta = theta_init
        for i in range(self.max_steps):

            liklihoods.append(self.liklihood(frame_energy, theta))
            
            theta_new = self.em_step(frame_energy, theta)

            delta = self.liklihood(frame_energy, theta_new) - self.liklihood(frame_energy, theta)
            if delta < 0:
                print('Warning: log-liklihood decreased')
            if np.abs(delta) < self.convergence_thr:
                print(f'converged on step {i+1}')
                return theta_new, liklihoods
            
            else:
                theta = theta_new

        # If we reach here, converge thr is not met
        print('WARNING: unable to converge')
        return theta, liklihoods


if __name__ == "__main__":

    from multiparty_diarization.datasets.ami_multichannel import AMIMultiChannel
    from multiparty_diarization.eval.metrics import compute_sample_diarization_metrics

    model = EMCloseTalkVAD(speech_thr=0.5, window_size=64, window_offset=32, max_steps=400, convergence_thr=1e-9)

    dset = AMIMultiChannel(
                    array_root_path = '/proj/disney/amicorpus_array',
                    lav_root_path = '/proj/disney/amicorpus_lav',
                    sample_len_s = 20
                    )
    
    idx = 123#np.random.randint(0, len(dset))
    (array_audio, lav_audio), sample, sample_info = dset[idx]

    lav_file = {"waveform": lav_audio, "sample_rate": 16000}
    array_file = {"waveform": array_audio, "sample_rate": 16000}

    preds = model((array_audio, lav_audio))

    metrics = compute_sample_diarization_metrics(sample, preds)

    for metric, val in metrics.items():
        print(f"{metric}: {val:.4f}")

