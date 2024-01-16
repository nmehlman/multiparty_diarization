from torch import Tensor
from torch_audiomentations.utils.object_dict import ObjectDict
from torch_audiomentations import Identity
from pyannote.audio.tasks.segmentation.speaker_diarization import SpeakerDiarization
from pyannote.audio.utils.permutation import permutate

import math
import random
import warnings
from io import IOBase
import torch
from torch.utils.data._utils.collate import default_collate
from pathlib import Path
from typing import Mapping, Optional, Text, Tuple, Union

from pyannote.database.protocol.protocol import Scope, Subset
from pyannote.core import Segment, SlidingWindowFeature

Subsets = list(Subset.__args__)
Scopes = list(Scope.__args__)

import numpy as np
import torch.nn.functional as F
import torchaudio
from pyannote.core import Segment
from torch import Tensor

torchaudio.set_audio_backend("soundfile")

AudioFile = Union[Text, Path, IOBase, Mapping]


class MultiChannelIdentity:
    """Utitlity placeholder augmentation class for multi-channel audio"""
    def __call__(self, samples: Tensor = None, sample_rate: int | None = None, targets: Tensor | None = None, target_rate: int | None = None) -> ObjectDict:
        return ObjectDict({"samples": samples, "sample_rate": sample_rate, "targets": targets, "target_rate": target_rate})
    
    def train(*args, **kwargs):
        pass

class MultiChannelSpeakerDiarization(SpeakerDiarization):
    """Utility task class to permit multi-channel audio in SpeakerDiarization tasks"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.augmentation, Identity):
            self.augmentation = MultiChannelIdentity()


    def collate_X(self, batch) -> torch.Tensor:
        batch_X = torch.stack([b["X"] for b in batch])
        return batch_X


def get_torchaudio_info(file: AudioFile):
    """Protocol preprocessor used to cache output of torchaudio.info

    Args:
        file (AudioFile): The audio file for which information is to be retrieved.

    Returns:
        AudioMetaData: Metadata of the audio file, obtained from torchaudio.info.
    """

    info = torchaudio.info(file["audio"])

    # rewind if needed
    if isinstance(file["audio"], IOBase):
        file["audio"].seek(0)

    return info


class MutiChannelAudio:
    

    PRECISION = 0.001

    @staticmethod
    def power_normalize(waveform: Tensor) -> Tensor:
        """Power-normalize waveform

        Args:
            waveform (Tensor): The waveform(s) to be normalized, of shape (..., time).

        Returns:
            Tensor: The power-normalized waveform(s), of shape (..., time).
        """
        rms = waveform.square().mean(dim=-1, keepdim=True).sqrt()
        return waveform / (rms + 1e-8)

    @staticmethod
    def validate_file(file: AudioFile) -> Mapping:
        """Validate file for use with other Audio methods

        Args:
            file (AudioFile): The audio file to be validated.

        Returns:
            Mapping: The validated file in various formats, depending on the input.

        Raises:
            ValueError: If the file format is not valid or the file does not exist.

        """

        if isinstance(file, Mapping):
            pass

        elif isinstance(file, (str, Path)):
            file = {"audio": str(file), "uri": Path(file).stem}

        elif isinstance(file, IOBase):
            return {"audio": file, "uri": "stream"}

        else:
            raise ValueError

        if "waveform" in file:
            waveform: Union[np.ndarray, Tensor] = file["waveform"]
            if len(waveform.shape) != 2 or waveform.shape[0] > waveform.shape[1]:
                raise ValueError(
                    "'waveform' must be provided as a (channel, time) torch Tensor."
                )

            sample_rate: int = file.get("sample_rate", None)
            if sample_rate is None:
                raise ValueError(
                    "'waveform' must be provided with their 'sample_rate'."
                )

            file.setdefault("uri", "waveform")

        elif "audio" in file:
            if isinstance(file["audio"], IOBase):
                return file

            path = Path(file["audio"])
            if not path.is_file():
                raise ValueError(f"File {path} does not exist")

            file.setdefault("uri", path.stem)

        else:
            raise ValueError(
                "Neither 'waveform' nor 'audio' is available for this file."
            )

        return file

    def __init__(self, sample_rate=None):
        super().__init__()
        self.sample_rate = sample_rate

    def downmix_and_resample(self, waveform: Tensor, sample_rate: int) -> Tensor:
        """Downmix and resample waveform

        Args:
            waveform (Tensor): The waveform to be processed, of shape (channel, time).
            sample_rate (int): The sample rate of the waveform.

        Returns:
            Tuple[Tensor, int]: The downmixed and resampled waveform, along with its new sample rate.
        """
        

        # resample
        if (self.sample_rate is not None) and (self.sample_rate != sample_rate):
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, self.sample_rate
            )
            sample_rate = self.sample_rate

        return waveform, sample_rate

    def get_duration(self, file: AudioFile) -> float:
        """Get audio file duration in seconds

        Args:
            file (AudioFile): The audio file for which to find the duration.

        Returns:
            float: Duration of the audio file in seconds.
        """

        file = self.validate_file(file)

        if "waveform" in file:
            frames = len(file["waveform"].T)
            sample_rate = file["sample_rate"]

        else:
            if "torchaudio.info" in file:
                info = file["torchaudio.info"]
            else:
                info = get_torchaudio_info(file)

            frames = info.num_frames
            sample_rate = info.sample_rate

        return frames / sample_rate

    def get_num_samples(self, duration: float, sample_rate: int = None) -> int:
        """Get the deterministic number of samples from duration and sample rate

        Args:
            duration (float): The duration in seconds.
            sample_rate (int, optional): The sample rate. If not provided, the instance's sample rate is used.

        Returns:
            int: The number of samples corresponding to the given duration and sample rate.

        Raises:
            ValueError: If the sample rate is not provided and not set in the instance.
        """

        sample_rate = sample_rate or self.sample_rate

        if sample_rate is None:
            raise ValueError(
                "`sample_rate` must be provided to compute number of samples."
            )

        return math.floor(duration * sample_rate)

    def __call__(self, file: AudioFile) -> Tuple[Tensor, int]:
        """Obtain waveform and sample rate from an audio file

        Args:
            file (AudioFile): The audio file to be processed.

        Returns:
            Tuple[Tensor, int]: The waveform and its sample rate.
        """

        file = self.validate_file(file)

        if "waveform" in file:
            waveform = file["waveform"]
            sample_rate = file["sample_rate"]

        elif "audio" in file:
            waveform, sample_rate = torchaudio.load(file["audio"])

            # rewind if needed
            if isinstance(file["audio"], IOBase):
                file["audio"].seek(0)

        channel = file.get("channel", None)

        if channel is not None:
            waveform = waveform[channel : channel + 1]

        return self.downmix_and_resample(waveform, sample_rate)

    def crop(
        self,
        file: AudioFile,
        segment: Segment,
        duration: Optional[float] = None,
        mode="raise",
    ) -> Tuple[Tensor, int]:
        
        """Crop a segment from an audio file

        Args:
            file (AudioFile): The audio file to be processed.
            segment (Segment): The temporal segment to load.
            duration (float, optional): Optional duration to override the 'Segment' focus duration.
            mode (str, optional): Specifies how out-of-bounds segments will behave ('raise' or 'pad').

        Returns:
            Tuple[Tensor, int]: The cropped waveform segment and its sample rate.

        Raises:
            ValueError: If the requested segment is out of bounds or if the fixed duration is longer than the file duration.
        """
        
        file = self.validate_file(file)

        if "waveform" in file:
            waveform = file["waveform"]
            frames = waveform.shape[1]
            sample_rate = file["sample_rate"]

        elif "torchaudio.info" in file:
            info = file["torchaudio.info"]
            frames = info.num_frames
            sample_rate = info.sample_rate

        else:
            info = get_torchaudio_info(file)
            frames = info.num_frames
            sample_rate = info.sample_rate

        channel = file.get("channel", None)

        # infer which samples to load from sample rate and requested chunk
        start_frame = math.floor(segment.start * sample_rate)

        if duration:
            num_frames = math.floor(duration * sample_rate)
            end_frame = start_frame + num_frames

        else:
            end_frame = math.floor(segment.end * sample_rate)
            num_frames = end_frame - start_frame

        if mode == "raise":
            if num_frames > frames:
                raise ValueError(
                    f"requested fixed duration ({duration:6f}s, or {num_frames:d} frames) is longer "
                    f"than file duration ({frames / sample_rate:.6f}s, or {frames:d} frames)."
                )

            if end_frame > frames + math.ceil(self.PRECISION * sample_rate):
                raise ValueError(
                    f"requested chunk [{segment.start:.6f}s, {segment.end:.6f}s] (frames #{start_frame:d} to #{end_frame:d}) "
                    f"lies outside of {file.get('uri', 'in-memory')} file bounds [0., {frames / sample_rate:.6f}s] ({frames:d} frames)."
                )
            else:
                end_frame = min(end_frame, frames)
                start_frame = end_frame - num_frames

            if start_frame < 0:
                raise ValueError(
                    f"requested chunk [{segment.start:.6f}s, {segment.end:.6f}s] (frames #{start_frame:d} to #{end_frame:d}) "
                    f"lies outside of {file.get('uri', 'in-memory')} file bounds [0, {frames / sample_rate:.6f}s] ({frames:d} frames)."
                )

        elif mode == "pad":
            pad_start = -min(0, start_frame)
            pad_end = max(end_frame, frames) - frames
            start_frame = max(0, start_frame)
            end_frame = min(end_frame, frames)
            num_frames = end_frame - start_frame

        if "waveform" in file:
            data = file["waveform"][:, start_frame:end_frame]

        else:
            try:
                data, _ = torchaudio.load(
                    file["audio"], frame_offset=start_frame, num_frames=num_frames
                )
                # rewind if needed
                if isinstance(file["audio"], IOBase):
                    file["audio"].seek(0)
            except RuntimeError:
                if isinstance(file["audio"], IOBase):
                    msg = "torchaudio failed to seek-and-read in file-like object."
                    raise RuntimeError(msg)

                msg = (
                    f"torchaudio failed to seek-and-read in {file['audio']}: "
                    f"loading the whole file instead."
                )

                warnings.warn(msg)
                waveform, sample_rate = self.__call__(file)
                data = waveform[:, start_frame:end_frame]

                # storing waveform and sample_rate for next time
                # as it is very likely that seek-and-read will
                # fail again for this particular file
                file["waveform"] = waveform
                file["sample_rate"] = sample_rate

        if channel is not None:
            data = data[channel : channel + 1, :]

        # pad with zeros
        if mode == "pad":
            data = F.pad(data, (pad_start, pad_end))

        return self.downmix_and_resample(data, sample_rate)
