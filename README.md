# Disney Project Multiparty Diarization

Code for benchmarking open-source diarization models on multiparty datasets.

## Conventions
- All models should be wrapped in a `DiarizationModel` instance to support uniform interface. The `__call__` method should accept a 2D (channels, samples) waveform pytorch tensor and return a list of tuples `[(speaker, start, end), ...]`

## Tasks

1. Test Pyannote model on AMI dataset @Nick
2. Setup NEMO diarization model @Anfeng
3. Look into additional open-source dataset (e.g. Diehard, Dinney party)