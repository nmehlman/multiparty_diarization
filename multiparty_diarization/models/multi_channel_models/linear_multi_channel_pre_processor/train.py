from pyannote.database import registry
from pyannote.database import FileFinder

from multiparty_diarization.multi_channel_models.pyannote_multi_channel.utils import MultiChannelSpeakerDiarization
from multiparty_diarization.multi_channel_models.linear_multi_channel_pre_processor.pyannote_preprocessor import PyanNetPreprocessor
from multiparty_diarization.utils import load_configs

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

if __name__ == "__main__":

    cfg_path = "./configs/pyannote_preprocessor.yaml"

    config = load_configs(cfg_path)

    registry.load_database(config['database_cfg'])

    protocol = registry.get_protocol(config['protocol'],  preprocessors={"audio": FileFinder()})
    task = MultiChannelSpeakerDiarization(protocol=protocol, **config['task'])

    logger = TensorBoardLogger(**config["tensorboard"])
    model = PyanNetPreprocessor(task=task, **config['model'])

    trainer = pl.Trainer(logger=logger, **config['trainer'])
    
    trainer.fit(model)