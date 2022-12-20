from pathlib import Path
from . import dataset as Dataset
from base import BaseDataLoader


class ClothoDataLoader(BaseDataLoader):
    """
    Clotho dataloader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, optional_dir=None, **kwargs):
        self.data_dir = Path(data_dir)
        self.optional_dir = optional_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        for att in ['tokenizer','max_audio_len', 'max_token_len', 'input_field', 'output_field']:
            assert att in kwargs.keys(), f"{att} is not defined."
        self.kwargs = kwargs

        self.dataset = Dataset.ClothoDataset(cfg=self.kwargs, data_dir=self.data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    def split_validation(self):
        assert self.optional_dir, "Cannot split validation set!"
        return ClothoDataLoader(self.optional_dir, self.batch_size, num_workers=self.num_workers, shuffle=False, training=False, **self.kwargs)
