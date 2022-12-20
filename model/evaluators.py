import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class SemanticEvaluator(BaseModel):
    """
    Refer `Diverse Audio Captioning via Adversarial Training`
    We use CNN-extracted feature from pretrained model.
    We just load the already extracted feature from dataset, so pass implementation of CNN extractor for now.
    """
    def __init__(self, input_dims:int, out_dims:int):
        # self.cnn = None
        self.lm = nn.GRU(
            input_size=input_dims, hidden_size=out_dims, num_layers=1, 
            batch_first=True, dropout=0.1, bidirectional=True
        )

    def forward(self, audio_inputs, target_sequences):
        """
        Assume audio feature size (B, L, H), text inputs size (B, L)
        and return ((B, L, H), (B, L, H))
        """
        # au_embeddings = self.cnn(audio_inputs)
        au_embeddings = audio_inputs
        lm_embeddings, h_n = self.lm(target_sequences)
        return au_embeddings, lm_embeddings
