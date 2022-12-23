import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class SemanticEvaluator(BaseModel):
    """
    Refer `Diverse Audio Captioning via Adversarial Training`
    We use CNN-extracted feature from pretrained model.
    We just load the already extracted feature from dataset, so pass implementation of CNN extractor for now.
    Only use a Linear layer for reducing the length of the audio embeddings to match that of text.
    """
    def __init__(self, input_dims:int, out_dims:int, vocab_size:int):
        super().__init__()
        self.vocab_size = vocab_size
        """CNN - Temporarily implemented"""
        self.cnn = nn.Linear(
            in_features=32,
            out_features=1
        )
        self.lm = nn.Sequential(
            nn.Linear(self.vocab_size, input_dims),
            nn.GRU(
                input_size=input_dims, hidden_size=out_dims, num_layers=6, 
                batch_first=True, dropout=0.1, bidirectional=True
            ),
        )
        self.lm_fc = nn.Linear(out_dims*2, out_dims)

    def forward(self, audio_inputs, target_sequences):
        """
        Assume audio feature size (B, L, H), text inputs size (B, L)
        and return ((B, L, H), (B, L, H))
        """
        au_embeddings = self.cnn(audio_inputs.permute(0,2,1)).squeeze() # (B, H)

        target_sequences = F.one_hot(target_sequences, num_classes=self.vocab_size).to(audio_inputs.dtype)
        lm_hiddens, h_n = self.lm(target_sequences)
        lm_embeddings = self.lm_fc(lm_hiddens[:, -1, :])
        return au_embeddings, lm_embeddings
