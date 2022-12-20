import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class SingleLayerGRU(BaseModel):
    """
    Refer `Diverse Audio Captioning via Adversarial Training`
    Use final output from GRU layer to predict wether is human-annotated or machin-captioned.
    """
    def __init__(self, input_dims:int, out_dims:int):
        self.discriminator = nn.GRU(
            input_size=input_dims, hidden_size=out_dims, num_layers=1, 
            batch_first=True, dropout=0.1, bidirectional=True
        )
        self.reducer = nn.Linear(out_dims * 2, 1)

    def forward(self, inputs):
        """
        Assume Input size (B, L, H)
        and return (B,)
        """
        outputs, h_n = self.discriminator(inputs)
        cls_outputs = outputs[:, -1, :]
        cls_outputs = self.reducer(cls_outputs)
        probs = F.sigmoid(cls_outputs)
        return probs.view(-1)