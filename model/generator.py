import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from .policy import TransformerPolicy

class CaptionGenerator(BaseModel):
    """
    Refer `Diverse Audio Captioning via Adversarial Training`
    """
    def __init__(
        self,
        input_dim:int, 
        randomness=False
    ):
        super().__init__()
        self.random_vector = None if randomness else torch.randn(input_dim)
        self.policy = TransformerPolicy()

    def forward(self, inputs):
        """
        inputs : (Batch, Length, Dimension)
        """
        B, L, D = inputs.size()

        if self.random_vector:
            random_vector = self.random_vector.repeat(B, L, 1) # Use fixed vector for generation.
        else:
            random_vector = torch.randn(B, L, 1)
        randomized_inputs = torch.cat([inputs, random_vector], dim=-1)

        pred_seq, logp = self.policy.get_action_and_logp(randomized_inputs)
        
        return pred_seq, logp