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
        vocab_size:int,
        num_layers:int = None,
        num_heads:int = None,
        ff_factor:int = None,
        max_seq_len:int = None,
        randomness:bool =False,
    ):
        super().__init__()
        # self.random_vector = None if randomness else torch.randn(input_dim)
        self.random_dim = 0
        self.max_seq_len = max_seq_len
        self.policy = TransformerPolicy(
            vocab_size=vocab_size, input_dims=input_dim + self.random_dim, # input dimension + random vector dimension
            output_dims=vocab_size, num_layers=num_layers, 
            num_heads=num_heads, ff_factor=ff_factor)

    def forward(self, inputs, input_masks, test=False):
        """
        inputs : (Batch, Length, Dimension)
        """
        if test:
            return self.infer(inputs, input_masks)

        B, L, D = inputs.size()

        # if self.random_vector is not None:
        #     random_vector = self.random_vector.repeat(B, L, 1) # Use fixed vector for generation.
        # else:
        #     random_vector = torch.randn(B, L, 1)
        # randomized_inputs = torch.cat([inputs, random_vector.to(inputs.device)], dim=-1)
        randomized_inputs = inputs

        logp = torch.full((inputs.size(0), ), 0., device=inputs.device)
        pred_seq = torch.full((self.max_seq_len, inputs.size(0)), 0, device=inputs.device) # Assume sos token's ID : 0
        terminate = torch.full((inputs.size(0), ), False, dtype=bool, device=inputs.device)
        for index in range(self.max_seq_len - 1):
            action, _logp = self.policy.get_action_and_logp(randomized_inputs.permute(1,0,2), pred_seq[:index + 1], input_masks)
            pred_seq[index + 1] = action
            logp += _logp
            terminate[~terminate] |= action[~terminate] == 1 # Assume eos token's ID : 1
            if terminate.sum() == inputs.size(0): break
        return pred_seq.permute(1,0), logp
        
    @torch.no_grad()
    def infer(self, inputs, input_masks):
        """
        inputs : (Batch, Length, Dimension)
        """
        B, L, D = inputs.size()

        # if self.random_vector:
        #     random_vector = self.random_vector.repeat(B, L, 1) # Use fixed vector for generation.
        # else:
        #     random_vector = torch.randn(B, L, 1)
        # randomized_inputs = torch.cat([inputs, random_vector], dim=-1)
        randomized_inputs = inputs

        terminate = torch.full((inputs.size(0), ), False, dtype=bool, device=inputs.device)
        pred_seq = torch.full((self.max_seq_len, inputs.size(0)), 0, device=inputs.device) # Assume sos token's ID : 0
        for index in range(self.max_seq_len - 1):
            action = self.policy.act(randomized_inputs.permute(1,0,2), pred_seq[:index+1], input_masks, greedy=True)
            pred_seq[index + 1] = action # torch.cat([pred_seq, action.unsqueeze(0)], dim=0)
            terminate[terminate] |= action[terminate] == 1 # Assume eos token's ID : 1
            if terminate.sum() == inputs.size(0): break
        
        return pred_seq.permute(1,0)