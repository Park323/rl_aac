import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from base import BaseModel


class PolicyNet(BaseModel):
    """
    Reinforcement Learning
    """
    def __init__(self,):
        super().__init__()

    def forward(self, inputs):
        '''abstract method'''
        pass

    def get_action_and_logp(self, x):
        action_prob = self.forward(x)
        m = distributions.Categorical(action_prob)
        action = m.sample()
        logp = m.log_prob(action)
        return action, logp

    def act(self, x):
        action, _ = self.get_action_and_logp(x)
        return action


class TransformerPolicy(PolicyNet):
    def __init__(self, vocab_size:int, input_dims:int, output_dims:int, num_layers:int, num_heads:int, ff_factor:int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedder = nn.Linear(self.vocab_size, input_dims)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=input_dims, nhead=num_heads, dim_feedforward=ff_factor*input_dims
            ), num_layers
        )
        self.fc = nn.Linear(input_dims, output_dims)

    def forward(self, input_features, target_sequence, input_masks, target_masks):
        """
        input : (S, N, E)
        target : (T, N, E)
        input_masks : (S, S)
        target_masks : (T, T)
        """
        target_sequence = self.embedder(F.one_hot(target_sequence, num_classes=self.vocab_size).to(input_features.dtype))
        # Need for validate masks' value range.
        tgt_mask = self.generate_square_subsequent_mask(target_sequence.size(0)).to(input_features.device)

        outputs = self.decoder(
            target_sequence, 
            input_features,
            tgt_mask = tgt_mask,
            tgt_key_padding_mask = target_masks, 
            memory_key_padding_mask = input_masks)
        outputs = self.fc(outputs)
        return outputs

    def _forward_step(self, input_features, target_sequence, input_masks=None):
        """
        input : (S, N, E)
        target : (T, N, E)
        input_masks : (N, S) := key padding mask
        """
        target_sequence = self.embedder(F.one_hot(target_sequence, num_classes=self.vocab_size).to(input_features.dtype))
        
        outputs = self.decoder(
            target_sequence, 
            input_features,
            memory_key_padding_mask = input_masks)
        outputs = self.fc(outputs[-1])
        return outputs.softmax(-1)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.full((sz, sz),-float('inf'))
        mask = torch.triu(mask, diagonal=1)
        return mask

    def get_action_and_logp(self, input_features, target_sequence, input_masks):
        action_prob = self._forward_step(input_features, target_sequence, input_masks)
        m = distributions.Categorical(action_prob)
        action = m.sample()
        logp = m.log_prob(action)
        return action, logp

    def act(self, input_features, target_sequence, input_masks, greedy=False):
        if greedy:
            action_prob = self._forward_step(input_features, target_sequence, input_masks)
            action = action_prob.argmax(dim=-1)
        else:
            action, _ = self.get_action_and_logp(input_features, target_sequence, input_masks)
        return action