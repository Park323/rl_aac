import string
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class ClothoDataset(Dataset):
    """
    Load Clotho data
    """
    def __init__(self, cfg, data_dir, tokenizer=None):
        super().__init__()
        self.examples = sorted(data_dir.iterdir())
        
        self.max_audio_len = cfg['max_audio_len']
        self.max_token_len = cfg['max_token_len']
        self.input_name = cfg['input_field']
        self.output_name = cfg['output_field']

        if tokenizer:
            self.tokenizer = tokenizer
        else:
            print("Initialized Tokenizer in ClothoDataset Module")
            self.tokenizer = AutoTokenizer.from_pretrained(cfg['tokenizer'], use_fast=True)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        ex = None
        captions = list()
        for i in range(5): # Assume the number of labels for each audio sample := 5
            fpath = self.examples[(item//5)*5 + i]
            _ex = np.load(str(fpath), allow_pickle=True)
            captions.append(_ex[self.output_name].item())
            if item % 5 == i: 
                ex = _ex   
                fname = Path(fpath).name
        
        # ----- Labels/Decoder inputs -----
        
        tok_e = {'input_ids': None, 'attention_mask': None}
        for ou_e in captions:
            ou_e = ou_e.translate(str.maketrans('', '', string.punctuation))
            ou_e = ou_e.lower()
            
            _tok_e = self.tokenizer(ou_e, max_length=self.max_token_len, return_tensors='pt', padding='max_length')
            if _tok_e['input_ids'].size(1) > self.max_token_len:
                print('Found caption longer than max_token_len parameter ({} tokens).'.format(_tok_e['input_ids'].size(1)))
                _tok_e['input_ids'] = _tok_e['input_ids'][:,:self.max_token_len]
                _tok_e['attention_mask'] = _tok_e['attention_mask'][:,:self.max_token_len]
            tok_e['input_ids'] = torch.cat([tok_e['input_ids'],_tok_e['input_ids']], dim=0) if tok_e['input_ids'] is not None else _tok_e['input_ids']
            tok_e['attention_mask'] = torch.cat([tok_e['attention_mask'],_tok_e['attention_mask']], dim=0) if tok_e['attention_mask'] is not None else _tok_e['attention_mask']
            

        # ou_e = ex[self.output_name].item()
        
        # if ou_e is not None:
        #     ou_e = ou_e.translate(str.maketrans('', '', string.punctuation))
        #     ou_e = ou_e.lower()
            
        #     tok_e = self.tokenizer(ou_e, max_length=self.max_token_len, return_tensors='pt', padding='max_length')
        #     if tok_e['input_ids'].size(1) > self.max_token_len:
        #         print('Found caption longer than max_token_len parameter ({} tokens).'.format(tok_e['input_ids'].size(1)))
        #         tok_e['input_ids'] = tok_e['input_ids'][:,:self.max_token_len]
        #         tok_e['attention_mask'] = tok_e['attention_mask'][:,:self.max_token_len]
        # else:
        #     tok_e = {'input_ids': None, 'attention_mask': None}
        
        # ----- Audio conditioning -----
        in_e = ex[self.input_name].item()
        
        in_e = torch.Tensor(in_e).float().unsqueeze(0)
        
        in_e = in_e.squeeze()
        if len(list(in_e.size())) == 1: # Single embedding in sequence
            in_e = in_e.unsqueeze(0)
        
        # ----- Reformat audio inputs -----
        audio_att_mask = torch.zeros((self.max_audio_len,)).long()
        
        audio_att_mask[:in_e.size(0)] = 1
        if in_e.size(0) > self.max_audio_len:
            in_e = in_e[:self.max_audio_len, :]
        elif in_e.size(0) < self.max_audio_len:
            in_e = torch.cat([in_e, torch.zeros(self.max_audio_len - in_e.size(0), in_e.size(1)).float()])
        
        return {'audio_features': in_e,
                'attention_mask': audio_att_mask,
                'decoder_attention_mask': tok_e['attention_mask'].squeeze() if tok_e['attention_mask'] is not None else None,
                'file_name': ex['file_name'].item(),
                'labels': tok_e['input_ids'].squeeze().long() if tok_e['input_ids'] is not None else None}
