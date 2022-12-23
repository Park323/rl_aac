from typing import Any
from transformers import AutoTokenizer

class ExtTokenizer():
    def __init__(self, path):
        self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.tokenizer(*args, **kwds)

    def decode(self, *args, **kwds):
        return self.tokenizer.decode(*args, **kwds)

    def batch_decode(self, *args, **kwds):
        return self.tokenizer.batch_decode(*args, **kwds)