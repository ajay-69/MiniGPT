import torch
from torch.utils.data import Dataset
from transformers import GPT2TokenizerFast
class GPTDataset(Dataset):
    def __init__(self, text, max_seq_len):
        self.max_seq_len = max_seq_len
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokens = self.tokenizer.encode(text, add_special_tokens=False)

    def __len__(self):
        return (len(self.tokens) - 1)//self.max_seq_len 
        #(tells PyTorch how many complete training examples exist.)

    def __getitem__ (self, idx):
        start = idx*self.max_seq_len
        end = start + self.max_seq_len

        input_ids = self.tokens[start:end]
        target_ids = self.tokens[start+1 : end+1]

        return(
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long)
        )