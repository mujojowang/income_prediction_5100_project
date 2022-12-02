import torch
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, labels, tokenizer):
        self.labels = [labels[label] for label in dataset['category']]
        self.texts = [tokenizer(text, padding='max_length', max_length=90, \
                                truncation=True, return_tensors="pt") \
                      for text in dataset['text']]
    
    def classes(self):
        return self.labels
    
    def __len__(self):
        return len(self.labels)
    
    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])
    
    def get_batch_texts(self, idx):
        return self.texts[idx]
    
    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
      
        return batch_texts, batch_y
