# placeholder for src/dataset.py
from torch.utils.data import Dataset

class SummDataset(Dataset):
    def __init__(self, encoder_input, decoder_input, labels, length):
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.labels = labels
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()}
        item2 = {key: val[idx].clone().detach() for key, val in self.decoder_input.items()}
        item2['decoder_input_ids'] = item2['input_ids']
        item2['decoder_attention_mask'] = item2['attention_mask']
        item2.pop('input_ids')
        item2.pop('attention_mask')
        item.update(item2)
        item['labels'] = self.labels['input_ids'][idx]
        return item
