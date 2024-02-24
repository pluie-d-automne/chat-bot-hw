import pandas as pd
import torch
from torch.utils.data import Dataset

def prepare_data(raw_df: pd.DataFrame) -> list:
    context = list()
    dataset = list()
    for _, row in raw_df.iterrows():
        charachter, phrase = row['raw_character_text'], row['spoken_words']
        if not isinstance(charachter, str):
            context = list()
        elif len(context)>0:
            item = {
                'prompt': context[-1],
                'response': 'UNK' if not isinstance(phrase, str) else phrase,
                'context': context[:-2] if len(context) < 10 else context[-10 :-2],
                'label': 1 if charachter == 'Homer Simpson' else 0
            }
            dataset.append(item)
            if len(context[-1].split(".?!")) > 1:
                item_2 = {
                    'prompt': context[-1].split(".?!")[-1],
                    'response': 'UNK' if not isinstance(phrase, str) else phrase.split(".?!")[0],
                    'context': context[:-2] if len(context) < 10 else context[-10 :-2],
                    'label': 1 if charachter == 'Homer Simpson' else 0
                }
                dataset.append(item_2)
            context.append('UNK' if not isinstance(phrase, str) else phrase)
        else:
            context.append('UNK' if not isinstance(phrase, str) else phrase)
    return dataset

class HomerDataset(Dataset):
    def __init__(self, dataset: list, tokenizer, MAX_LENGTH = 128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.MAX_LENGTH = MAX_LENGTH
        self._init_data()

    def _init_data(self) -> None:
        # Токенизация
        
        self.tokenized_prompts = self.tokenizer([data["prompt"] for data in self.dataset],
                                    max_length=self.MAX_LENGTH,
                                    padding="max_length",
                                    truncation=True,
                                    verbose=True
                                    )
        self.tokenized_responses = self.tokenizer([data["response"] for data in self.dataset],
                                        max_length=self.MAX_LENGTH,
                                        padding="max_length",
                                        truncation=True,
                                        verbose=True
                                )
        self.labels = (data["label"] for data in self.dataset)
        # Датасет для тернировки
        self.data = []
        for pt_ids, pt_am, rt_ids, rt_am, label in zip(
            self.tokenized_prompts["input_ids"],
            self.tokenized_prompts["attention_mask"],
            self.tokenized_responses["input_ids"],
            self.tokenized_responses["attention_mask"],
            self.labels
        ):
            data = {}
            data["prompt_input_ids"]  = torch.tensor(pt_ids, dtype=torch.long)
            data["prompt_attention_mask"] = torch.tensor(pt_am, dtype=torch.long)
            data["response_input_ids"] = torch.tensor(rt_ids, dtype=torch.long)
            data["response_attention_mask"] = torch.tensor(rt_am, dtype=torch.long)
            data["label"] = torch.tensor(label, dtype=torch.long) # метка класса
            self.data.append(data)

    def __getitem__(self, ix: int) -> dict[str, torch.tensor]:
        return self.data[ix]

    def __len__(self) -> int:
        return len(self.data)
    
    
class CEDataset(Dataset):
    def __init__(self, dataset: list, tokenizer, MAX_LENGTH=128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.tokens = self.tokenizer([data["prompt"] for data in dataset],
                            [data["response"] for data in dataset],
                            max_length=MAX_LENGTH,
                            padding="max_length",
                            truncation=True,
                            verbose=True)
        self.labels = [data["label"] for data in dataset]
        
    def __getitem__(self, ix: int) -> dict[str, torch.tensor]:
        return {
            "input_ids": torch.tensor(self.tokens["input_ids"][ix], dtype=torch.long),
            "attention_mask": torch.tensor(self.tokens["attention_mask"][ix], dtype=torch.long),
            "labels": torch.tensor(self.labels[ix], dtype=torch.float)
        }

    def __len__(self) -> int:
        return len(self.tokens["input_ids"])
    