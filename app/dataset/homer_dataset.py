import pandas as pd
import torch
from torch.utils.data import Dataset

class HomerDataset(Dataset):
    def __init__(self, raw_df: pd.DataFrame, tokenizer):
        self.raw_df = raw_df
        self.tokenizer = tokenizer
        self._init_data()

    def _init_data(self) -> None:
        context = list()
        self.dataset = list()
        # Предподготовка данных
        for _, row in self.raw_df.iterrows():
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
                self.dataset.append(item)
                context.append('UNK' if not isinstance(phrase, str) else phrase)
            else:
                context.append('UNK' if not isinstance(phrase, str) else phrase)
        # Токенизация
        MAX_LENGTH = 128
        self.tokenized_prompts = self.tokenizer([data["prompt"] for data in self.dataset],
                                    max_length=MAX_LENGTH,
                                    padding="max_length",
                                    truncation=True,
                                    verbose=True
                                    )
        self.tokenized_responses = self.tokenizer([data["response"] for data in self.dataset],
                                        max_length=MAX_LENGTH,
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