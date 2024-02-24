import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

class CrossEncoderBert(torch.nn.Module):
    def __init__(self, max_length: int = 128):
        super().__init__()
        self.max_length = max_length
        self.bert_model = AutoModel.from_pretrained('distilbert-base-uncased')
        self.bert_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.linear = torch.nn.Linear(self.bert_model.config.hidden_size, 1) # Предсказывать будет метку: оценка от 1 до 5

    def forward(self, input_ids, attention_mask):
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        # (seq_len, bert_dim)
        pooled_output = outputs.last_hidden_state[:,0] # Use the CLS token`s output (в нем весь текст)
        return self.linear(pooled_output)
    
def test_cross_encoder_bert():
    max_length = 128
    model = CrossEncoderBert(max_length=max_length)

    # Check __init__ method
    assert model.max_length == max_length, "Incorrect max_length initialization"
    assert isinstance(model.linear, torch.nn.Linear), "linear is not an instance of torch.nn.Linear"
    assert model.linear.in_features == model.bert_model.config.hidden_size, "Incorrect input size for linear layer"

    # Prepare dummy data for forward method check
    input_text = ["Hello, world!"]
    inputs = model.bert_tokenizer(input_text, return_tensors="pt", max_length=max_length, truncation=True, padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Check forward method
    output = model(input_ids, attention_mask)
    assert output.shape == (1, 1), "Output shape is incorrect"
    
def train_step_fn(model, optimizer, scheduler, loss_fn, batch, device):
    model.train()
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    optimizer.zero_grad()
    logits = model(input_ids, attention_mask)
    loss = loss_fn(logits.squeeze(-1), labels)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.item()

def val_step_fn(model, loss_fn, batch, device):
    model.eval()
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
    loss = loss_fn(logits.squeeze(-1), labels)
    return loss.item()

def mini_batch(model, dataloader, optimizer, scheduler, loss_fn, step_fn, batch_size, device, is_training=True):
    mini_batch_losses = []
    for i, batch in enumerate(dataloader):
        if is_training:
            loss = step_fn(model, optimizer, scheduler, loss_fn, batch, device)
        else:
            loss = step_fn(model, loss_fn, batch, device)
        mini_batch_losses.append(loss)
        if i % (batch_size * 4) == 0:
            print(f"Step {i:>5}/{len(dataloader)}, Loss = {loss:.3f}")
    return np.mean(mini_batch_losses), mini_batch_losses

def get_ranked_docs(
    tokenizer: AutoTokenizer, finetuned_ce: CrossEncoderBert,
    query: str, corpus: list[str],
    device, MAX_LENGTH=128, 
) -> None:

    queries = [query] * len(corpus)
    tokenized_texts = tokenizer(
        queries, corpus, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt"
    ).to(device)

    # Finetuned CrossEncoder model scoring
    with torch.no_grad():
        ce_scores = finetuned_ce(tokenized_texts['input_ids'], tokenized_texts['attention_mask']).squeeze(-1)
        ce_scores = torch.sigmoid(ce_scores)  # Apply sigmoid if needed

    # Process scores for finetuned model
    print(f"Query - {query} [Finetuned Cross-Encoder]\n---")
    scores = ce_scores.cpu().numpy()
    scores_ix = np.argsort(scores)[::-1]
    for ix in scores_ix:  # Limit to corpus size
        print(f"{scores[ix]: >.2f}\t{corpus[ix]}")
        
def get_1st_rank(
    tokenizer: AutoTokenizer, finetuned_ce: CrossEncoderBert,
    query: str, corpus: list[str],
    device, MAX_LENGTH=128, 
) -> None:

    queries = [query] * len(corpus)
    tokenized_texts = tokenizer(
        queries, corpus, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt"
    ).to(device)

    # Finetuned CrossEncoder model scoring
    with torch.no_grad():
        ce_scores = finetuned_ce(tokenized_texts['input_ids'], tokenized_texts['attention_mask']).squeeze(-1)
        ce_scores = torch.sigmoid(ce_scores)  # Apply sigmoid if needed

    scores = ce_scores.cpu().numpy()
    scores_ix = np.argsort(scores)[::-1]
    ix = scores_ix[0]
    return corpus[ix]
