from flask import Flask, render_template, request
import faiss
import numpy as np
from transformers import AutoTokenizer
from biencoder.sentence_bert import SentenceBert, encode
from reranker.reranker import CrossEncoderBert, get_1st_rank

device = "cpu"

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = SentenceBert(device=device)
model_location = "static/weights/sentence_bert_biencoder"
model.bert_model = model.bert_model.from_pretrained(model_location)
model.to(device)

base = np.load("static/weights/faiss_base_tokens.npy")
homer_vocab = np.load("static/weights/faiss_base_originals.npy")
index = faiss.IndexFlatL2(base.shape[1])
index.add(base)

ce_model = CrossEncoderBert().to(device)
ce_model.bert_model.from_pretrained("static/weights/ce_bert")

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('base.html')

@app.route("/get")
def get_homer_response():
    query = request.args.get('msg') 
    pooled_embeds = encode(query, model.bert_tokenizer, model.bert_model, device)
    pooled_embeds = pooled_embeds.cpu().detach().numpy()
    D, I = index.search(pooled_embeds, 10)
    candidates = [homer_vocab[i] for i in I[0]]
    return get_1st_rank(tokenizer, ce_model, query, candidates, device=device)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')