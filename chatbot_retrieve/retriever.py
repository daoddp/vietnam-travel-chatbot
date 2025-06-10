from rank_bm25 import BM25Okapi
from pyvi.ViTokenizer import tokenize
from sentence_transformers import SentenceTransformer
import numpy as np
import string
import pickle

def split_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.lower().split()
    return [word for word in words if len(word.strip()) > 0]

class Retriever:
    def __init__(self, corpus, corpus_emb_path, model_name):
        self.corpus = corpus
        self.tokenized_corpus = [split_text(doc["passage"]) for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        with open(corpus_emb_path, 'rb') as f:
            self.corpus_embs = pickle.load(f)
        
        self.embedder = SentenceTransformer(model_name)

    def retrieve(self, question, topk=50):
        tokenized_query = split_text(question)
        segmented_question = tokenize(question)
        question_emb = self.embedder.encode([segmented_question])
        question_emb /= np.linalg.norm(question_emb, axis=1)[:, np.newaxis]

        bm25_scores = self.bm25.get_scores(tokenized_query)
        semantic_scores = question_emb @ self.corpus_embs.T
        semantic_scores = semantic_scores[0]

        max_bm25_score = max(bm25_scores)
        min_bm25_score = min(bm25_scores)
        normalize = lambda x: (x - min_bm25_score + 0.1) / (max_bm25_score - min_bm25_score + 0.1)

        for i, doc in enumerate(self.corpus):
            doc["bm25_score"] = bm25_scores[i]
            doc["bm25_normed_score"] = normalize(bm25_scores[i])
            doc["semantic_score"] = semantic_scores[i]
            doc["combined_score"] = doc["bm25_normed_score"] * 0.4 + doc["semantic_score"] * 0.6

        return sorted(self.corpus, key=lambda x: x["combined_score"], reverse=True)[:topk]
