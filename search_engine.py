import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from nltk.stem import PorterStemmer


DEFAULT_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "so", "for", "of", "on",
    "in", "to", "from", "by", "with", "as", "at", "is", "are", "was", "were", "be", "been",
    "being", "this", "that", "these", "those", "it", "its", "into", "about", "after", "before",
    "over", "under", "between", "during", "through", "out", "up", "down", "off", "not", "no",
    "can", "could", "should", "would", "will", "just", "very", "more", "most", "some", "such",
    "you", "your", "we", "our", "they", "their", "he", "she", "his", "her", "them", "i", "me",
    "my", "mine", "do", "does", "did", "done", "have", "has", "had", "having"
}


class MiniSearchEngine:
    def __init__(self, corpus_path="corpus.json", k1=1.5, b=0.75):
        self.corpus_path = Path(corpus_path)
        self.k1 = k1
        self.b = b
        self.stemmer = PorterStemmer()
        self.documents = []
        self.inverted_index = defaultdict(list)
        self.doc_term_freqs = {}
        self.doc_lengths = {}
        self.avg_doc_length = 0.0
        self.vocabulary = set()
        self._load_corpus()
        self._build_index()

    def _load_corpus(self):
        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {self.corpus_path}")

        with open(self.corpus_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("corpus.json must contain a list of documents")

        validated = []
        for i, doc in enumerate(data):
            if not isinstance(doc, dict):
                continue
            text = str(doc.get("text", "")).strip()
            title = str(doc.get("title", f"Document {i+1}")).strip()
            source = str(doc.get("source", "Unknown source")).strip()
            doc_id = str(doc.get("id", f"doc{i+1}")).strip()
            category = str(doc.get("category", "general")).strip()
            if len(text.split()) < 10:
                continue
            validated.append({
                "id": doc_id,
                "title": title,
                "source": source,
                "category": category,
                "text": text
            })

        self.documents = validated

    def preprocess_text(self, text):
        text = text.lower()
        tokens = re.findall(r"[a-z0-9]+", text)
        filtered = [self.stemmer.stem(tok) for tok in tokens if tok not in DEFAULT_STOPWORDS]
        return filtered

    def _build_index(self):
        total_length = 0

        for idx, doc in enumerate(self.documents):
            tokens = self.preprocess_text(doc["text"])
            total_length += len(tokens)
            self.doc_lengths[idx] = len(tokens)
            term_freq = Counter(tokens)
            self.doc_term_freqs[idx] = term_freq

            for term, freq in term_freq.items():
                self.inverted_index[term].append((idx, freq))
                self.vocabulary.add(term)

        self.avg_doc_length = total_length / len(self.documents) if self.documents else 0.0

    def bm25_score(self, query_terms, doc_idx):
        score = 0.0
        N = len(self.documents)
        doc_len = self.doc_lengths.get(doc_idx, 0)

        for term in query_terms:
            postings = self.inverted_index.get(term, [])
            if not postings:
                continue

            df = len(postings)
            tf = self.doc_term_freqs[doc_idx].get(term, 0)
            if tf == 0:
                continue

            idf = math.log(1 + ((N - df + 0.5) / (df + 0.5)))
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_length if self.avg_doc_length else 0))
            score += idf * (numerator / denominator)

        return score

    def search(self, query, top_k=10):
        start = time.perf_counter()
        query_terms = self.preprocess_text(query)

        if not query_terms:
            return {
                "query": query,
                "results": [],
                "stats": self.get_stats(search_time_ms=0.0)
            }

        candidate_docs = set()
        for term in query_terms:
            for doc_idx, _ in self.inverted_index.get(term, []):
                candidate_docs.add(doc_idx)

        scored_results = []
        for doc_idx in candidate_docs:
            score = self.bm25_score(query_terms, doc_idx)
            if score > 0:
                doc = self.documents[doc_idx]
                scored_results.append({
                    "doc_id": doc["id"],
                    "title": doc["title"],
                    "source": doc["source"],
                    "category": doc.get("category", "general"),
                    "text": doc["text"],
                    "score": round(score, 4)
                })

        scored_results.sort(key=lambda x: x["score"], reverse=True)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 3)

        return {
            "query": query,
            "results": scored_results[:top_k],
            "stats": self.get_stats(search_time_ms=elapsed_ms)
        }

    def get_stats(self, search_time_ms=0.0):
        return {
            "total_documents": len(self.documents),
            "vocabulary_size": len(self.vocabulary),
            "avg_doc_length": round(self.avg_doc_length, 2),
            "search_time_ms": search_time_ms
        }
