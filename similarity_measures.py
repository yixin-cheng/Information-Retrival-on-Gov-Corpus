from abc import abstractmethod
from collections import defaultdict
from math import log, sqrt


class CosineSimilarity:
    """
    This class calculates a similarity score between a given query and all documents in an inverted index.
    """

    def __init__(self, postings):
        self.postings = postings
        self.doc_to_norm = dict()
        self.set_document_norms()

    def __call__(self, query):
        doc_to_score = defaultdict(lambda: 0)
        self.get_scores(doc_to_score, query)
        return doc_to_score

    @abstractmethod
    def set_document_norms(self):
        """
        Set self.doc_to_norm to contain the norms of every document.
        """
        pass

    @abstractmethod
    def get_scores(self, doc_to_score, query):
        """
        For each document add an entry to doc_to_score with this document's similarity to query.
        """
        pass


class TF_Similarity(CosineSimilarity):
    def set_document_norms(self):
        for doc, token_counts in self.postings.doc_to_token_counts.items():
            self.doc_to_norm[doc] = sqrt(sum([tf ** 2 for tf in token_counts.values()]))

    def get_scores(self, doc_to_score, query):
        for token, query_term_frequency in query.items():
            for doc, document_term_frequency in self.postings.token_to_doc_counts[token].items():
                doc_to_score[doc] += query_term_frequency * document_term_frequency / self.doc_to_norm[doc]


class TFIDF_Similarity(CosineSimilarity):
    def set_document_norms(self):
        N = len(self.postings.doc_to_token_counts)
        df = defaultdict(lambda: 0)
        for token, doc_counts in self.postings.token_to_doc_counts.items():
            for document_term_frequency in doc_counts.values():  # To get the df value
                if document_term_frequency > 0:
                    df[token] += 1
        for doc, token_counts in self.postings.doc_to_token_counts.items():
            res = 0
            for token, counts in token_counts.items():
                res += (counts * log(N / df[token])) ** 2
            self.doc_to_norm[doc] = sqrt(res)

    def get_scores(self, doc_to_score, query):
        N = len(self.postings.doc_to_token_counts)
        df = 0
        for token, query_term_frequency in query.items():
            for document_term_frequency in self.postings.token_to_doc_counts[token].values():  # To get the df value
                if document_term_frequency > 0:
                    df += 1
            for doc, counts in self.postings.token_to_doc_counts[token].items():
                if counts > 0:
                    doc_to_score[doc] += query_term_frequency * counts * (log(N / df) ** 2) / self.doc_to_norm[doc]
            df = 0


class BM25_Similarity(CosineSimilarity):
    def set_document_norms(self):
        pass

    def get_scores(self, doc_to_score, query):
        N = len(self.postings.doc_to_token_counts)
        df = 0
        sumdl=0
        for doc, token_counts in self.postings.doc_to_token_counts.items():
            sumdl += sum(token_counts.values())
        for token, query_term_frequency in query.items():
            for document_term_frequency in self.postings.token_to_doc_counts[token].values():  # To get the df value
                if document_term_frequency > 0:
                    df += 1
            for doc, counts in self.postings.token_to_doc_counts[token].items():
                k1 = 4
                b = 0.75
                avgdl = sumdl/ N
                idf = log((N - df + 0.5) / (df + 0.5))  # get idf
                d = sum(self.postings.doc_to_token_counts[doc].values())
                if counts > 0:
                    doc_to_score[doc] += idf * (counts * (k1 + 1) / (counts + k1 * (1 - b + b * d / avgdl)))   # As formula given
            df = 0
