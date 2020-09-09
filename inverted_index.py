from collections import defaultdict
import os
import pickle
import nltk


def get_zero():
    return 0


def get_empty_postings():
    return defaultdict(get_zero)


class SparseMatrix:
    """
    Used to represent a frequency count matrix.
    token_to_doc_counts maps a token (row) to a dict which maps a doc (column) to a count.
    doc_to_token_counts maps a doc (column) to a dict which maps a token (row) to a count.
    Both of these dicts contain the same data, they just allow for different accessing methods (rows vs columns).
    """
    def __init__(self):
        self.token_to_doc_counts = defaultdict(get_empty_postings)
        self.doc_to_token_counts = defaultdict(get_empty_postings)
        self.num_docs = 0


class InvertedIndex:
    """
    Handles reading raw text files into inverted index form,
    as well as running queries over the created inverted index.
    """
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.similarity_measure = None
        self.postings = SparseMatrix()

    def index_directory(self, directory, use_stored_index=True):
        """
        Grab every file inside directory and add them to the index.
        After indexing is finished, the created SparseMatrix is written to a .pkl file.
        If use_stored_index=True and a .pkl file exists for this directory then the inverted index
        will be read from the file instead of running the indexing process again.
        """
        store_file = f'{directory}_inverted_index.pkl'
        if use_stored_index and os.path.exists(store_file):
            print(f'Loading index from {store_file}.')
            with open(store_file, 'rb') as f:
                self.postings = pickle.load(f)
        else:
            for path, subdirs, files in os.walk(directory):
                print(f'Indexing dir: {path}')
                for file in files:
                    with open(os.path.join(path, file), 'r', encoding='utf-8') as fr:
                        self.index_document(file, fr.read())
            with open(store_file, 'wb') as f:
                pickle.dump(self.postings, f)

    def index_document(self, doc, text):
        tokens = self.preprocessor(text)
        for token in tokens:
            self.postings.token_to_doc_counts[token][doc] += 1
            self.postings.doc_to_token_counts[doc][token] += 1
        self.postings.num_docs += 1


    def run_query(self, query, max_results_returned=10):
        """
        :param query: string of text to be queried for.
        :param max_results_returned: the maximum number of documents to return.
        :return: list of pairs of (document, similarity), for the max_results_returned most similar documents.
        """
        query_tokens = self.preprocessor(query)
        query_vector = defaultdict(lambda: 0)
        for token in query_tokens:
            query_vector[token] += 1
        sim_scores = self.similarity_measure(query_vector)
        # TODO sort the results in sim_scores and return only the max_results_returned docs with the highest scores.
        sorted_sim_scores = []
        if sim_scores is not None:
            highest_score = (sorted(sim_scores.values()))[-10:]  # Get top 10 scores
            for document_id in sim_scores:
                if sim_scores[document_id] in highest_score and len(sorted_sim_scores) <= max_results_returned:
                    sorted_sim_scores.append([sim_scores[document_id], document_id])   # Add id and sim-score into list
        else:
            raise IOError
        return reversed(sorted(sorted_sim_scores))          # Reverse the list for next process(rank from high to low)

    def set_similarity(self, sim):
        self.similarity_measure = sim(self.postings)
