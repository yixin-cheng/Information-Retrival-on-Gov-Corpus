import argparse
import os

from inverted_index import InvertedIndex
from preprocessor import Preprocessor
from similarity_measures import TF_Similarity, TFIDF_Similarity, BM25_Similarity

parser = argparse.ArgumentParser(description='Run all queries on the inverted index.')
parser.add_argument('--new', default=True, help='If True then build a new index from scratch. If False then attempt to'
                                                ' reuse existing index')
parser.add_argument('--sim', default='BM25', help='The type of similarity to use. Should be "TF" or "TFIDF" or "BM25')
args = parser.parse_args()

index = InvertedIndex(Preprocessor())
index.index_directory(os.path.join('gov', 'documents'), use_stored_index=True)

sim_name_to_class = {'TF': TF_Similarity,
                     'TFIDF': TFIDF_Similarity,
                     'BM25': BM25_Similarity}

sim = sim_name_to_class[args.sim]
index.set_similarity(sim)
print(f'Setting similarity to {sim.__name__}')

print()
print('Index ready.')


topics_file = os.path.join('gov', 'topics', 'gov.topics')
runs_file = os.path.join('runs', 'retrieved.runs')

# TODO run queries
tf = open(topics_file, "r")
rank = 0
query_id = str()
for query in tf:
    query_stem = query.lstrip('1234567890 ')   # Remove the first numbers and a space of a query
    for digit in query:                         # extract the query id
        if digit.isdigit():
            query_id += digit
        else:
            break
    for document_id in index.run_query(query_stem):  # the default number of max_result docs is 10
        rf = open(runs_file, "a")
        rf.write('{} {} {} {} {} {}\n'.format(query_id, 'Q0', document_id[1], rank, document_id[0], 'MY_IR_SYSTEM'))
        rank += 1
        rf.close()
    rank = 0
    query_id = str()
tf.close()

"""
You will need to:
    1. Read in the topics_file.
    2. For each line in the topics file create a query string (note each line has both a query_id and query_text,
       you just want to search for the text)  and run this query on index with index.run_query().
    3. Write the results of the query to runs_file IN TREC_EVAL FORMAT
        - Trec eval format requires that each retrieval is on a separate line of the form
          query_id Q0 document_id rank similarity_score MY_IR_SYSTEM
"""
