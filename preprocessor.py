import nltk
from functools import lru_cache
from nltk.corpus import stopwords


class Preprocessor:
    def __init__(self):
        # Stemming is the most time-consuming part of the indexing process, we attach a lru_cache to the stermmer
        # which will store upto 100000 stemmed forms and reuse them when possible instead of applying the
        # stemming algorithm.
        self.stem = lru_cache(maxsize=10000)(nltk.SnowballStemmer('english').stem)
        self.tokenize = nltk.tokenize.RegexpTokenizer('\w+|\$[\d\.]+|\S+ ').tokenize

    def __call__(self, text):
        stop_words = set(stopwords.words('english'))
        tokens = nltk.RegexpTokenizer('\w+|\$[\d\.]+|\S+ ').tokenize(text)
        tokens =[token for token in tokens if token.isalpha()]  # Evaluate if it is an alpha
        tokens=[token for token in tokens if not token in stop_words]  # remove stopwords
        tokens = [self.stem(token)for token in tokens]
        print(tokens)
        # tokens = [nltk.stem.WordNetLemmatizer().lemmatize(token) for token in tokens]
        return tokens
