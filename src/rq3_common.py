"""Shared RQ3 helpers: one definition of the modelling corpus and of an LDA fit,
used by model selection (05), the ensemble (06), topic analysis (07) and the independent check."""
import logging

import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel

from config import PROCESSED, RQ3_ITERATIONS, RQ3_PASSES

logging.disable(logging.WARNING)
NO_BELOW, NO_ABOVE = 10, 0.5  # vocabulary filter: term in >= 10 documents and <= 50% of documents


def build(corpus_df=None):
    """Return (corpus dataframe, token lists, gensim Dictionary, bag-of-words corpus)."""
    c = pd.read_csv(PROCESSED / "rq3_corpus.csv") if corpus_df is None else corpus_df
    texts = c.tokens.str.split().tolist()
    dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=NO_BELOW, no_above=NO_ABOVE, keep_n=None)
    corpus = [dictionary.doc2bow(t) for t in texts]
    return c, texts, dictionary, corpus


def fit(corpus, dictionary, k, seed, alpha="symmetric", eta="symmetric"):
    return LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=k,
        random_state=seed,
        passes=RQ3_PASSES,
        iterations=RQ3_ITERATIONS,
        alpha=alpha,
        eta=eta,
        chunksize=2000,
        eval_every=None,
    )
