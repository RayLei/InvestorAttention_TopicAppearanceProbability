#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 17:17:00 2020

The script calculates the C_V coherence scores for all topics.

@author: LEIHAO
"""

from os import listdir
import sys
sys.path.insert(1, '/Volumes/Transcend/Git/personal_class/')
import numpy as np
import pandas as pd
import pickle as pk

from sqlalchemy import create_engine
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import FeatureUnion
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction import text
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from gensim.models.coherencemodel import CoherenceModel
from gensim.test.utils import common_corpus, common_dictionary
from gensim.matutils import Sparse2Corpus
from gensim import corpora

from term_replace import Replace_Term
import lda_util as lu


author = 'MT Newswires'
db_dir = '~/Dropbox/TopicSentiment/05_11author_db/'
for i, file in enumerate(sorted(listdir(db_dir))):
    engine = create_engine('sqlite:///' + db_dir + file)
    # placeholder = ', '.join(['?'] * len(group1_authors))
    article_query = "SELECT date, words FROM news \
                     WHERE author=? ORDER BY date ASC"
    if i == 0:
        df_article = pd.read_sql(article_query, engine, params=(author,))
        print(df_article.shape)
    else:
        df_temp = pd.read_sql(article_query, engine, params=(author,))
        print(df_temp.shape)
        df_article = pd.concat([df_article, df_temp])

df_article = df_article.set_index('date')

# =============================================================================
# Stem words
# =============================================================================
stp_wds = text.ENGLISH_STOP_WORDS
stemmer = PorterStemmer()


def stem_doc(doc):
    wds = word_tokenize(doc)
    stem_wds = [stemmer.stem(w.lower()) for w in wds if w.lower() not in stp_wds]
    return ' '.join(stem_wds)


df_article.loc[:, 'stem_wds'] = df_article.words.apply(stem_doc)
df_article.loc[:, 'stem_lt'] = df_article.stem_wds.apply(lambda x: x.split())

cnt = CountVectorizer(stop_words='english', max_df=0.8, min_df=20)
X = cnt.fit_transform(df_article.stem_wds)

X_gensim = Sparse2Corpus(X, False)
# transform scikit vocabulary into gensim dictionary
vocab_gensim = {}
for key, val in cnt.vocabulary_.items():
    vocab_gensim[val] = key



word2id = dict((k, v) for k, v in cnt.vocabulary_.items())
id2word = dict((v, k) for k, v in cnt.vocabulary_.items())
d = corpora.Dictionary()
d.id2token = id2word
d.token2id = word2id
# =============================================================================
# Get vocab
# =============================================================================

in_dir = '~/Dropbox/TopicSentiment/01_mtne/'
with open(in_dir + 'p_doctpc_mtne2.pk', 'rb') as f:
    doctpc = pk.load(f)

with open(in_dir + 'p_tpcwds_mtne2.pk', 'rb') as f:
    tpcwds = pk.load(f)

with open(in_dir + 'p_vocab_mtne.pk', 'rb') as f:
    vocab = pk.load(f)


#with open(in_dir + 'p_cv_bestmodel_mtne.pk', 'rb') as f:
#    md = pk.load(f)

# =============================================================================
# Get topic words
# =============================================================================
n_top_words = 100
def get_top_words(tpc, feature_names, n_top_words):
    tpc_wds = []
    for topic_idx, topic in enumerate(tpc):
        tpc_wds.append([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
    return tpc_wds

tpc_gensim = get_top_words(tpcwds, vocab, n_top_words)

for i in np.arange(20):
    cm = CoherenceModel(topics=[tpc_gensim[i]], corpus=X_gensim,
                        texts=df_article.stem_lt.tolist(),
                        dictionary=d, coherence='c_v', topn=n_top_words)
    coherence = cm.get_coherence()
    print(i, coherence)

# =============================================================================
# 
# =============================================================================
prb = []
for i in np.arange(20):
    prb.append(sum(sorted(tpcwds[i, :])[-100:]))

prbnp = np.array(prb)
