#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 07:41:07 2019

Run LDA: Select topic using perplexity

@author: LEIHAO
"""
from os import listdir
import sys
sys.path.insert(1, '/hpctmp/personal_class/')
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


from term_replace import Replace_Term
import lda_util as lu

# =============================================================================
# Read in the 19 authors, who consistently contribute from 2012 to 2017
# =============================================================================
# author_dir = '/home//00_data/'
# with open(author_dir + 'p_active_author_edt.txt', 'r') as f:
#     active_authors = f.read().split('\n')[:11]  # last one is ''
# 
# grp1_index = [3, 4, 6]
# grp2_index = [0, 1, 2, 7, 8, 9]
# grp3_index = [5]
# grp4_index = [10]
# group1_authors = [active_authors[i] for i in grp1_index]
# group2_authors = [active_authors[i] for i in grp2_index]
# =============================================================================
# Read in the news data and symbol infor from the nasda.db
# =============================================================================
author = 'MT Newswires'
db_dir = '/hpc/Data/NASDAQ_Related/02_11authors_lemtxt_db/'
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

# =============================================================================
# Transform the text to matrix
# =============================================================================
cnt = CountVectorizer(stop_words='english', max_df=0.8, min_df=20)
X = cnt.fit_transform(df_article.stem_wds)
print("X's shape is ", X.shape)

# =============================================================================
# Find the number of topics
# =============================================================================

search_param = {'n_components': [16, 18, 20, 22, 24]}


class MyLDAWithPerplexityScorer(LatentDirichletAllocation):
    def score(self, X, y=None):

        # You can change the options passed to perplexity here
        score = super(MyLDAWithPerplexityScorer, self).perplexity(X, sub_sampling=False)

        # Since perplexity is lower for better, so we do negative
        return -1*score


lda = MyLDAWithPerplexityScorer(learning_method='batch', random_state=0,
                                learning_offset=50)
model = GridSearchCV(lda, param_grid=search_param, cv=5)
model.fit(X)

# =============================================================================
# Save the Cross-Validation result
# =============================================================================
out_dir = '/hpc/Data/NASDAQ_Related/04_11authors_lda_indiv/'
df_cv_results = pd.DataFrame.from_dict(model.cv_results_)
df_cv_results.to_csv(out_dir + 'p_crossValid_summary_' + 'mtne2' + '.csv')

n_tpcs = model.best_params_['n_components']
print("The best topic number of mtne is {:d}".format(n_tpcs))

best_lda_model = model.best_estimator_
doc_tpc_mat = best_lda_model.transform(X)
tpc_wds_mat = best_lda_model.components_ / best_lda_model.components_.sum(
        axis=1)[:, np.newaxis]


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


vocab = cnt.get_feature_names()
print_top_words(best_lda_model, vocab, 20)

with open(out_dir + 'p_cv_bestmodel2_mtne.pk', 'wb') as f:
    pk.dump(best_lda_model, f)

with open(out_dir + 'p_tpcwds_mtne2.pk', 'wb') as f4:
    pk.dump(tpc_wds_mat , f4)

with open(out_dir + 'p_doctpc_mtne2.pk', 'wb') as f5:
    pk.dump(doc_tpc_mat, f5)
