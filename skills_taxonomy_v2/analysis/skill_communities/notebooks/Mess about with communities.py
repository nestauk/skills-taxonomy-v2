# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# Find skill communities using the Industrial Taxonomy pipeline.
#
# - Tokenise and lemmatise job descriptions
# - Use TFIDF & VADER to remove generic terms
# - Create a co-occurrence network of words
# - Use community modelling to identify communities
# - Identify salient terms in the communities and use these to tag job ads with skills - Compare average word2vec job ads to do this
#
#
# Sentence based?/ a few sentences/ or whole JD?

# %%
import json
from collections import Counter
from itertools import chain, combinations

import spacy
import pandas as pd
import networkx as nx
from cdlib import algorithms, ensemble, evaluation
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# %%
nlp = spacy.load("en_core_web_sm")

# %%
with open('../../../../inputs/TextKernel_sample/jobs_new.1.jsonl', 'r') as file:
    data = [json.loads(line) for line in file]

# %%
data[0]['full_text']

# %%
# Separate out words in whole JD
# Future: n-grams, lemmatize, clean out punctuation/spaces
job_d_words = []
for job_id in range(0,1000):
    text = data[job_id]['full_text']
    text = text.replace('\n',' ')
    text = ' '.join(text.split())
    doc = nlp(text)
    doc_words = []
    for sent in doc:
        doc_words.append(sent.text)
    doc_words = [word for word in doc_words if (len(word)>1 and not word.isnumeric())] # removes '.' etc
    job_d_words.append(doc_words)


# %% [markdown]
# ## Build network from co-occurences

# %%
def get_cooccurences_network(sentence_words):
    # Sort list of words in alphabetical order
    # so you get tuples in the same order e.g. ('A_word', 'B_word') not ('B_word', 'A_word')
    pairs = list(chain(*[list(combinations(sorted(x), 2)) for x in sentence_words]))
    pairs = [x for x in pairs if len(x) > 0]
    
    edge_list = pd.DataFrame(pairs, columns=["source", "target"])
    edge_list["weight"] = 1
    edge_list_weighted = (
            edge_list.groupby(["source", "target"])["weight"].sum().reset_index(drop=False)
        )
    # Build network
    net = nx.from_pandas_edgelist(edge_list_weighted, edge_attr=True)

    # # Filter out low frequency words
    # thres = 0.1
    # freq = pd.Series(list(chain(*sentences_broken))).value_counts() / len(sentences_broken)
    # to_keep = freq.loc[freq < thres].index.tolist()
    # net_filt = net.subgraph(to_keep)
    
    return net
    


# %%
# job_d_words_example = [['first', 'sentence', 'in', 'words'], ['some', 'words', 'in', 'sentence']]
# net = get_cooccurences_network(job_d_words_example)

# %%
net = get_cooccurences_network(job_d_words)


# %% [markdown]
# ## Community detection algorithms
# - Trying out different ones

# %%
def best_cd_algo(net):
    method_pars = {
        algorithms.louvain: [
            ensemble.Parameter(name="resolution", start=0.7, end=1, step=0.1)
        ],
        algorithms.leiden:[ensemble.BoolParameter(name='weights',value='weight')],
        algorithms.agdl: [ensemble.Parameter(name="number_communities", start=20, end=50, step=10),
        ensemble.Parameter(name="kc", start=4, end=10, step=2)],
        algorithms.chinesewhispers: [],
        algorithms.label_propagation: [],
        algorithms.markov_clustering: [],
    }
    
    qual=evaluation.erdos_renyi_modularity
    aggregate=max
    
    algos = list(method_pars.keys())
    pars = list(method_pars.values())

    results = ensemble.pool_grid_filter(net, algos, pars, qual, aggregate=aggregate)
    
    results_container = []
    comm_assignments_container = {}
    for comm, score in results:
        try:
            print(comm.method_name)
            out = [
                comm.method_name,
                len(comm.communities),
                comm.method_parameters,
                score.score,
                comm,
            ]
            results_container.append(out)
            comm_assignments_container[comm.method_name] = comm.communities
        except:
            print(f"error with algorithm {comm.method_name}")
            pass

    results_df = pd.DataFrame(
        results_container,
        columns=["method", "comm_n", "parametres", "score", "instance"],
    )
    
    return results_df, comm_assignments_container
    


# %%
results_df, comm_assignments_container = best_cd_algo(net)

# %%
results_df

# %% [markdown]
# ## Best algorithm results

# %%
best_algo_name = results_df.iloc[results_df['score'].argmax()]['method']
best_algo_name

# %%
results_df.iloc[results_df['score'].argmax()]['parametres']

# %%
communities = comm_assignments_container[best_algo_name]

# %% [markdown]
# ## Or just use one

# %%
coms = algorithms.louvain(net, weight='weight', resolution=0.7, randomize=None)

# %%
communities = coms.communities


# %% [markdown]
# ## Name communities by top TFIDF terms

# %%
def get_top_tf_idf_words(clusters_vect, feature_names, top_n=2):
    """
    From https://stackoverflow.com/questions/34232190/scikit-learn-tfidfvectorizer-how-to-get-top-n-terms-with-highest-tf-idf-score
    """
    sorted_nzs = np.argsort(clusters_vect.data)[: -(top_n + 1) : -1]
    return feature_names[clusters_vect.indices[sorted_nzs]]


# %%
communities_merged = [' '.join(c) for c in communities]

# %%
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
tfidf_vects = tfidf_vectorizer.fit_transform(communities_merged)

num_top_words=2
# Top n words for each cluster + other info
feature_names = np.array(tfidf_vectorizer.get_feature_names())
cluster_info = {
    i: "-".join(
            list(get_top_tf_idf_words(document, feature_names, num_top_words))
        )
    for i, document in enumerate(tfidf_vects)
}

# %%
cluster_info

# %%
