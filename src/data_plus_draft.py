#%%
import pandas as pd
import numpy as np
import json

json_path = "../../../data/datasets/nyt_new%20structure/entities_new.json"
# lines = open(json_path).readlines()
d_list = json.load(open(json_path))
# %%
d_list[0].keys()
# %%
name = d_list[0]["name"]
name
# %%
import os

name_list = os.listdir("../../../data/datasets/entities")
name in name_list
# %%
from news_tls.data import Dataset

flag = 0
dataset = Dataset("/data1/su/app/text_forecast/data/datasets/entities/")
# for col in dataset.collections:
#     print(col.name)  # topic name
#     print(col.keywords)  # topic keywords
#     a = next(col.articles())  # articles collection of this topic
#     print("article pub time", a.time)
#     if col.name == name and flag > 10:
#         break

#     for s in a.sentences:
#         if flag > 10:
#             break
#         print(s.raw, s.time)
#         flag += 1
# %%
name

# %%
name_list2 = [col.name for col in dataset.collections]

# %%
name in name_list2
# %%

collection_dict = {col.name: col for col in dataset.collections}
col = collection_dict[name]
col

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

print("vectorizer...")
vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
fit_input = [s.raw for a in col.articles() for s in a.sentences]
# TODO: print and get a functions for BertTLS
vectorizer.fit(fit_input)

#%%
raw_sents = [s.raw for s in next(col.articles()).sentences]
X = vectorizer.transform(raw_sents)
X[2].data
X[2].toarray().shape
# %%
from news_tls.datewise import MentionCountDateRanker, PM_Mean_SentenceCollector

times = [a.time for a in col.articles()]
col.start = min(times)
col.end = max(times)

date_ranker = MentionCountDateRanker()
sent_collector = PM_Mean_SentenceCollector(5, 2)
# TODO: leanrning how to assign date ranke to sentence

print("date ranking...")
ranked_dates = date_ranker.rank_dates(col)

start = col.start.date()
end = col.end.date()
ranked_dates = [d for d in ranked_dates if start <= d <= end]
# %%
ranked_dates

# %%
articles_generator = col.articles()
a = next(articles_generator)

# %%
a = next(articles_generator)
a.time
# %%
times

# %%
from datetime import date


# %%
rank_id = []
for a in col.articles():
    a_id_list = []
    for s in a.sentences:
        if s.time:
            target_time = s.time
        else:
            target_time = a.time
        try:
            target = date(target_time.year, target_time.month, target_time.day)
            id_ = ranked_dates.index(target)
        except:
            id_ = -1
        a_id_list.append(id_)
    rank_id.append(a_id_list)
# %%
rank_id
# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from news_tls.clust import TemporalMarkovClusterer, ClusterDateMentionCountRanker

#%%
import news_tls.clust
from importlib import reload


TemporalMarkovClusterer = news_tls.clust.TemporalMarkovClusterer
ClusterDateMentionCountRanker = news_tls.clust.ClusterDateMentionCountRanker
#%%
clusterer = TemporalMarkovClusterer()
cluster_ranker = ClusterDateMentionCountRanker()
print("clustering articles...")
doc_vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
clusters = clusterer.cluster(col, doc_vectorizer)

print("assigning cluster times...")
for c in clusters:
    c.time = c.most_mentioned_time()
    if c.time is None:
        c.time = c.earliest_pub_time()

print("ranking clusters...")
ranked_clusters = cluster_ranker.rank(clusters, col)

# %%
x = ranked_clusters[1].articles[0]
# %%
for c in ranked_clusters:
    for a in c.articles:
        for s in a.sentences:
            s.new_attr = "haha"
# %%
list(ranked_clusters)[0].articles[0].sentences[0].new_attr
# %%
