#%%
# TODO: intergrate mongodb and tensorboard, tensorboard for realtime debug, and mongodb save the result for the paper
from pymongo import MongoClient

db = "news-tls"
collection = "acl_v1"
CONN = MongoClient("localhost")
DB = CONN[db]
COLLECTION = DB[collection]
# %%
COLLECTION.find_one()
# %%
import os

file_nyt_model = [file_ for file_ in os.listdir("../models") if "pt" in file_]
file_nyt_model
# %%
set([file_.split("_")[2] for file_ in file_nyt_model])
# %%
log_id = [
    "1611925285",
    "1611926105",
    "1611926317",
    "1611926842",
    "1611981734",
    "1612055512",
]

# %%
list(COLLECTION.find({"log_id": int(log_id[5])}))[0]
# %%
res3_list = [
    (one["log_id"], one["args"]["bert_data_path"])
    for one in COLLECTION.find()
    if one["name"] == "info"
    and "bert_data_path" in one["args"].keys()
    and "3" in one["args"]["bert_data_path"]
]
res3_list.sort(key=lambda x: x[0])
#%%

for id_, data_path in res3_list:
    print([file_ for file_ in file_nyt_model if str(id_) in file_])
#%%
target_id = "1612170357"
list(COLLECTION.find({"log_id": int(target_id)}))
# %%
filter_id = list(
    set(
        [
            one["log_id"]
            for one in COLLECTION.find()
            if one["name"] == "info"
            and one["args"]["encoder"] == "rnn"
            and one["args"]["bert_data_path"] == "../bert_data/t17"
            and ("use_date" not in one["args"].keys() or not one["args"]["use_date"])
            and "nyt" not in one["args"]["train_from"]
        ]
    )
)
filter_id
# %%
for id_ in filter_id:
    print([file_ for file_ in file_nyt_model if str(id_) in file_])
# %%