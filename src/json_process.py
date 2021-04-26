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
entities_test_names = [
    "Bashar_al-Assad",
    "Dilma_Rousseff",
    "Enron",
    "John_Boehner",
    "Marco_Rubio",
    "Morgan_Tsvangirai",
    "Phil_Spector",
    "Sarah_Palin",
    "Tiger_Woods",
]
# %%
d_train_list = []
d_test_list = []
for d in d_list:
    if d["name"] in entities_test_names:
        d_test_list.append(d)
    else:
        print(d["name"])
        d_train_list.append(d)
print(len(d_train_list), len(d_test_list))
# %%
target_path = "../../../data/datasets/labeldata_new_structure/"
json.dump(d_train_list, open(target_path + "entities.train.0.json", "w"))
json.dump(d_test_list, open(target_path + "entities.test.0.json", "w"))
# %%

json_path = "../../../data/datasets/nyt_new%20structure/%20train.demo.json"
target_path = "../../../data/datasets/pretrain_tl/"
d_list = json.load(open(json_path))
# %%

#%%
test_idx_list = np.random.choice(
    list(range(len(d_list))), int(len(d_list) * 0.2), replace=False
)
# %%
nyt_test_names = []
nyt_test_list = []
nyt_train_names = []
nyt_train_list = []
for i in range(len(d_list)):
    if i in test_idx_list:
        nyt_test_list.append(d_list[i])
        nyt_test_names.append(d_list[i]["name"])
    else:
        nyt_train_list.append(d_list[i])
        nyt_train_names.append(d_list[i]["name"])
print(len(nyt_test_names), len(nyt_train_names))

# %%
json.dump(nyt_train_list, open(target_path + "nyt.train.0.json", "w"))
json.dump(nyt_test_list, open(target_path + "nyt.test.0.json", "w"))

# %%

json_path = "../../../data/datasets/nyt_new%20structure/crisis_new.json"
d_list = json.load(open(json_path))
#%%
d_list[0]["src"][0]
#%%
def get_new_structure_one(x):
    num_list = [key.replace("tgt_date", "") for key in x.keys() if "tgt_date" in key]
    new_dict = {key: x[key] for key in ["name", "src", "src_date"]}
    new_dict["tgt"] = [x["tgt%s" % num] for num in num_list]
    new_dict["tgt_date"] = [x["tgt_date%s" % num] for num in num_list]
    return new_dict


get_new_structure_one(d_list[0]).keys()
# %%

crisis_test_names = ["yemen"]
crisis_train_list = []
crisis_test_list = []
for d in d_list:
    if d["name"] in crisis_test_names:
        crisis_test_list.append(get_new_structure_one(d))
    else:
        print(d["name"])
        crisis_train_list.append(get_new_structure_one(d))
print(len(crisis_train_list), len(crisis_test_list))
target_path = "../../../data/datasets/multi_tl/"
json.dump(crisis_train_list, open(target_path + "crisis.train.0.json", "w"))
json.dump(crisis_test_list, open(target_path + "crisis.test.0.json", "w"))
# %%
json_path = "../../../data/datasets/nyt_new%20structure/T17_new.json"
d_list = json.load(open(json_path))

t17_test_names = ["mj", "syria"]
t17_train_list = []
t17_test_list = []
for d in d_list:
    if d["name"] in t17_test_names:
        t17_test_list.append(get_new_structure_one(d))
    else:
        print(d["name"])
        t17_train_list.append(get_new_structure_one(d))
print(len(t17_train_list), len(t17_test_list))
target_path = "../../../data/datasets/multi_tl/"
json.dump(t17_train_list, open(target_path + "t17.train.0.json", "w"))
json.dump(t17_test_list, open(target_path + "t17.test.0.json", "w"))