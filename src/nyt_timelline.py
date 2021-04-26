sample_tl = [
    [
        "2009-06-25T00:00:00",
        [
            "Dr Murray finds Jackson unconscious in the bedroom of his Los Angeles mansion .",
            "Paramedics are called to the house while Dr Murray is performing CPR , according to a recording of the 911 emergency call .",
            "He travels with the singer in an ambulance to UCLA medical center where Jackson later dies .",
        ],
    ],
    [
        "2009-06-28T00:00:00",
        [
            "Los Angeles police interview Dr Murray for three hours .",
            "His spokeswoman insists he is `` not a suspect '' .",
        ],
    ],
    [
        "2009-07-22T00:00:00",
        [
            "The doctor 's clinic in Houston is raided by officers from the Drug Enforcement Agency -LRB- DEA -RRB- looking for evidence of manslaughter ."
        ],
    ],
    [
        "2009-07-28T00:00:00",
        [
            "Dr Murray 's home is also raided .",
            "The search warrant allows `` authorised investigators to look for medical records relating to Michael Jackson and all of his reported aliases '' .",
            "A computer hard drive and mobile phones are seized , and a pharmacy in Las Vegas is later raided in connection with the case .",
        ],
    ],
    [
        "2009-07-29T00:00:00",
        [
            "Court documents filed in Nevada show that Dr Murray is heavily in debt , owing more than $ 780,000 -LRB- \u00ac # 501,000 -RRB- in judgements against him and his medical practice , outstanding mortgage payments on his house , child support and credit cards ."
        ],
    ],
]

# %%
import json
import torch

d_list = json.load(open("../../../data/datasets/pretrain_tl/nyt.test.0.json"))

# %%
from models.data_loader import load_dataset


class Arg:
    bert_data_path = "../bert_data/nyt"


# nyt 还可以换成 crisis t17 entities


args = Arg()
data = load_dataset(args, "test", shuffle=False)
# train 还可以换成 test
# %%
data_list = list(data)
# %%
sum(data_list[0][2]["labels"])
# %%
len(data_list)
# %%
len(data_list[0])
# %%
data_list[0][0].keys()
# %%
d_list[0].keys()
# %%
data_list[0][-1]["src_txt"][0]
# %%
" ".join(d_list[-1]["src"][0])
# %%

# %%
data_list[0][-1]["tgt_date"]
# %%
timeline_dict = {}
from tilse.data import timelines
from others.utils import get_dict_for_tl

for i in range(22):
    tl = get_dict_for_tl(
        data_list[0][i]["tgt_txt"].split("<q>"), data_list[0][i]["tgt_date"]
    )
    timeline_dict[d_list[i]["name"]] = tl

# %%
timeline_dict["slovenia_and"]
# %%
target_path = "../../../data/datasets_acl/nyt/"
# %%
import os

for name in os.listdir(target_path):
    json_dump_list = [
        [parse_date(key), value] for key, value in timeline_dict[name].items()
    ]
    json.dump(json_dump_list, open(target_path + name + "/timelines.jsonl", "w"))
# %%
def parse_date(date_str):
    ymd, cms = date_str.split("T")
    y, m, d = ymd[:4], ymd[4:6], ymd[6:]
    c, min_, s = cms[:2], cms[2:4], cms[4:]
    return "-".join([y, m, d]) + "T" + ":".join([c, min_, s])


# %%
