from models import encoder
from importlib import reload

reload(encoder)
t2v = encoder.T2V()
#%%
import torch

size = (1, 512, 768)
x = torch.ones(size)
x.size()

t
#%%
from others.utils import datestr2datetime

date2day = [[datestr2datetime(date) for date in one] for one in date_input]


x = date2day[0][0]
print(x.year, x.month, x.day)
#%%
def datetime2vec(date_list):
    emb_list = [
        (
            position_encoder.pe[:, x.year]
            + position_encoder.pe[:, x.month]
            + position_encoder.pe[:, x.day]
        ).unsqueeze(1)
        for date in date_list
    ]
    return torch.cat(emb_list, dim=1)


time_emb = torch.cat([datetime2vec(one) for one in date2day])
time_emb.size()
# %%
n_sents = 20
position_encoder = encoder.PositionalEncoding(0.1, 768)
position_encoder.pe[:, :n_sents].size()
# %%
date_input = [
    [
        "2011-01-15T00:00:00+00:00",
        "2011-01-15T00:00:00+00:00",
        "2011-01-15T00:00:00+00:00",
        "2011-01-16T00:00:00+00:00",
        "2011-01-16T00:00:00+00:00",
        "2011-01-16T00:00:00+00:00",
        "2011-01-17T00:00:00+00:00",
        "2011-01-18T00:00:00+00:00",
        "2011-01-18T00:00:00+00:00",
        "2011-01-19T00:00:00+00:00",
        "2011-01-19T00:00:00+00:00",
        "2011-01-19T00:00:00+00:00",
        "2011-01-20T00:00:00+00:00",
        "2011-01-20T00:00:00+00:00",
        "2011-01-20T00:00:00+00:00",
        "2011-01-20T00:00:00+00:00",
        "2011-01-20T00:00:00+00:00",
        "2011-01-22T00:00:00+00:00",
        "2011-01-22T00:00:00+00:00",
        "2011-01-22T00:00:00+00:00",
        "2011-01-22T00:00:00+00:00",
        "2011-01-23T00:00:00+00:00",
        "2011-01-23T00:00:00+00:00",
        "2011-01-23T00:00:00+00:00",
        "2011-01-23T00:00:00+00:00",
        "2011-01-23T00:00:00+00:00",
        "2011-01-23T00:00:00+00:00",
        "2011-01-23T00:00:00+00:00",
        "2011-01-24T00:00:00+00:00",
        "2011-01-24T00:00:00+00:00",
        "2011-01-24T00:00:00+00:00",
        "2011-01-24T00:00:00+00:00",
        "2011-01-24T00:00:00+00:00",
        "2011-01-24T00:00:00+00:00",
        "2011-01-25T00:00:00+00:00",
        "2011-01-25T00:00:00+00:00",
        "2011-01-26T00:00:00+00:00",
        "2011-01-26T00:00:00+00:00",
        "2011-01-26T00:00:00+00:00",
        "2011-01-26T00:00:00+00:00",
        "2011-01-27T00:00:00+00:00",
        "2011-01-27T00:00:00+00:00",
        "2011-01-27T00:00:00+00:00",
        "2011-01-27T00:00:00+00:00",
        "2011-01-27T00:00:00+00:00",
        "2011-01-27T00:00:00+00:00",
        "2011-01-27T00:00:00+00:00",
        "2011-01-27T00:00:00+00:00",
        "2011-01-27T00:00:00+00:00",
        "2011-01-27T00:00:00+00:00",
        "2011-01-27T00:00:00+00:00",
        "2011-01-27T00:00:00+00:00",
        "2011-01-27T00:00:00+00:00",
        "2011-01-27T00:00:00+00:00",
        "2011-01-27T00:00:00+00:00",
        "2011-01-27T00:00:00+00:00",
        "2011-01-27T00:00:00+00:00",
        "2011-01-27T00:00:00+00:00",
        "2011-01-28T00:00:00+00:00",
        "2011-01-28T00:00:00+00:00",
        "2011-01-28T00:00:00+00:00",
        "2011-01-28T00:00:00+00:00",
        "2011-01-28T00:00:00+00:00",
        "2011-01-28T00:00:00+00:00",
        "2011-01-28T00:00:00+00:00",
        "2011-01-28T00:00:00+00:00",
        "2011-01-28T00:00:00+00:00",
        "2011-01-28T00:00:00+00:00",
        "2011-01-28T00:00:00+00:00",
        "2011-01-28T00:00:00+00:00",
        "2011-01-29T00:00:00+00:00",
        "2011-01-29T00:00:00+00:00",
        "2011-01-29T00:00:00+00:00",
        "2011-01-29T00:00:00+00:00",
        "2011-01-29T00:00:00+00:00",
        "2011-01-29T00:00:00+00:00",
        "2011-01-29T00:00:00+00:00",
        "2011-01-30T00:00:00+00:00",
        "2011-01-30T00:00:00+00:00",
        "2011-01-30T00:00:00+00:00",
        "2011-01-30T00:00:00+00:00",
        "2011-01-30T00:00:00+00:00",
        "2011-01-30T00:00:00+00:00",
        "2011-01-30T00:00:00+00:00",
        "2011-01-30T00:00:00+00:00",
        "2011-01-31T00:00:00+00:00",
        "2011-01-31T00:00:00+00:00",
        "2011-01-31T00:00:00+00:00",
        "2011-01-31T00:00:00+00:00",
        "2011-01-31T00:00:00+00:00",
        "2011-01-31T00:00:00+00:00",
        "2011-01-31T00:00:00+00:00",
        "2011-01-31T00:00:00+00:00",
        "2011-01-31T00:00:00+00:00",
        "2011-01-31T00:00:00+00:00",
        "2011-01-31T00:00:00+00:00",
        "2011-02-01T00:00:00+00:00",
        "2011-02-01T00:00:00+00:00",
        "2011-02-01T00:00:00+00:00",
        "2011-02-01T00:00:00+00:00",
    ]
]

# %%
