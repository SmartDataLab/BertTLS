#%%

from models.data_loader import load_dataset


class Arg:
    bert_data_path = "../bert_data/entities"


# nyt 还可以换成 crisis t17 entities


args = Arg()
data = load_dataset(args, "train", shuffle=False)
# train 还可以换成 test
# %%
data_list = list(data)
# %%
sum(data_list[0][2]["labels"])
# %%
