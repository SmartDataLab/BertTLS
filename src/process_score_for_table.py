orcle_demo = "rouge1 : 0.36421817916898436, rouge2: 0.3621352403926356, date_f1: 0.373126873126873, cat_rouge1 : 0.36421817916898436, cat_rouge2: 0.3576324773891456"
classic_demo = """({'f_score': 0.2503508228206934,
  'precision': 0.26716562517574577,
  'recall': 0.23552726740260682},
 {'f_score': 0.23621786479914636,
  'precision': 0.25477902221320253,
  'recall': 0.22017749715715498},
 {'f_score': 0.2894843225310515,
  'precision': 0.29177489177489174,
  'recall': 0.2872294372294372},
 {'f_score': 0.46274077537583463,
  'precision': 0.47148150263816413,
  'recall': 0.45431823543727917},
 {'f_score': 0.3124187310480987,
  'precision': 0.3269136030637145,
  'recall': 0.29915465047490547})"""
bert_demo = "rouge1 : 0.08036787483068851, rouge2: 0.07795146102527528, date_f1: 0.08467714502403498, cat_rouge1 : 0.2137284256264502, cat_rouge2: 0.10288088021523226"

# the better way is to make it as a module
# use @ will be better


def float_format_fun(float_str):
    return "{:.2f}".format(float(float_str) * 100)


def res_format_fun(res_list):
    return "\t".join(res_list)


def process_my_format(
    res_str, float_format_fun=float_format_fun, res_format_fun=res_format_fun, idx=None
):
    res_list = [float_format_fun(one.split(": ")[1]) for one in res_str.split(", ")]
    if idx:
        res_list = [res_list[i] for i in idx]
    return res_format_fun(res_list)


import json


def process_classic_format(
    res_str, float_format_fun=float_format_fun, res_format_fun=res_format_fun, idx=None
):
    res_list = [
        float_format_fun(str(x["f_score"]))
        for x in json.loads("[%s]" % res_str[1:-1].replace("'", '"'))
    ]
    if idx:
        res_list = [res_list[i] for i in idx]
    return res_format_fun(res_list)


print(process_my_format(orcle_demo))
print(process_my_format(bert_demo))
print(process_classic_format(classic_demo))

target_str = "rouge1 : 0.011360186532713644, rouge2: 0.004176610978520287, date_f1: 0.07463527851458886, cat_rouge1 : 0.22905027932960895, cat_rouge2: 0.045964125560538124"
target_str = """({'f_score': 0.08937082980304231,
  'precision': 0.06822831695893293,
  'recall': 0.12950020536582843},
 {'f_score': 0.02760754443842192,
  'precision': 0.021829915472510092,
  'recall': 0.03754420663041306},
 {'f_score': 0.3296626984126984,
  'precision': 0.3296626984126984,
  'recall': 0.3296626984126984},
 {'f_score': 0.33968446384103756,
  'precision': 0.2541155944283048,
  'recall': 0.5121376249763993},
 {'f_score': 0.08480647290898881,
  'precision': 0.06545094587416511,
  'recall': 0.12041682249359907})"""
print(process_my_format(target_str, idx=[3, 4, 2, 0, 1]))
print(process_classic_format(target_str, idx=[3, 4, 2, 0, 1]))
