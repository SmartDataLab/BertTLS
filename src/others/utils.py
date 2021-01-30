import os
import re
import shutil
import time
import collections

from others import pyrouge

import datetime
from tilse.data import timelines
from tilse.evaluation import rouge
import pandas as pd

REMAP = {
    "-lrb-": "(",
    "-rrb-": ")",
    "-lcb-": "{",
    "-rcb-": "}",
    "-lsb-": "[",
    "-rsb-": "]",
    "``": '"',
    "''": '"',
}


def clean(x):
    return re.sub(
        r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''", lambda m: REMAP.get(m.group()), x
    )


def process(params):
    temp_dir, data = params
    candidates, references, pool_id = data
    cnt = len(candidates)
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    tmp_dir = os.path.join(temp_dir, "rouge-tmp-{}-{}".format(current_time, pool_id))
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
        os.mkdir(tmp_dir + "/candidate")
        os.mkdir(tmp_dir + "/reference")
    try:

        for i in range(cnt):
            if len(references[i]) < 1:
                continue
            with open(
                tmp_dir + "/candidate/cand.{}.txt".format(i), "w", encoding="utf-8"
            ) as f:
                f.write(candidates[i])
            with open(
                tmp_dir + "/reference/ref.{}.txt".format(i), "w", encoding="utf-8"
            ) as f:
                f.write(references[i])
        r = pyrouge.Rouge155(temp_dir=temp_dir)
        r.model_dir = tmp_dir + "/reference/"
        r.system_dir = tmp_dir + "/candidate/"
        r.model_filename_pattern = "ref.#ID#.txt"
        r.system_filename_pattern = r"cand.(\d+).txt"
        rouge_results = r.convert_and_evaluate()
        print(rouge_results)
        results_dict = r.output_to_dict(rouge_results)
    finally:
        pass
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
    return results_dict


def test_rouge(temp_dir, cand, ref):
    candidates = [line.strip() for line in open(cand, encoding="utf-8")]
    references = [line.strip() for line in open(ref, encoding="utf-8")]
    assert len(candidates) == len(references)

    cnt = len(candidates)
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    tmp_dir = os.path.join(temp_dir, "rouge-tmp-{}".format(current_time))
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
        os.mkdir(tmp_dir + "/candidate")
        os.mkdir(tmp_dir + "/reference")
    try:

        for i in range(cnt):
            if len(references[i]) < 1:
                continue
            with open(
                tmp_dir + "/candidate/cand.{}.txt".format(i), "w", encoding="utf-8"
            ) as f:
                f.write(candidates[i])
            with open(
                tmp_dir + "/reference/ref.{}.txt".format(i), "w", encoding="utf-8"
            ) as f:
                f.write(references[i])
        r = pyrouge.Rouge155(temp_dir=temp_dir)
        r.model_dir = tmp_dir + "/reference/"
        r.system_dir = tmp_dir + "/candidate/"
        r.model_filename_pattern = "ref.#ID#.txt"
        r.system_filename_pattern = r"cand.(\d+).txt"
        rouge_results = r.convert_and_evaluate()
        results_dict = r.output_to_dict(rouge_results)
    finally:
        pass
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
    return results_dict


def rouge_results_to_str(results_dict):
    return ">> ROUGE-F(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\nROUGE-R(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\n".format(
        results_dict["rouge_1_f_score"] * 100,
        results_dict["rouge_2_f_score"] * 100,
        results_dict["rouge_l_f_score"] * 100,
        results_dict["rouge_1_recall"] * 100,
        results_dict["rouge_2_recall"] * 100,
        results_dict["rouge_l_recall"] * 100,
    )


def datestr2datetime(date_str):
    try:
        date_str = date_str[:10]
        if "-" in date_str:
            y, m, d = date_str.split("-")
        elif "T" in date_str:
            y, m, d = date_str[:4], date_str[4:6], date_str[6:8]
        return datetime.date(int(y), int(m), int(d))
    except Exception as e:
        print("date parse fail", e, date_str)
        return datetime.date(1970, 1, 1)


def get_dict_for_tl(src_list, date_list):
    tl_dict = {date: [] for date in set(date_list)}
    for i in range(len(date_list)):
        tl_dict[date_list[i]].append(src_list[i])
    return tl_dict


def cal_rouge_tls(
    doc_sent_list,
    doc_date_list,
    abstract_sent_list,
    abstract_date_list,
    mode="date",
    multi_tl=False,
):
    """
    docstring
    """
    evaluator = rouge.TimelineRougeEvaluator(measures=["rouge_1", "rouge_2"])
    doc_datetime_list = [datestr2datetime(date_str) for date_str in doc_date_list]
    predicted_timeline = timelines.Timeline(
        get_dict_for_tl(doc_sent_list, doc_datetime_list)
    )
    if multi_tl:
        gold_datetime_list = [
            [datestr2datetime(date_str) for date_str in one_tl]
            for one_tl in abstract_date_list
        ]
        groundtruth = timelines.GroundTruth(
            [
                timelines.Timeline(get_dict_for_tl(one_tl, gold_datetime_list[i]))
                for i, one_tl in enumerate(abstract_sent_list)
            ]
        )
    else:
        gold_datetime_list = [
            datestr2datetime(date_str) for date_str in abstract_date_list
        ]
        groundtruth = timelines.GroundTruth(
            [
                timelines.Timeline(
                    get_dict_for_tl(abstract_sent_list, gold_datetime_list)
                )
            ]
        )
    if mode == "cat":
        res_dict = evaluator.evaluate_concat(predicted_timeline, groundtruth)
    elif mode == "date":
        res_dict = evaluator.evaluate_align_date_content_costs(
            predicted_timeline, groundtruth
        )
    return res_dict["rouge_1"]["f_score"], res_dict["rouge_2"]["f_score"]


def date_truncate(x):
    if x and len(x) > 10:
        return x[:10]
    return x


def cal_date_f1(pred_date_list, abstract_date_list, multi_tl=False):
    pred_counts = collections.Counter([date_truncate(x) for x in pred_date_list])

    ref_counts = collections.Counter([date_truncate(x) for x in abstract_date_list])
    match = 0
    for tok in pred_counts:
        match += min(pred_counts[tok], ref_counts[tok])

    prec_denom = len(pred_date_list)

    recall_denom = len(abstract_date_list)

    return {
        "p": match / prec_denom,
        "r": match / recall_denom,
        "f1": 2 * match / (prec_denom + recall_denom),
    }


def _order_source(sent_str_list, src_date_list):
    map_tuple = tuple(zip(sent_str_list, src_date_list))
    map_tuple = sorted(map_tuple, key=lambda x: x[1])
    return [x[0] for x in map_tuple], [x[1] for x in map_tuple]


def _sent2token(sents_str_list):
    return [[x for x in sent_str.split()] for sent_str in sents_str_list]


def _process_source(sent_str_list, src_date_list):
    new_sent_str_list = []
    new_str_date_list = []
    for i, sent_token_list in enumerate(sent_str_list):
        if len(sent_token_list) == 0:
            continue
        sents_one_date = [_rouge_clean(x) for x in " ".join(sent_token_list).split(".")]
        dates_one_date = [src_date_list[i]] * len(sents_one_date)
        new_sent_str_list += sents_one_date
        new_str_date_list += dates_one_date

    return new_sent_str_list, new_str_date_list


def taxostat_distance(raw_taxostr_lst, depth=4) -> list:
    """
    params:
    timeline: 一条时间线，包括日期与文本
    [['date0', ['id : raw_taxostr : page : text']],
    ['date1', ['id : raw_taxostr : page : text'],
    ...]
    depth: taxonomic classifier的最大追索深度,一般最大距离为depth-1
    return: list,每一个时间节点距离基准taxonomic classifier的平均距离
    """
    raw_taxostr_lst = [x if type(x) == str else "" for x in raw_taxostr_lst]

    taxostr_lst = [raw_taxostr.split("|") for raw_taxostr in raw_taxostr_lst]
    # taxostr分段
    taxo_unit_lst = [[taxostr.split("/") for taxostr in unit] for unit in taxostr_lst]

    # 计算距离
    # 以出现频次最高的taxo为基准
    try:
        base_taxo = (
            pd.value_counts(
                [
                    "/".join(taxo[0 : min(depth, len(taxo))])
                    for unit in taxo_unit_lst
                    for taxo in unit
                ]
            )
            .index[0]
            .split("/")
        )
    except Exception as e:
        print(e)
        return []
    base_len = len(base_taxo)
    # 计算每个时间节点内taxo的平均距离
    taxo_distance_lst = []
    for taxo_unit in taxo_unit_lst:
        curr_scores = []
        for taxo in taxo_unit:  # 计算每一个taxo距离base_taxo的距离
            minus = 1
            for i in range(min(base_len, len(taxo))):
                if taxo[i] != base_taxo[i]:
                    minus = 0
                    break

            score = depth - minus - i
            curr_scores.append(score)

        taxo_distance_lst.append(sum(curr_scores) * 1.0 / len(taxo_unit))

    return taxo_distance_lst


def _rouge_clean(s):
    return re.sub(r"[^a-zA-Z0-9 ]", "", s)


#%%
def _rm_blank(s):
    return " ".join([x for x in s.split(" ") if len(x) > 0])


# %%
