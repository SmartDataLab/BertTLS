import gc
import glob
import hashlib
import itertools
import json
import os
import re
import subprocess
import time
from os.path import join as pjoin

import torch
from multiprocess import Pool
from pytorch_pretrained_bert import BertTokenizer
from tqdm import tqdm
from others.logging import logger
from others.utils import (
    clean,
    cal_rouge_tls,
    cal_rouge_tls_given,
    cal_date_f1,
    _rouge_clean,
    _process_source,
    _rm_blank,
    _sent2token,
    taxostat_distance,
)
from prepro.utils import _get_word_ngrams

import datetime
from tilse.data import timelines
from tilse.evaluation import rouge

import math
import numpy as np
import random


def load_json(p, lower):
    source = []
    tgt = []
    flag = False
    for sent in json.load(open(p))["sentences"]:
        tokens = [t["word"] for t in sent["tokens"]]
        if lower:
            tokens = [t.lower() for t in tokens]
        if tokens[0] == "@highlight":
            flag = True
            continue
        if flag:
            tgt.append(tokens)
            flag = False
        else:
            source.append(tokens)

    source = [clean(" ".join(sent)).split() for sent in source]
    tgt = [clean(" ".join(sent)).split() for sent in tgt]
    return source, tgt


def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def combination_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r"[^a-zA-Z0-9 ]", "", s)

    max_rouge = 0.0
    max_idx = (0, 0)
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(" ".join(abstract)).split()
    sents = [_rouge_clean(" ".join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    impossible_sents = []
    for s in range(summary_size + 1):
        combinations = itertools.combinations(
            [i for i in range(len(sents)) if i not in impossible_sents], s + 1
        )
        for c in combinations:
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)["f"]
            rouge_2 = cal_rouge(candidates_2, reference_2grams)["f"]

            rouge_score = rouge_1 + rouge_2
            if s == 0 and rouge_score == 0:
                impossible_sents.append(c[0])
            if rouge_score > max_rouge:
                max_idx = c
                max_rouge = rouge_score
    return sorted(list(max_idx))


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r"[^a-zA-Z0-9 ]", "", s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(" ".join(abstract)).split()
    sents = [_rouge_clean(" ".join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if i in selected:
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)["f"]
            rouge_2 = cal_rouge(candidates_2, reference_2grams)["f"]
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if cur_id == -1:
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)


def weak_supervision_selection(
    doc_sent_list,
    doc_date_list,
    doc_page_list,
    doc_taxo_list,
    abstract_size=8,
    page_weight=1,
    taxo_weight=10,
    use_date=True,
    date_size=2,
):

    doc_taxoscore_list = taxostat_distance(doc_taxo_list)
    origin_tuple = tuple(
        zip(range(len(doc_page_list)), doc_page_list, doc_taxoscore_list, doc_date_list)
    )

    # oracle选取初步想法：
    # 1.先由page, taxo加权排序得到sorted_tuple(page越大越好，taxo越小越好),
    # 2.再取sorted_tuple中前n*abstract_size个(n待定)组成小集合selected_tuple
    # 3.再对小集合里所有的timeline组合取'date方差'最小的一组得到result_tuple

    sorted_tuple = sorted(
        origin_tuple,
        key=lambda x: page_weight * x[1] + taxo_weight * (4 - x[2]),
        reverse=True,  # w_page * page + w_taxo * (4-taxo)
    )

    if len(sorted_tuple) <= abstract_size:
        use_date = False  # 当总文章数小于或等于时间线size数时，无法使用date筛选

    if use_date == True:
        select_size = min(math.ceil(date_size * abstract_size), len(sorted_tuple))
        selected_tuple = sorted_tuple[0 : select_size - 1]
        min_Var_date = float("inf")
        for timeline in itertools.combinations(selected_tuple, abstract_size):
            dates = list(zip(*timeline))[3]
            datenum = [int(date[0:8]) for date in dates]
            Var_date = np.var(datenum)
            if Var_date < min_Var_date:
                min_Var_date = Var_date
                result_tuple = timeline

    else:
        result_tuple = sorted_tuple[0 : min(abstract_size, len(sorted_tuple))]

    result_tuple = sorted(result_tuple, key=lambda x: x[3])  # result_tuple按时间顺序排序

    abstract = []
    abstract_date = []
    oracle_ids = []
    for i in range(len(result_tuple)):
        idx = result_tuple[i][0]
        oracle_ids.append(idx)
        abstract.append(doc_sent_list[idx])
        abstract_date.append(doc_date_list[idx])

    return abstract, abstract_date, oracle_ids


def weak_supervision_selection_bak(
    doc_sent_list, doc_date_list, doc_page_list, doc_taxo_list, abstract_size
):
    # simple method sort first
    sort_tuple = sorted(
        tuple(zip(doc_page_list, range(len(doc_page_list)))), key=lambda x: x[0]
    )
    abstract = []
    abstract_date = []
    oracle_ids = []
    for i in range(min(abstract_size, len(sort_tuple))):
        idx = sort_tuple[i][1]
        oracle_ids.append(idx)
        abstract.append(doc_sent_list[idx])
        abstract_date.append(doc_date_list[idx])
    return abstract, abstract_date, oracle_ids


def combination_selection_tls(
    doc_sent_list,
    abstract_sent_list,
    doc_date_list,
    abstract_date_list,
    summary_size,
    multi_tl=False,
):

    max_rouge = 0.0
    max_idx = (0, 0)
    if multi_tl:
        abstract_str_list = [
            [_rm_blank(_rouge_clean(" ".join(s))) for s in one_sent_list]
            for one_sent_list in abstract_sent_list
        ]
    else:
        abstract_str_list = [
            _rm_blank(_rouge_clean(" ".join(s))) for s in abstract_sent_list
        ]
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(" ".join(abstract)).split()
    sents_str_list, sents_date_list = _process_source(doc_sent_list, doc_date_list)

    impossible_sents = []
    for s in range(summary_size + 1):
        combinations = itertools.combinations(
            [i for i in range(len(sents_str_list)) if i not in impossible_sents], s + 1
        )
        for c in combinations:
            sent_str_combination = [_rm_blank(sents_str_list[idx]) for idx in c]
            sent_date_combination = [sents_date_list[idx] for idx in c]
            rouge_1, rouge_2 = cal_rouge_tls(
                sent_str_combination,
                sent_date_combination,
                abstract_str_list,
                abstract_date_list,
                multi_tl=multi_tl,
            )

            date_f1 = cal_date_f1(
                sent_date_combination, abstract_date_list, multi_tl=multi_tl
            )["f1"]
            rouge_score = rouge_1 + rouge_2 + date_f1
            if s == 0 and rouge_score == 0:
                impossible_sents.append(c[0])
            if rouge_score > max_rouge:
                max_idx = c
                max_rouge = rouge_score
    return sorted(list(max_idx)), sents_str_list, sents_date_list


def greedy_selection_tls(
    doc_sent_list,
    abstract_sent_list,
    doc_date_list,
    abstract_date_list,
    summary_size,
    multi_tl=False,
):

    max_rouge = 0.0
    if multi_tl:
        abstract_str_list = [
            [_rm_blank(_rouge_clean(" ".join(s))) for s in one_sent_list]
            for one_sent_list in abstract_sent_list
        ]
    else:
        abstract_str_list = [
            _rm_blank(_rouge_clean(" ".join(s))) for s in abstract_sent_list
        ]
    sents_str_list, sents_date_list = _process_source(doc_sent_list, doc_date_list)

    selected = []
    for _ in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents_str_list)):
            if i in selected or len(_rm_blank(sents_str_list[i])) == 0:
                continue
            c = selected + [i]
            sent_str_combination = [_rm_blank(sents_str_list[idx]) for idx in c]
            sent_date_combination = [sents_date_list[idx] for idx in c]
            rouge_1, rouge_2 = cal_rouge_tls(
                sent_str_combination,
                sent_date_combination,
                abstract_str_list,
                abstract_date_list,
                multi_tl=multi_tl,
            )
            date_f1 = cal_date_f1(
                sent_date_combination, abstract_date_list, multi_tl=multi_tl
            )["f1"]
            rouge_score = rouge_1 + rouge_2 + date_f1
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if cur_id == -1:
            return selected, sents_str_list, sents_date_list
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected), sents_str_list, sents_date_list


def random_greed_selection_tls(
    ranked_clusters, # collection of articles
    timelines,
    summary_size, # change_summary_size to 3
    random_size=20, # change this to 50 to make more to be look up
    multi_tl=False,
):
    max_rouge = 0.0
    sents_str_list, sents_date_list = _process_source(doc_sent_list, doc_date_list)
    # TODO: using this for processing of d

    selected = []
    for _ in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for _ in range(random_size):
            i = int(random.random() * len(ranked_clusters))
            j = int(random.random() * len(ranked_clusters[i]))
            a = ranked_clusters[i].ariticles[j]
            id_ = a.id
            if i in selected:
                continue
            c = selected + [(i,j,id_)]
            sent_str_combination = [_rm_blank(_rouge_clean(ranked_clusters[i].ariticles[j].text)) for idx_i, idx_j, idx_id in c]
            sent_date_combination = [ranked_clusters[i].ariticles[j].time for idx_i, idx_j, idx_id in c]
            rouge_1, rouge_2 = cal_rouge_tls_given(
                sent_str_combination,
                sent_date_combination,
                timelines,
                mode="cat",
                multi_tl=multi_tl,
            )
            tmp = [["%s-%s-%s" % (one.year, one.month, one.day) for one in one_tl.times] for one_tl in timelines]
            if not multi_tl:
                tmp = tmp[0]
            date_f1 = cal_date_f1(
                ["%s-%s-%s" % (one.year, one.month, one.day) for one in sent_date_combination], tmp, multi_tl=multi_tl
            )["f1"]
            print(rouge_1, rouge_2, date_f1)
            rouge_score = rouge_1 + rouge_2 + date_f1
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = (i,j,id_)
        if cur_id == -1:
            return selected, sents_str_list, sents_date_list
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected, key=lambda x:x[-1])


def independent_greedy_selection_tls(
    doc_sent_list,
    abstract_sent_list,
    doc_date_list,
    abstract_date_list,
    summary_size,
    random_size=20,
    multi_tl=False,
):
    if multi_tl:
        abstract_str_list = [
            [_rm_blank(_rouge_clean(" ".join(s))) for s in one_sent_list]
            for one_sent_list in abstract_sent_list
        ]
    else:
        abstract_str_list = [
            _rm_blank(_rouge_clean(" ".join(s))) for s in abstract_sent_list
        ]
    sents_str_list, sents_date_list = _process_source(doc_sent_list, doc_date_list)

    selected = []

    score_list = []
    min_size = min(summary_size, int(0.33 * len(sents_str_list)))
    for j in tqdm(range(len(sents_str_list) // 10 - 1)):
        c = [
            j * 10 + i
            for i in range(10)
            if len(_rm_blank(sents_str_list[j * 10 + i])) != 0
        ]
        sent_str_combination = [_rm_blank(sents_str_list[idx]) for idx in c]
        sent_date_combination = [sents_date_list[idx] for idx in c]
        rouge_1, rouge_2 = cal_rouge_tls(
            sent_str_combination,
            sent_date_combination,
            abstract_str_list,
            abstract_date_list,
            multi_tl=multi_tl,
        )
        date_f1 = cal_date_f1(
            sent_date_combination, abstract_date_list, multi_tl=multi_tl
        )["f1"]
        rouge_score = rouge_1 + rouge_2 + date_f1
        for idx in c:
            score_list.append((idx, rouge_score))

    score_list = sorted(score_list, key=lambda x: x[1], reverse=True)[:min_size]
    print(score_list)
    selected = [i for i, score in score_list]

    return sorted(selected), sents_str_list, sents_date_list


#%%


#%%
def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode("utf-8"))
    return h.hexdigest()


class BertData:
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        self.sep_vid = self.tokenizer.vocab["[SEP]"]
        self.cls_vid = self.tokenizer.vocab["[CLS]"]
        self.pad_vid = self.tokenizer.vocab["[PAD]"]

    def preprocess(self, src, tgt, oracle_ids):

        if len(src) == 0:
            return None

        original_src_txt = [" ".join(s) for s in src]

        labels = [0] * len(src)
        for l in oracle_ids:
            labels[l] = 1

        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens)]

        src = [src[i][: self.args.max_src_ntokens] for i in idxs]
        labels = [labels[i] for i in idxs]
        src = src[: self.args.max_nsents]
        labels = labels[: self.args.max_nsents]

        if len(src) < self.args.min_nsents:
            return None
        if len(labels) == 0:
            return None

        src_txt = [" ".join(sent) for sent in src]
        text = " [SEP] [CLS] ".join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)
        src_subtokens = src_subtokens[:510]
        src_subtokens = ["[CLS]"] + src_subtokens + ["[SEP]"]

        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if i % 2 == 0:
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        labels = labels[: len(cls_ids)]

        tgt_txt = "<q>".join([" ".join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]
        return (
            src_subtoken_idxs,
            labels,
            segments_ids,
            cls_ids,
            src_txt,
            tgt_txt,
        )

        # b_data = bert.preprocess_tls(
        #     ranked_clusters, ranked_dates, oracle_ids, multi_tl=args.multi_tl
        # )

    def preprocess_tls(self, ranked_clusters, ranked_dates, timelines, oracle_ids, multi_tl=False):
        # TODO: there may be a col as input, add the detail to attribute to the res
        # oracle_ids search, article

        if len(src) == 0:
            return None

        original_src_txt = []
        src = []
        src_date = []
        labels = []
        date_rank = []
        clust_rank = []
        idx_selected = [id_ for i,j,id_ in oracle_ids]
        for rank,c in enumerate(ranked_clusters):
            for a in c.articles:
                if a.id in idx_selected:
                    labels += [1] * len(a.sentences)
                else:
                    labels += [0] * len(a.sentences)
                clust_rank += [rank] * len(a.sentences)
                for s in a.sentences:
                    tmp_text = _rm_blank(_rouge_clean(s.raw))
                    original_src_txt += [tmp_text]
                    tmp_date = s.time if s.time else a.time
                    try:
                        rank2 = ranked_dates.index(date(tmp_date.year, tmp_date.month, tmp_date.day))
                    except:
                        rank2 = -1
                    date_rank += [rank2]
                    src_date+= ["%s-%s-%s" % (tmp_date.year, tmp_date.month, tmp_date.day)]

        tgt = []
        tgt_date = []
        for tl in timelines:
            one_tl_tgt = []
            one_tl_date = []
            for date_key, value_list in tl.date_to_summaries.items():
                one_tl_date += ["%s-%s-%s" % (date_key.year, date_key.month, date_key.day)]
                one_tl_tgt += value_list
            tgt.append(one_tl_tgt)
            tgt_date.append(one_tl_date)
        if len(tgt) == 1:
            tgt = tgt[0]
        if len(tgt_date) == 1:
            tgt = tgt_date[0]

        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens)]
        src = [src[i][: self.args.max_src_ntokens] for i in idxs]
        # TODO: truncate
        labels = [labels[i] for i in idxs]
        src = src[: self.args.max_nsents]
        labels = labels[: self.args.max_nsents]
        src_date = src_date[: self.args.max_nsents]

        if len(src) < self.args.min_nsents:
            return None
        if len(labels) == 0:
            return None

        src_txt = [" ".join(sent) for sent in src]
        text = " [SEP] [CLS] ".join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)
        # src_subtokens = src_subtokens[:510]
        if len(src_subtokens) < 510:
            src_subtokens += ["[PAD]"] * (510 - len(src_subtokens))
        else:
            truncate_point = (len(src_subtokens) // 512 + 1) * 512 - 2
            if truncate_point > len(src_subtokens):
                src_subtokens += ["[PAD]"] * (truncate_point - len(src_subtokens))
            else:
                src_subtokens = src_subtokens[:truncate_point]
        src_subtokens = ["[CLS]"] + src_subtokens + ["[SEP]"]

        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if i % 2 == 0:
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        labels = labels[: len(cls_ids)]
        if self.args.multi_tl:
            tgt_txt = "<t>".join(
                ["<q>".join([" ".join(tt) for tt in one_tl]) for one_tl in tgt]
            )
        else:
            tgt_txt = "<q>".join([" ".join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]
        return (
            src_subtoken_idxs,
            labels,
            segments_ids,
            cls_ids,
            src_txt,
            tgt_txt,
            src_date,
            tgt_date,
            idf_info,
            date_rank,
            clust_rank
        )


def format_to_bert(args):
    # if args.dataset != "":
    #     datasets = [args.dataset]
    # else:
    #     datasets = ["train", "valid", "test"]
    _format_to_bert_tls(args)
    # for corpus_type in datasets:
    #     a_lst = []
    #     for json_f in glob.glob(pjoin(args.raw_path, "*" + corpus_type + ".*.json")):
    #         real_name = json_f.split("/")[-1]
    #         if args.oracle_mode == "independent_greedy":
    #             real_name = (
    #                 real_name.split(".")[0] + "3." + ".".join(real_name.split(".")[1:])
    #             )
    #         a_lst.append(
    #             (
    #                 json_f,
    #                 args,
    #                 pjoin(args.save_path, real_name.replace("json", "bert.pt")),
    #             )
    #         )
    #         print(json_f)
    #         param = (
    #             json_f,
    #             args,
    #             pjoin(args.save_path, real_name.replace("json", "bert.pt")),
    #         )
    #         _format_to_bert_tls(param)
        # pool = Pool(args.n_cpus)
        # imap_fun = (
        #     _format_to_bert_tls
        #     if args.tls_mode in ["pretrain", "finetune"]
        #     else _format_to_bert
        # )
        # for d in pool.imap(imap_fun, a_lst):
        #     pass

        # pool.close()
        # pool.join()


def tokenize(args):
    stories_dir = os.path.abspath(args.raw_path)
    tokenized_stories_dir = os.path.abspath(args.save_path)

    print("Preparing to tokenize %s to %s..." % (stories_dir, tokenized_stories_dir))
    stories = os.listdir(stories_dir)
    # make IO list file
    print("Making list of files to tokenize...")
    with open("mapping_for_corenlp.txt", "w") as f:
        for s in stories:
            if not s.endswith("story"):
                continue
            f.write("%s\n" % (os.path.join(stories_dir, s)))
    command = [
        "java",
        "edu.stanford.nlp.pipeline.StanfordCoreNLP",
        "-annotators",
        "tokenize,ssplit",
        "-ssplit.newlineIsSentenceBreak",
        "always",
        "-filelist",
        "mapping_for_corenlp.txt",
        "-outputFormat",
        "json",
        "-outputDirectory",
        tokenized_stories_dir,
    ]
    print(
        "Tokenizing %i files in %s and saving in %s..."
        % (len(stories), stories_dir, tokenized_stories_dir)
    )
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping_for_corenlp.txt")

    # Check that the tokenized stories directory contains the same number of files as the original directory
    num_orig = len(os.listdir(stories_dir))
    num_tokenized = len(os.listdir(tokenized_stories_dir))
    if num_orig != num_tokenized:
        raise Exception(
            "The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?"
            % (tokenized_stories_dir, num_tokenized, stories_dir, num_orig)
        )
    print(
        "Successfully finished tokenizing %s to %s.\n"
        % (stories_dir, tokenized_stories_dir)
    )


def _format_to_bert(params):
    json_file, args, save_file = params
    # if os.path.exists(save_file):
    #     logger.info("Ignore %s" % save_file)
    #     return

    bert = BertData(args)

    logger.info("Processing %s" % json_file)
    jobs = json.load(open(json_file))
    datasets = []
    for d in jobs:
        source, tgt = d["src"], d["tgt"]
        if args.oracle_mode == "greedy":
            oracle_ids = greedy_selection(source, tgt, 3)
        elif args.oracle_mode == "combination":
            oracle_ids = combination_selection(source, tgt, 3)
        b_data = bert.preprocess(source, tgt, oracle_ids)
        if b_data is None:
            continue
        indexed_tokens, labels, segments_ids, cls_ids, src_txt, tgt_txt = b_data
        b_data_dict = {
            "src": indexed_tokens,
            "labels": labels,
            "segs": segments_ids,
            "clss": cls_ids,
            "src_txt": src_txt,
            "tgt_txt": tgt_txt,
        }
        datasets.append(b_data_dict)
    logger.info("Saving to %s" % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


import news_tls.data import Dataset
from news_tls.datewise import MentionCountDateRanker, PM_Mean_SentenceCollector
from news_tls.clust import TemporalMarkovClusterer, ClusterDateMentionCountRanker
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import date

def _format_to_bert_tls(args):
    # json_file, args, save_file = params
    # if os.path.exists(save_file):
    #     logger.info("Ignore %s" % save_file)
    #     return

    # TODO: there need two params
    # dataset_path = /data1/su/app/text_forecast/data/datasets/entities/
    
    bert = BertData(args)

    # logger.info("Processing %s" % json_file)
    # jobs = json.load(open(json_file))
    jobs = Dataset(args.dataset_path)

    
    datasets = []
    for d in jobs.collections:
        times=[]
        fit_input_sent = []
        fit_input_doc = []
        for a in d.articles():
            times.append(a.time)
            tmp = []
            for s in a.sentences:
                tmp.append(s.raw)
            fit_input_sent += tmp
            fit_input_doc.append(" ".join(tmp))
        d.start = min(times)
        d.end = max(times)

        sent_vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
        doc_vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
        sent_vectorizer.fit(fit_input_sent)
        doc_vectorizer.fit(fit_input_doc)

        date_ranker = MentionCountDateRanker()
        ranked_dates = date_ranker.rank_dates(d)
        
        clusterer = TemporalMarkovClusterer() 
        cluster_ranker = ClusterDateMentionCountRanker()
        clusters = clusterer.cluster(col, doc_vectorizer)
        ranked_clusters = cluster_ranker.rank(clusters, d)

        if args.tls_mode == "pretrain":
            source, sents_date_list = d["src"], d["time"]
            page, taxo = d["page"], d["taxo"]
            tgt, tgt_date, oracle_ids = weak_supervision_selection(
                source, sents_date_list, page, taxo, args.tgt_size
            )
        elif args.tls_mode == "finetune":
            if args.oracle_mode == "greedy":
                oracle_ids, sents_str_list, sents_date_list = greedy_selection_tls(
                    source, tgt, source_date, tgt_date, 3, multi_tl=args.multi_tl
                )
            elif args.oracle_mode == "combination":
                oracle_ids, sents_str_list, sents_date_list = combination_selection_tls(
                    source, tgt, source_date, tgt_date, len(tgt), multi_tl=args.multi_tl
                )
            elif args.oracle_mode == "random_greedy":
                oracle_ids = random_greed_selection_tls(
                    ranked_clusters,
                    d.timelines,
                    summary_size=10,
                    random_size=20,
                    multi_tl=args.multi_tl,
                )
            elif args.oracle_mode == "independent_greedy":
                (
                    oracle_ids,
                    sents_str_list,
                    sents_date_list,
                ) = independent_greedy_selection_tls(
                    source,
                    tgt,
                    source_date,
                    tgt_date,
                    summary_size=args.tgt_size,
                    random_size=10,
                    multi_tl=args.multi_tl,
                )

            source = _sent2token(sents_str_list) # 其实是在分词
        if len(source) == 0:
            continue
        # print("source", source)
        # TODO: add tf-idf matrix and make it into bertdata, notice should add the last in the data loader and change the i
        b_data = bert.preprocess_tls(
            ranked_clusters, ranked_dates, d.timelines, oracle_ids, multi_tl=args.multi_tl
        )
        if b_data is None:
            continue
        (
            indexed_tokens,
            labels,
            segments_ids,
            cls_ids,
            src_txt,
            tgt_txt,
            src_date,
            tgt_date,
            idf_info,
            date_rank,
            clust_rank
        ) = b_data
        print(labels)
        b_data_dict = {
            "src": indexed_tokens,
            "labels": labels,
            "segs": segments_ids,
            "clss": cls_ids,
            "src_txt": src_txt,
            "tgt_txt": tgt_txt,
            "src_date": src_date,
            "tgt_date": tgt_date,
            "idf_info": idf_info,
            "date_rank": date_rank,
            "clust_rank": clust_rank
        }
        datasets.append(b_data_dict)
    logger.info("Saving to %s" % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


def format_to_lines(args):
    corpus_mapping = {}
    for corpus_type in ["valid", "test", "train"]:
        temp = []
        for line in open(pjoin(args.map_path, "mapping_" + corpus_type + ".txt")):
            temp.append(hashhex(line.strip()))
        corpus_mapping[corpus_type] = {key.strip(): 1 for key in temp}
    train_files, valid_files, test_files = [], [], []
    for f in glob.glob(pjoin(args.raw_path, "*.json")):
        real_name = f.split("/")[-1].split(".")[0]
        if real_name in corpus_mapping["valid"]:
            valid_files.append(f)
        elif real_name in corpus_mapping["test"]:
            test_files.append(f)
        elif real_name in corpus_mapping["train"]:
            train_files.append(f)

    corpora = {"train": train_files, "valid": valid_files, "test": test_files}
    for corpus_type in ["train", "valid", "test"]:
        a_lst = [(f, args) for f in corpora[corpus_type]]
        pool = Pool(args.n_cpus)
        dataset = []
        p_ct = 0
        for d in pool.imap_unordered(_format_to_lines, a_lst):
            dataset.append(d)
            if len(dataset) > args.shard_size:
                pt_file = "{:s}.{:s}.{:d}.json".format(
                    args.save_path, corpus_type, p_ct
                )
                with open(pt_file, "w") as save:
                    # save.write('\n'.join(dataset))
                    save.write(json.dumps(dataset))
                    p_ct += 1
                    dataset = []

        pool.close()
        pool.join()
        if len(dataset) > 0:
            pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
            with open(pt_file, "w") as save:
                # save.write('\n'.join(dataset))
                save.write(json.dumps(dataset))
                p_ct += 1
                dataset = []


def _format_to_lines(params):
    f, args = params
    print(f)
    source, tgt = load_json(f, args.lower)
    return {"src": source, "tgt": tgt}
