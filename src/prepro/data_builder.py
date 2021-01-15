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

from others.logging import logger
from others.utils import clean
from prepro.utils import _get_word_ngrams

import datetime
from tilse.data import timelines
from tilse.evaluation import rouge


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
    doc_sent_list, doc_date_list, doc_page_list, doc_taxo_list, abstract_size
):
    # simple method sort first
    sort_tuple = sorted(
        tuple(zip(doc_page_list, range(len(doc_page_list)))), key=lambda x: x[0]
    )
    # print(sort_tuple)
    abstract = []
    abstract_date = []
    oracle_ids = []
    for i in range(min(abstract_size, len(sort_tuple))):
        idx = sort_tuple[i][1]
        oracle_ids.append(idx)
        abstract.append(doc_sent_list[idx])
        abstract_date.append(doc_date_list[idx])
    return abstract, abstract_date, oracle_ids


def datestr2datetime(date_str):
    try:
        date_str = date_str.split("T")[0]
        return datetime.date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
    except Exception as e:
        print("date parse fail")
        return datetime.date(1970, 1, 1)


def cal_rouge_tls(doc_sent_list, doc_date_list, abstract_sent_list, abstract_date_list):
    """
    docstring
    """

    evaluator = rouge.TimelineRougeEvaluator(measures=["rouge_1", "rouge_2"])
    doc_datetime_list = [datestr2datetime(date_str) for date_str in doc_date_list]
    predicted_timeline = timelines.Timeline(dict(zip(doc_datetime_list, doc_sent_list)))
    groundtruth = timelines.GroundTruth(
        [timelines.Timeline(dict(zip(doc_datetime_list, doc_sent_list)))]
    )
    res_dict = evaluator.evaluate_concat(predicted_timeline, groundtruth)
    return res_dict["rouge_1"]["f_score"], res_dict["rouge_2"]["f_score"]


def combination_selection_tls(
    doc_sent_list, abstract_sent_list, doc_date_list, abstract_date_list, summary_size
):
    def _rouge_clean(s):
        return re.sub(r"[^a-zA-Z0-9 ]", "", s)

    max_rouge = 0.0
    max_idx = (0, 0)
    abstract_str_list = [_rouge_clean(" ".join(s)) for s in abstract_sent_list]
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(" ".join(abstract)).split()
    sent_str_list = [_rouge_clean(" ".join(s)) for s in doc_sent_list]
    sents = [s.split() for s in sent_str_list]

    # evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    # reference_1grams = _get_word_ngrams(1, [abstract])
    # evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    # reference_2grams = _get_word_ngrams(2, [abstract])

    impossible_sents = []
    for s in range(summary_size + 1):
        combinations = itertools.combinations(
            [i for i in range(len(sents)) if i not in impossible_sents], s + 1
        )
        for c in combinations:
            # candidates_1 = [evaluated_1grams[idx] for idx in c]
            # candidates_1 = set.union(*map(set, candidates_1))
            # candidates_2 = [evaluated_2grams[idx] for idx in c]
            # candidates_2 = set.union(*map(set, candidates_2))
            sent_str_combination = [sent_str_list[idx] for idx in c]
            # rouge_1 = cal_rouge(candidates_1, reference_1grams)["f"]
            # rouge_2 = cal_rouge(candidates_2, reference_2grams)["f"]
            rouge_1, rouge_2 = cal_rouge_tls(
                sent_str_combination,
                doc_date_list,
                abstract_str_list,
                abstract_date_list,
            )

            rouge_score = rouge_1 + rouge_2
            if s == 0 and rouge_score == 0:
                impossible_sents.append(c[0])
            if rouge_score > max_rouge:
                max_idx = c
                max_rouge = rouge_score
    return sorted(list(max_idx))


def greedy_selection_tls(
    doc_sent_list, doc_date_list, abstract_sent_list, abstract_date_list, summary_size
):
    def _rouge_clean(s):
        return re.sub(r"[^a-zA-Z0-9 ]", "", s)

    max_rouge = 0.0
    abstract_str_list = [_rouge_clean(" ".join(s)) for s in abstract_sent_list]
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(" ".join(abstract)).split()
    sent_str_list = [_rouge_clean(" ".join(s)) for s in doc_sent_list]
    sents = [s.split() for s in sent_str_list]
    # evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    # reference_1grams = _get_word_ngrams(1, [abstract])
    # evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    # reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if i in selected:
                continue
            c = selected + [i]
            sent_str_combination = [sent_str_list[idx] for idx in c]
            # candidates_1 = [evaluated_1grams[idx] for idx in c]
            # candidates_1 = set.union(*map(set, candidates_1))
            # candidates_2 = [evaluated_2grams[idx] for idx in c]
            # candidates_2 = set.union(*map(set, candidates_2))
            # rouge_1 = cal_rouge(candidates_1, reference_1grams)["f"]
            # rouge_2 = cal_rouge(candidates_2, reference_2grams)["f"]
            rouge_1, rouge_2 = cal_rouge_tls(
                sent_str_combination,
                doc_date_list,
                abstract_str_list,
                abstract_date_list,
            )
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if cur_id == -1:
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)


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
        # text = [' '.join(ex['src_txt'][i].split()[:self.args.max_src_ntokens]) for i in idxs]
        # text = [_clean(t) for t in text]
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

    def preprocess_tls(self, src, tgt, src_date, tgt_date, oracle_ids):

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
        src_date = src_date[: self.args.max_nsents]

        if len(src) < self.args.min_nsents:
            return None
        if len(labels) == 0:
            return None

        src_txt = [" ".join(sent) for sent in src]
        # text = [' '.join(ex['src_txt'][i].split()[:self.args.max_src_ntokens]) for i in idxs]
        # text = [_clean(t) for t in text]
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
            src_date,
            tgt_date,
        )


def format_to_bert(args):
    if args.dataset != "":
        datasets = [args.dataset]
    else:
        datasets = ["train", "valid", "test"]
    for corpus_type in datasets:
        a_lst = []
        for json_f in glob.glob(pjoin(args.raw_path, "*" + corpus_type + ".*.json")):
            real_name = json_f.split("/")[-1]
            a_lst.append(
                (
                    json_f,
                    args,
                    pjoin(args.save_path, real_name.replace("json", "bert.pt")),
                )
            )
        print(a_lst)
        param = (
            json_f,
            args,
            pjoin(args.save_path, real_name.replace("json", "bert.pt")),
        )
        _format_to_bert_tls(param)
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
    if os.path.exists(save_file):
        logger.info("Ignore %s" % save_file)
        return

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


def _format_to_bert_tls(params):
    json_file, args, save_file = params
    if os.path.exists(save_file):
        logger.info("Ignore %s" % save_file)
        return

    bert = BertData(args)

    logger.info("Processing %s" % json_file)
    jobs = json.load(open(json_file))
    datasets = []
    for d in jobs:
        if args.tls_mode == "pretrain":
            source, source_date = d["src"], d["time"]
            page, taxo = d["page"], d["taxo"]
            # doc_sent_list, doc_date_list, doc_page_list, doc_taxo_list, abstract_size
            tgt, tgt_date, oracle_ids = weak_supervision_selection(
                source, source_date, page, taxo, 3
            )
        elif args.tls_mode == "finetune":
            source, source_date, tgt, tgt_date = (
                d["src"],
                d["src_time"],
                d["tgt"],
                d["tgt_time"],
            )
            if args.oracle_mode == "greedy":
                oracle_ids = greedy_selection_tls(source, tgt, source_date, tgt_date, 3)
            elif args.oracle_mode == "combination":
                oracle_ids = combination_selection_tls(
                    source, tgt, source_date, tgt_date, 3
                )

        b_data = bert.preprocess_tls(source, tgt, source_date, tgt_date, oracle_ids)
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
        ) = b_data
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
