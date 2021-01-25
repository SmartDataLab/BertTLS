from others.utils import _rouge_clean, _process_source
import json
import pytest  # 引入pytest包


def test_process_sents_one_date():
    d = json.load(
        open(
            "/data1/su/app/text_forecast/data/datasets/labeldata_new_structure/entities_train.0.json"
        )
    )
    count = 0
    for x in d:
        sent_str_list = x["src"]
        src_date_list = x["src_date"]
        new_sent_str_list, _ = _process_source(sent_str_list, src_date_list)
        for sent in new_sent_str_list:
            if len(sent.split()) > 100:
                count += 1
    print(count)
    if count > 20:
        assert 0
    print("pass")
    assert 1


test_process_sents_one_date()