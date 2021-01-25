import collections
from others.utils import cal_date_f1
import pytest  # 引入pytest包

# TODO(sujinhua): ImportError: attempted relative import with no known parent package


def test_count():  # test开头的测试函数
    pred_date = ["2017-12-09T20", "2017-12-09", "2017-12-10", "2017-12-12"]
    ref_date = [
        "2017-12-09 20:10",
        "2017-12-09",
        "2017-12-09",
        "2017-12-11",
        "2017-12-12",
    ]
    res = cal_date_f1(pred_date, ref_date)
    if (
        res["p"] == 3 / 4
        and res["r"] == 3 / 5
        and res["f1"] == 1 / (0.5 * (4 / 3 + 5 / 3))
    ):
        print("pass")
        assert 1  # 断言成功
    else:
        assert 0


test_count()
# if __name__ == "__main__":
#     pytest.main("-s  date_f1_test.py")  # 调用pytest的main函数执行测试
