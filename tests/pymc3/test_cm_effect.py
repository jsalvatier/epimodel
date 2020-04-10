import pytest
from pytest import approx
import numpy as np

theano = pytest.importorskip("theano")
pm = pytest.importorskip("pymc3")

from epimodel.pymc3_models import utils, cm_effect

CMs = [
    "Business suspended - many",
    "Schools and universities closed",
    "General curfew - permissive",
]

Rs = ["CZ", "DE", "FR", "GB"]


def test_loader(datadir):
    data = cm_effect.Loader(
        "2020-02-10",
        "2020-02-25",
        Rs,
        CMs,
        data_dir=datadir,
        active_cm_file="countermeasures-model-0to1-split.csv",
    )
    data.print_stats()


def test_modelv1(datadir):
    data = cm_effect.Loader(
        "2020-02-10",
        "2020-02-25",
        Rs,
        CMs,
        data_dir=datadir,
        active_cm_file="countermeasures-model-0to1-split.csv",
    )
    with cm_effect.CMModelV1(data, 5.0) as model:
        model.build()
    print(model.check_test_point())


def test_modelv2(datadir):
    data = cm_effect.Loader(
        "2020-02-10",
        "2020-02-25",
        Rs,
        CMs,
        data_dir=datadir,
        active_cm_file="countermeasures-model-0to1-split.csv",
    )
    with cm_effect.CMModelV2(data, 35.0) as model:
        model.build()
    print(model.check_test_point())
