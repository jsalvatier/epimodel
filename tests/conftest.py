from pathlib import Path

import pytest

import epimodel


@pytest.fixture
def datadir(request):
    p = Path(request.module.__file__).parent / "data"
    if p.exists():
        return p
    p = Path(request.module.__file__).parent.parent / "data"
    if p.exists():
        return p
    raise Exception("test data dir not found")


@pytest.fixture
def regions(datadir):
    return epimodel.RegionDataset.load(datadir / "regions.csv")
