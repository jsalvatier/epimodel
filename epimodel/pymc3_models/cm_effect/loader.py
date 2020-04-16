### Initial imports

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import math

from ... import RegionDataset, read_csv

log = logging.getLogger(__name__)


class Loader:
    def __init__(
        self,
        start,
        end,
        regions,
        CMs,
        data_dir=None,
        active_cm_file="countermeasures-model-0to1.csv",
        a=False,
    ):
        if data_dir is None:
            data_dir = Path(__file__).parents[3] / "data"
        self.data_dir = data_dir

        # Days
        self.Ds = pd.date_range(start=start, end=end, tz="utc")

        # CM features
        self.CMs = list(CMs)

        # Countries / regions
        self.Rs = list(regions)

        self.rds = RegionDataset.load(self.data_dir / "regions.csv")

        # Raw data, never modified
        self.johns_hopkins = read_csv(self.data_dir / "johns-hopkins.csv")
        self.features = read_csv(self.data_dir / active_cm_file)

        self.TheanoType = "float64"

        self.Confirmed = None
        self.ConfirmedCutoff = 10.0
        self.Deaths = None
        self.DeathsCutoff = 10.0
        self.Active = None
        self.ActiveCutoff = 10.0
        self.Recovered = None
        self.RecoveredCutoff = 10.0

        self.ActiveCMs = None

        self.update()

    def update(self):
        """(Re)compute the values used in the model after any parameter/region/etc changes."""

        def prep(name, cutoff=None):
            v = (
                self.johns_hopkins[name]
                .astype(self.TheanoType)
                .unstack(1)
                .reindex(index=self.Rs, columns=self.Ds)
                .values
            )
            assert v.shape == (len(self.Rs), len(self.Ds))
            if cutoff is not None:
                v[v < cutoff] = np.nan
            return np.ma.masked_invalid(v)

        self.Confirmed = prep("Confirmed", self.ConfirmedCutoff)
        self.Deaths = prep("Deaths", self.DeathsCutoff)
        self.Recovered = prep("Recovered", self.RecoveredCutoff)
        self.Active = prep("Active", self.ActiveCutoff)

        self.ActiveCMs = self.get_ActiveCMs(self.Ds)

    def get_ActiveCMs(self, dates):
        ActiveCMs = np.stack(
            [
                self.features.loc[rc]
                .astype(self.TheanoType)
                .reindex(index=dates, method='pad')
                .reindex(columns=self.CMs)
                .values.T
                for rc in self.Rs
            ]
        )
        assert ActiveCMs.shape == (len(self.Rs), len(self.CMs), len(dates))
        # [region, CM, day] Which CMs are active, and to what extent
        return ActiveCMs

    def print_stats(self):
        """Print data stats, plot graphs, ..."""

        print("\nCountermeasures                            min   .. mean  .. max")
        for i, cm in enumerate(self.CMs):
            vals = np.array(self.features[cm])
            print(
                f"{i:2} {cm:42} {vals.min():.3f} .. {vals.mean():.3f}"
                f" .. {vals.max():.3f}"
                f"  {set(vals) if len(set(vals)) <= 4 else ''}"
            )

    def create_delay_dist(self, delay_mean):
        """
        Generate and return CMDelayProb and CMDelayCut.
        """
        # Poisson distribution
        CMDelayProb = np.array(
            [
                delay_mean ** k * np.exp(-delay_mean) / math.factorial(k)
                for k in range(100)
            ]
        )
        assert abs(sum(CMDelayProb) - 1.0) < 1e-3

        # Shorten the distribution to have >99% of the mass
        CMDelayProb = CMDelayProb[np.cumsum(CMDelayProb) <= 0.999]
        # Cut off first days to have 90% of pre-existing intervention effect
        CMDelayCut = sum(np.cumsum(CMDelayProb) < 0.9)
        log.debug(
            f"CM delay: mean {np.sum(CMDelayProb * np.arange(len(CMDelayProb)))}, "
            f"len {len(CMDelayProb)}, cut at {CMDelayCut}"
        )
        return CMDelayProb, CMDelayCut

    def plot_cm_correlation(self, delay_mean=7.0):
        delay_dist, _ = self.create_delay_dist(delay_mean)
        dcs = {}
        for cmi, cm in enumerate(self.CMs):
            dcs[cm] = []
            for ri in range(len(self.Rs)):
                dcs[cm].extend(np.convolve(self.ActiveCMs[ri, cmi, :], delay_dist))
        corr = pd.DataFrame(dcs).corr()
        mask = np.triu(np.ones_like(corr, dtype=np.bool))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(
            corr,
            mask=mask,
            center=0,
            annot=True,
            square=True,
            cmap=cmap,
            linewidths=0.5,
            cbar_kws={"shrink": 0.5},
        )

    def filter_regions(
        self, regions, min_feature_sum=1.0, min_final_jh=400, jh_col="Confirmed"
    ):
        """Filter and return list of region codes."""
        res = []
        for rc in regions:
            r = self.rds[rc]
            if rc in self.johns_hopkins.index and rc in self.features_0to1.index:
                if self.johns_hopkins.loc[(rc, self.Ds[-1]), jh_col] < min_final_jh:
                    print(f"Region {r} has <{min_final_jh} final JH col {jh_col}")
                    continue
                # TODO: filter by features?
                # if self.active_features.loc[(rc, self.Ds)] ...
                #    print(f"Region {r} has <{min_final_jh} final JH col {jh_col}")
                #    continue
                res.append(rc)
        return res


def split_0to1_features(features_0to1, exclusive=False):
    """
    Split joined features in model-0to1 into separate bool features.

    If `exclusive`, only one of a chain of features is activated.
    Otherwise all up to the active level are active.
    Resulting DF is returned.
    """
    fs = {}
    f01 = features_0to1

    fs["Masks over 60"] = f01["Mask wearing"] >= 60

    fs["Asymptomatic contact isolation"] = f01["Asymptomatic contact isolation"]

    fs["Gatherings limited to 10"] = f01["Gatherings limited to"] > 0.84
    fs["Gatherings limited to 100"] = f01["Gatherings limited to"] > 0.35
    fs["Gatherings limited to 1000"] = f01["Gatherings limited to"] > 0.05
    if exclusive:
        fs["Gatherings limited to 1000"] &= ~fs["Gatherings limited to 100"]
        fs["Gatherings limited to 100"] &= ~fs["Gatherings limited to 10"]

    fs["Business suspended - some"] = f01["Business suspended"] > 0.1
    fs["Business suspended - many"] = f01["Business suspended"] > 0.6
    if exclusive:
        fs["Business suspended - some"] &= ~fs["Business suspended - many"]

    fs["Schools and universities closed"] = f01["Schools and universities closed"]

    fs["Distancing and hygiene over 0.2"] = (
        f01["Minor distancing and hygiene measures"] > 0.2
    )

    fs["General curfew - permissive"] = f01["General curfew"] > 0.1
    fs["General curfew - strict"] = f01["General curfew"] > 0.6
    if exclusive:
        fs["General curfew - permissive"] &= ~fs["General curfew - strict"]

    fs["Healthcare specialisation over 0.2"] = f01["Healthcare specialisation"] > 0.2

    fs["Phone line"] = f01["Phone line"]

    return pd.DataFrame(fs).astype("f4")
