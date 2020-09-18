"""
:code:`models.py`

Contains a variety of models of NPI effectiveness, all subclassed from BaseCMModel. 
"""
import numpy as np
import pymc3 as pm

import theano.tensor as T
import theano.tensor as tt
from pymc3.theanof import floatX, intX, take_along_axis
from pymc3.distributions.dist_math import bound, factln, binomln, betaln, logpow, random_choice
from pymc3.distributions.distribution import Discrete, draw_values, generate_samples
import theano.tensor.signal.conv as C
import scipy.stats as stats

from epimodel import EpidemiologicalParameters
from .base_model import BaseCMModel


class DefaultModel(BaseCMModel):
    """
    Default Model

    Default EpidemicForecasting.org NPI effectiveness model.
    Please see also https://www.medrxiv.org/content/10.1101/2020.05.28.20116129v3
    """

    def __init__(self, data, cm_plot_style=None, name="", model=None):
        """
        Initialiser.
        """
        super().__init__(data, cm_plot_style, name, model)

    def build_model(self, R_prior_mean=3.28, cm_prior_scale=10, cm_prior='skewed',
                    gi_mean_mean=5, gi_mean_sd=1, gi_sd_mean=2, gi_sd_sd=2, growth_noise_scale=0.2,
                    deaths_delay_mean_mean=21, deaths_delay_mean_sd=1, deaths_delay_disp_mean=9, deaths_delay_disp_sd=1,
                    cases_delay_mean_mean=10, cases_delay_mean_sd=1, cases_delay_disp_mean=5, cases_delay_disp_sd=1,
                    deaths_truncation=48, cases_truncation=32):
        """
        Build NPI effectiveness model

        :param R_prior_mean: R_0 prior mean
        :param cm_prior_scale: NPI effectiveness prior scale
        :param cm_prior: NPI effectiveness prior type. Either 'normal', 'icl' or skewed (asymmetric laplace)
        :param gi_mean_mean: mean of normal prior placed over the generation interval mean
        :param gi_mean_sd: sd of normal prior placed over the generation interval mean
        :param gi_sd_mean: mean of normal prior placed over the generation interval sd
        :param gi_sd_sd: sd of normal prior placed over the generation interval sd
        :param growth_noise_scale: growth noise scale
        :param deaths_delay_mean_mean: mean of normal prior placed over death delay mean
        :param deaths_delay_mean_sd: sd of normal prior placed over death delay mean
        :param deaths_delay_disp_mean: mean of normal prior placed over death delay dispersion (alpha / psi)
        :param deaths_delay_disp_sd: sd of normal prior placed over death delay dispersion (alpha / psi)
        :param cases_delay_mean_mean: mean of normal prior placed over cases delay mean
        :param cases_delay_mean_sd: sd of normal prior placed over cases delay mean
        :param cases_delay_disp_mean: mean of normal prior placed over cases delay dispersion
        :param cases_delay_disp_sd: sd of normal prior placed over cases delay dispersion
        :param deaths_truncation: maximum death delay
        :param cases_truncation: maximum reporting delay
        """
        with self.model:
            # build NPI Effectiveness priors
            self.build_npi_prior(cm_prior, cm_prior_scale)

            self.CMReduction = pm.Deterministic("CMReduction", T.exp((-1.0) * self.CM_Alpha))

            # build R_0 prior
            self.HyperRVar = pm.HalfNormal(
                "HyperRVar", sigma=0.5
            )

            self.RegionR_noise = pm.Normal("RegionLogR_noise", 0, 1, shape=(self.nRs))
            self.RegionR = pm.Deterministic("RegionR", R_prior_mean + self.RegionLogR_noise * self.HyperRVar)

            # load CMs active, compute log-R reduction and region log-R based on NPIs active
            self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs)

            self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs, 1))
                    * self.ActiveCMs
            )

            self.LogRReduction = T.sum(self.ActiveCMReduction, axis=1)

            self.ExpectedLogR = T.reshape(pm.math.log(self.RegionR), (self.nRs, 1)) - self.LogRReduction

            # convert R into growth rates
            if gi_mean_sd > 0:
                self.GI_mean = pm.TruncatedNormal('GI_mean', gi_mean_mean, gi_mean_sd, lower=0.01)
            else:
                print('Using a fixed value for the generation interval mean')
                self.GI_mean = gi_mean_mean

            self.GI_sd = pm.TruncatedNormal('GI_sd', gi_sd_mean, gi_sd_sd, lower=0.01)

            gi_beta = self.GI_mean / self.GI_sd ** 2
            gi_alpha = self.GI_mean ** 2 / self.GI_sd ** 2

            self.ExpectedGrowth = gi_beta * (pm.math.exp(self.ExpectedLogR / gi_alpha) - T.ones((self.nRs, self.nDs)))

            # exclude 40 days of noise, slight increase in runtime.
            self.GrowthCasesNoise = pm.Normal("GrowthCasesNoise", 0, growth_noise_scale,
                                              shape=(self.nRs, self.nDs - 40))
            self.GrowthDeathsNoise = pm.Normal("GrowthDeathsNoise", 0, growth_noise_scale,
                                               shape=(self.nRs, self.nDs - 40))

            self.GrowthCases = T.inc_subtensor(self.ExpectedGrowth[:, 30:-10], self.GrowthCasesNoise)
            self.GrowthDeaths = T.inc_subtensor(self.ExpectedGrowth[:, 30:-10], self.GrowthDeathsNoise)

            self.PsiCases = pm.HalfNormal('PsiCases', 5.)
            self.PsiDeaths = pm.HalfNormal('PsiDeaths', 5.)

            # Confirmed Cases
            # seed and produce daily infections which become confirmed cases
            self.InitialSizeCases_log = pm.Normal("InitialSizeCases_log", 0, 50, shape=(self.nRs, 1))
            self.InfectedCases = pm.Deterministic("InfectedCases", pm.math.exp(
                self.InitialSizeCases_log + self.GrowthCases.cumsum(axis=1)))

            if cases_delay_mean_sd > 0:
                self.CasesDelayMean = pm.Normal('CasesDelayMean', cases_delay_mean_mean, cases_delay_mean_sd)
            else:
                print('Using a fixed value for the reporting delay mean')
                self.CasesDelayMean = cases_delay_mean_mean

            self.CasesDelayDisp = pm.Normal('CasesDelayDisp', cases_delay_disp_mean, cases_delay_disp_sd)
            cases_delay_dist = NegativeBinomialCust.dist(mu=self.CasesDelayMean, alpha=self.CasesDelayDisp)
            bins = np.arange(0, cases_truncation)
            pmf = T.exp(cases_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            reporting_delay = pmf.reshape((1, cases_truncation))

            # convolve with delay to produce expectations
            expected_cases = C.conv2d(
                self.InfectedCases,
                reporting_delay,
                border_mode="full"
            )[:, :self.nDs]

            self.ExpectedCases = pm.Deterministic("ExpectedCases", expected_cases.reshape(
                (self.nRs, self.nDs)))

            # effectively handle missing values ourselves
            # output distribution
            self.ObservedCases = NegativeBinomialCust(
                "ObservedCases",
                mu=self.ExpectedCases.reshape((self.nRs * self.nDs,))[self.all_observed_active],
                alpha=self.PsiCases,
                shape=(len(self.all_observed_active),),
                observed=self.d.NewCases.data.reshape((self.nRs * self.nDs,))[self.all_observed_active]
            )

            # Deaths
            # seed and produce daily infections which become confirmed cases
            self.InitialSizeDeaths_log = pm.Normal("InitialSizeDeaths_log", 0, 50, shape=(self.nRs, 1))
            self.InfectedDeaths = pm.Deterministic("InfectedDeaths", pm.math.exp(
                self.InitialSizeDeaths_log + self.GrowthDeaths.cumsum(axis=1)))

            if deaths_delay_mean_sd > 0:
                self.DeathsDelayMean = pm.Normal('DeathsDelayMean', deaths_delay_mean_mean,
                                                          deaths_delay_mean_sd)
            else:
                print('Using a fixed value for the fatality delay mean')
                self.DeathsDelayMean = deaths_delay_mean_mean

            self.DeathsDelayDisp = pm.Normal('DeathsDelayDisp', deaths_delay_disp_mean, deaths_delay_disp_sd)
            deaths_delay_dist = NegativeBinomialCust.dist(mu=self.DeathsDelayMean, alpha=self.DeathsDelayDisp)
            bins = np.arange(0, deaths_truncation)
            pmf = T.exp(deaths_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            fatality_delay = pmf.reshape((1, deaths_truncation))

            # convolve with delay to production reports
            expected_deaths = C.conv2d(
                self.InfectedDeaths,
                fatality_delay,
                border_mode="full"
            )[:, :self.nDs]

            self.ExpectedDeaths = pm.Deterministic("ExpectedDeaths", expected_deaths.reshape(
                (self.nRs, self.nDs)))

            # effectively handle missing values ourselves
            # death output distribution
            self.ObservedDeaths = NegativeBinomialCust(
                "ObservedDeaths",
                mu=self.ExpectedDeaths.reshape((self.nRs * self.nDs,))[self.all_observed_deaths],
                alpha=self.PsiDeaths,
                shape=(len(self.all_observed_deaths),),
                observed=self.d.NewDeaths.data.reshape((self.nRs * self.nDs,))[self.all_observed_deaths]
            )


class DeathsOnlyModel(BaseCMModel):
    """
    Deaths only model.

    Identical to the default model, other than modelling only deaths.
    """

    def build_model(self, R_prior_mean=3.28, cm_prior_scale=10, cm_prior='skewed',
                    gi_mean_mean=5, gi_mean_sd=1, gi_sd_mean=2, gi_sd_sd=2, growth_noise_scale=0.2,
                    deaths_delay_mean_mean=21, deaths_delay_mean_sd=1, deaths_delay_disp_mean=9, deaths_delay_disp_sd=1,
                    deaths_truncation=48, **kwargs):
        """
        Build PyMC3 model.

        :param R_prior_mean: R_0 prior mean
        :param cm_prior_scale: NPI effectiveness prior scale
        :param cm_prior: NPI effectiveness prior type. Either 'normal', 'icl' or skewed (asymmetric laplace)
        :param gi_mean_mean: mean of normal prior placed over the generation interval mean
        :param gi_mean_sd: sd of normal prior placed over the generation interval mean
        :param gi_sd_mean: mean of normal prior placed over the generation interval sd
        :param gi_sd_sd: sd of normal prior placed over the generation interval sd
        :param growth_noise_scale: growth noise scale
        :param deaths_delay_mean_mean: mean of normal prior placed over death delay mean
        :param deaths_delay_mean_sd: sd of normal prior placed over death delay mean
        :param deaths_delay_disp_mean: mean of normal prior placed over death delay dispersion (alpha / psi)
        :param deaths_delay_disp_sd: sd of normal prior placed over death delay dispersion (alpha / psi)
        :param deaths_truncation: maximum death delay
        """

        for key, _ in kwargs.items():
            print(f'Argument: {key} not being used')

        with self.model:
            # build NPI Effectiveness priors
            self.build_npi_prior(cm_prior, cm_prior_scale)

            self.CMReduction = pm.Deterministic('CMReduction', T.exp((-1.0) * self.CM_Alpha))

            self.HyperRVar = pm.HalfNormal(
                'HyperRVar', sigma=0.5
            )

            self.RegionR_noise = pm.Normal('RegionLogR_noise', 0, 1, shape=(self.nRs), )
            self.RegionR = pm.Deterministic('RegionR', R_prior_mean + self.RegionLogR_noise * self.HyperRVar)

            self.ActiveCMs = pm.Data('ActiveCMs', self.d.ActiveCMs)

            self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs, 1))
                    * self.ActiveCMs
            )

            self.GrowthReduction = T.sum(self.ActiveCMReduction, axis=1)

            self.ExpectedLogR = pm.Deterministic(
                'ExpectedLogR',
                T.reshape(pm.math.log(self.RegionR), (self.nRs, 1)) - self.GrowthReduction
            )

            # convert R into growth rates
            self.GI_mean = pm.Normal('GI_mean', gi_mean_mean, gi_mean_sd)
            self.GI_sd = pm.Normal('GI_sd', gi_sd_mean, gi_sd_sd)

            gi_beta = self.GI_mean / self.GI_sd ** 2
            gi_alpha = self.GI_mean ** 2 / self.GI_sd ** 2

            self.ExpectedGrowth = gi_beta * (np.exp(self.ExpectedLogR / gi_alpha) - T.ones_like(
                self.ExpectedLogR))

            self.GrowthNoise = pm.Normal('Growth', 0, growth_noise_scale, shape=(self.nRs, self.nDs - 40))

            self.Growth = T.inc_subtensor(self.ExpectedGrowth[:, 30:-10], self.GrowthNoise)

            self.InitialSize_log = pm.Normal('InitialSize_log', 0, 50, shape=(self.nRs,))
            self.Infected_log = pm.Deterministic('Infected_log', T.reshape(self.InitialSize_log, (
                self.nRs, 1)) + self.Growth.cumsum(axis=1))

            self.Infected = pm.Deterministic('Infected', pm.math.exp(self.Infected_log))

            self.DeathsDelayMean = pm.Normal('DeathsDelayMean', deaths_delay_mean_mean, deaths_delay_mean_sd)
            self.DeathsDelayDisp = pm.Normal('DeathsDelayDisp', deaths_delay_disp_mean, deaths_delay_disp_sd)
            deaths_delay_dist = NegativeBinomialCust.dist(mu=self.DeathsDelayMean, alpha=self.DeathsDelayDisp)
            bins = np.arange(0, deaths_truncation)
            pmf = T.exp(deaths_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            fatality_delay = pmf.reshape((1, deaths_truncation))

            expected_deaths = C.conv2d(
                self.Infected,
                fatality_delay,
                border_mode='full'
            )[:, :self.nDs]

            self.ExpectedDeaths = pm.Deterministic('ExpectedDeaths', expected_deaths.reshape(
                (self.nRs, self.nDs)))

            self.Psi = pm.HalfNormal('Psi', 5)

            self.NewDeaths = pm.Data('NewDeaths',
                                     self.d.NewDeaths.data.reshape((self.nRs * self.nDs,))[self.all_observed_deaths])

            # effectively handle missing values ourselves
            self.ObservedDeaths = NegativeBinomialCust(
                'ObservedCases',
                mu=self.ExpectedDeaths.reshape((self.nRs * self.nDs,))[self.all_observed_deaths],
                alpha=self.Psi,
                shape=(len(self.all_observed_deaths),),
                observed=self.NewDeaths
            )


class CasesOnlyModel(BaseCMModel):
    """
    Cases only model.

    Identical to the default model, other than modelling only cases.
    """

    def build_model(self, R_prior_mean=3.28, cm_prior_scale=10, cm_prior='skewed',
                    gi_mean_mean=5, gi_mean_sd=1, gi_sd_mean=2, gi_sd_sd=2, growth_noise_scale=0.2,
                    cases_delay_mean_mean=10, cases_delay_mean_sd=1, cases_delay_disp_mean=5, cases_delay_disp_sd=1,
                    cases_truncation=32, **kwargs):
        """
        Build PyMC3 model.

        :param R_prior_mean: R_0 prior mean
        :param cm_prior_scale: NPI effectiveness prior scale
        :param cm_prior: NPI effectiveness prior type. Either 'normal', 'icl' or skewed (asymmetric laplace)
        :param gi_mean_mean: mean of normal prior placed over the generation interval mean
        :param gi_mean_sd: sd of normal prior placed over the generation interval mean
        :param gi_sd_mean: mean of normal prior placed over the generation interval sd
        :param gi_sd_sd: sd of normal prior placed over the generation interval sd
        :param growth_noise_scale: growth noise scale
        :param cases_delay_mean_mean: mean of normal prior placed over cases delay mean
        :param cases_delay_mean_sd: sd of normal prior placed over cases delay mean
        :param cases_delay_disp_mean: mean of normal prior placed over cases delay dispersion
        :param cases_delay_disp_sd: sd of normal prior placed over cases delay dispersion
        :param deaths_truncation: maximum death delay
        :param cases_truncation: maximum reporting delay
        """
        for key, _ in kwargs.items():
            print(f'Argument: {key} not being used')

        with self.model:
            # build NPI Effectiveness priors
            self.build_npi_prior(cm_prior, cm_prior_scale)

            self.CMReduction = pm.Deterministic('CMReduction', T.exp((-1.0) * self.CM_Alpha))

            self.HyperRVar = pm.HalfNormal(
                'HyperRVar', sigma=0.5
            )

            self.RegionR_noise = pm.Normal('RegionLogR_noise', 0, 1, shape=(self.nRs), )
            self.RegionR = pm.Deterministic('RegionR', R_prior_mean + self.RegionLogR_noise * self.HyperRVar)

            self.ActiveCMs = pm.Data('ActiveCMs', self.d.ActiveCMs)

            self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs, 1))
                    * self.ActiveCMs
            )

            growth_reduction = T.sum(self.ActiveCMReduction, axis=1)

            self.ExpectedLogR = pm.Deterministic(
                'ExpectedLogR',
                T.reshape(pm.math.log(self.RegionR), (self.nRs, 1)) - growth_reduction
            )

            # convert R into growth rates
            self.GI_mean = pm.Normal('GI_mean', gi_mean_mean, gi_mean_sd)
            self.GI_sd = pm.Normal('GI_sd', gi_sd_mean, gi_sd_sd)

            gi_beta = self.GI_mean / self.GI_sd ** 2
            gi_alpha = self.GI_mean ** 2 / self.GI_sd ** 2

            self.ExpectedGrowth = gi_beta * (np.exp(self.ExpectedLogR / gi_alpha) - T.ones_like(
                self.ExpectedLogR))

            self.GrowthNoise = pm.Normal('Growth', 0, growth_noise_scale, shape=(self.nRs, self.nDs - 40))

            self.Growth = T.inc_subtensor(self.ExpectedGrowth[:, 30:-10], self.GrowthNoise)

            self.InitialSize_log = pm.Normal('InitialSize_log', 0, 50, shape=(self.nRs,))
            self.Infected_log = pm.Deterministic('Infected_log', T.reshape(self.InitialSize_log, (
                self.nRs, 1)) + self.Growth.cumsum(axis=1))

            self.Infected = pm.Deterministic('Infected', pm.math.exp(self.Infected_log))

            self.CasesDelayMean = pm.Normal('CasesDelayMean', cases_delay_mean_mean, cases_delay_mean_sd)
            self.CasesDelayDisp = pm.Normal('CasesDelayDisp', cases_delay_disp_mean, cases_delay_disp_sd)
            cases_delay_dist = NegativeBinomialCust.dist(mu=self.CasesDelayMean, alpha=self.CasesDelayDisp)
            bins = np.arange(0, cases_truncation)
            pmf = T.exp(cases_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            reporting_delay = pmf.reshape((1, cases_truncation))

            expected_confirmed = C.conv2d(
                self.Infected,
                reporting_delay,
                border_mode='full'
            )[:, :self.nDs]

            self.ExpectedCases = pm.Deterministic('ExpectedCases', expected_confirmed.reshape(
                (self.nRs, self.nDs)))

            self.Psi = pm.HalfNormal('Phi', 5)

            # effectively handle missing values ourselves
            self.ObservedCases = NegativeBinomialCust(
                'ObservedCases',
                mu=self.ExpectedCases.reshape((self.nRs * self.nDs,))[self.all_observed_active],
                alpha=self.Psi,
                shape=(len(self.all_observed_active),),
                observed=self.d.NewCases.data.reshape((self.nRs * self.nDs,))[self.all_observed_active]
            )


class DiscreteRenewalModel(BaseCMModel):
    """
    Discrete Renewal Model.

    This model is the same as the default, but the infection model does not convert R into g using Wallinga, but rather
    uses a discrete renewal model, adding noise on R.
    """

    def build_model(self, R_prior_mean=3.28, cm_prior_scale=10, cm_prior='skewed',
                    gi_mean_mean=5, gi_mean_sd=1, gi_sd_mean=2, gi_sd_sd=2, R_noise_scale=0.8,
                    deaths_delay_mean_mean=21, deaths_delay_mean_sd=1, deaths_delay_disp_mean=9, deaths_delay_disp_sd=1,
                    cases_delay_mean_mean=10, cases_delay_mean_sd=1, cases_delay_disp_mean=5, cases_delay_disp_sd=1,
                    deaths_truncation=48, cases_truncation=32, gi_truncation=28, conv_padding=7, **kwargs):
        """
        Build NPI effectiveness model

        :param R_prior_mean: R_0 prior mean
        :param cm_prior_scale: NPI effectiveness prior scale
        :param cm_prior: NPI effectiveness prior type. Either 'normal', 'icl' or skewed (asymmetric laplace)
        :param gi_mean_mean: mean of normal prior placed over the generation interval mean
        :param gi_mean_sd: sd of normal prior placed over the generation interval mean
        :param gi_sd_mean: mean of normal prior placed over the generation interval sd
        :param gi_sd_sd: sd of normal prior placed over the generation interval sd
        :param R_noise_scale: multiplicative noise scale, now placed on R!
        :param deaths_delay_mean_mean: mean of normal prior placed over death delay mean
        :param deaths_delay_mean_sd: sd of normal prior placed over death delay mean
        :param deaths_delay_disp_mean: mean of normal prior placed over death delay dispersion (alpha / psi)
        :param deaths_delay_disp_sd: sd of normal prior placed over death delay dispersion (alpha / psi)
        :param cases_delay_mean_mean: mean of normal prior placed over cases delay mean
        :param cases_delay_mean_sd: sd of normal prior placed over cases delay mean
        :param cases_delay_disp_mean: mean of normal prior placed over cases delay dispersion
        :param cases_delay_disp_sd: sd of normal prior placed over cases delay dispersion
        :param deaths_truncation: maximum death delay
        :param cases_truncation: maximum reporting delay
        :param gi_truncation: truncation used for generation interval discretisation
        :param conv_padding: padding for renewal process
        """

        for key, _ in kwargs.items():
            print(f'Argument: {key} not being used')

        with self.model:
            # build NPI Effectiveness priors
            self.build_npi_prior(cm_prior, cm_prior_scale)
            self.CMReduction = pm.Deterministic('CMReduction', T.exp((-1.0) * self.CM_Alpha))

            self.HyperRVar = pm.HalfNormal(
                'HyperRVar', sigma=0.5
            )

            self.RegionR_noise = pm.Normal('RegionLogR_noise', 0, 1, shape=(self.nRs), )
            self.RegionR = pm.Deterministic('RegionR', R_prior_mean + self.RegionLogR_noise * self.HyperRVar)

            self.ActiveCMs = pm.Data('ActiveCMs', self.d.ActiveCMs)

            self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs, 1))
                    * self.ActiveCMs
            )

            self.RReduction = T.sum(self.ActiveCMReduction, axis=1)

            self.ExpectedLogR = T.reshape(T.reshape(pm.math.log(self.RegionR), (self.nRs, 1)) - self.RReduction,
                                          (1, self.nRs, self.nDs)).repeat(2, axis=0)

            self.LogR = pm.Normal('LogR', self.ExpectedLogR, R_noise_scale, shape=(2, self.nRs, self.nDs))

            self.InitialSize_log = pm.Normal('InitialSizeCases_log', 0, 50, shape=(2, self.nRs))

            self.GI_mean = pm.Normal('GI_mean', gi_mean_mean, gi_mean_sd)
            self.GI_sd = pm.Normal('GI_sd', gi_sd_mean, gi_sd_sd)

            GI_dist = pm.Gamma.dist(mu=self.GI_mean, sigma=self.GI_sd)
            points = np.arange(gi_truncation) + 1e-5
            pmf = T.exp(GI_dist.logp(points))
            pmf = pmf / T.sum(pmf)

            # we cannot differentiate through the CDF so instead use the normalise
            # pdf. This matches very well.
            # bins = np.zeros(gi_truncation + 1)
            # bins[1:] = np.arange(gi_truncation)
            # bins[2:] += 0.5
            # bins[:2] += 1e-5
            #
            # cdf_vals = T.exp(GI_dist.logcdf(bins))
            # pmf = cdf_vals[1:] - cdf_vals[:-1]
            GI_rev = T.repeat(T.reshape(pmf[::-1] / T.sum(pmf), (1, 1, gi_truncation)), 2, axis=0)

            infected = T.zeros((2, self.nRs, self.nDs + gi_truncation))
            infected = T.set_subtensor(infected[:, :, (gi_truncation - conv_padding):gi_truncation],
                                       pm.math.exp(self.InitialSize_log.reshape((2, self.nRs, 1)).repeat(
                                           conv_padding, axis=2)))

            # R is a lognorm
            R = pm.math.exp(self.LogR)
            for d in range(self.nDs):
                val = pm.math.sum(
                    R[:, :, d].reshape((2, self.nRs, 1)) * infected[:, :, d:(d + gi_truncation)] * GI_rev,
                    axis=2)
                infected = T.set_subtensor(infected[:, :, d + gi_truncation], val)

            res = infected

            self.InfectedCases = pm.Deterministic(
                'InfectedCases',
                res[0, :, gi_truncation:].reshape((self.nRs, self.nDs))
            )

            self.InfectedDeaths = pm.Deterministic(
                'InfectedDeaths',
                res[1, :, gi_truncation:].reshape((self.nRs, self.nDs))
            )

            self.CasesDelayMean = pm.Normal('CasesDelayMean', cases_delay_mean_mean, cases_delay_mean_sd)
            self.CasesDelayDisp = pm.Normal('CasesDelayDisp', cases_delay_disp_mean, cases_delay_disp_sd)
            cases_delay_dist = NegativeBinomialCust.dist(mu=self.CasesDelayMean, alpha=self.CasesDelayDisp)
            bins = np.arange(0, cases_truncation)
            pmf = T.exp(cases_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            reporting_delay = pmf.reshape((1, cases_truncation))

            self.DeathsDelayMean = pm.Normal('DeathsDelayMean', deaths_delay_mean_mean, deaths_delay_mean_sd)
            self.DeathsDelayDisp = pm.Normal('DeathsDelayDisp', deaths_delay_disp_mean, deaths_delay_disp_sd)
            deaths_delay_dist = NegativeBinomialCust.dist(mu=self.DeathsDelayMean, alpha=self.DeathsDelayDisp)
            bins = np.arange(0, deaths_truncation)
            pmf = T.exp(deaths_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            fatality_delay = pmf.reshape((1, deaths_truncation))

            self.PsiCases = pm.HalfNormal('PsiCases', 5.)
            self.PsiDeaths = pm.HalfNormal('PsiDeaths', 5.)

            expected_cases = C.conv2d(
                self.InfectedCases,
                reporting_delay,
                border_mode='full'
            )[:, :self.nDs]

            expected_deaths = C.conv2d(
                self.InfectedDeaths,
                fatality_delay,
                border_mode='full'
            )[:, :self.nDs]

            self.ExpectedCases = pm.Deterministic('ExpectedCases', expected_cases.reshape(
                (self.nRs, self.nDs)))

            self.ExpectedDeaths = pm.Deterministic('ExpectedDeaths', expected_deaths.reshape(
                (self.nRs, self.nDs)))

            self.NewCases = pm.Data('NewCases',
                                    self.d.NewCases.data.reshape((self.nRs * self.nDs,))[
                                        self.all_observed_active])
            self.NewDeaths = pm.Data('NewDeaths',
                                     self.d.NewDeaths.data.reshape((self.nRs * self.nDs,))[
                                         self.all_observed_deaths])

            self.ObservedCases = NegativeBinomialCust(
                'ObservedCases',
                mu=self.ExpectedCases.reshape((self.nRs * self.nDs,))[self.all_observed_active],
                alpha=self.PsiCases,
                shape=(len(self.all_observed_active),),
                observed=self.NewCases
            )

            self.ObservedDeaths = NegativeBinomialCust(
                'ObservedDeaths',
                mu=self.ExpectedDeaths.reshape((self.nRs * self.nDs,))[self.all_observed_deaths],
                alpha=self.PsiDeaths,
                shape=(len(self.all_observed_deaths),),
                observed=self.NewDeaths
            )


class NoisyRModel(BaseCMModel):
    """
    Noisy-R Model.
    
    This is the same as the default model, but adds noise to R_t before converting this to the growth rate, g_t. In the 
    default model, noise is added to g_t.
    """

    def build_model(self, R_prior_mean=3.28, cm_prior_scale=10, cm_prior='skewed',
                    gi_mean_mean=5, gi_mean_sd=1, gi_sd_mean=2, gi_sd_sd=2, R_noise_scale=0.8,
                    deaths_delay_mean_mean=21, deaths_delay_mean_sd=1, deaths_delay_disp_mean=9, deaths_delay_disp_sd=1,
                    cases_delay_mean_mean=10, cases_delay_mean_sd=1, cases_delay_disp_mean=5, cases_delay_disp_sd=1,
                    deaths_truncation=48, cases_truncation=32, **kwargs):
        """
        Build NPI effectiveness model
        :param R_prior_mean: R_0 prior mean
        :param cm_prior_scale: NPI effectiveness prior scale
        :param cm_prior: NPI effectiveness prior type. Either 'normal', 'icl' or skewed (asymmetric laplace)
        :param gi_mean_mean: mean of normal prior placed over the generation interval mean
        :param gi_mean_sd: sd of normal prior placed over the generation interval mean
        :param gi_sd_mean: mean of normal prior placed over the generation interval sd
        :param gi_sd_sd: sd of normal prior placed over the generation interval sd
        :param growth_noise_scale: growth noise scale
        :param deaths_delay_mean_mean: mean of normal prior placed over death delay mean
        :param deaths_delay_mean_sd: sd of normal prior placed over death delay mean
        :param deaths_delay_disp_mean: mean of normal prior placed over death delay dispersion (alpha / psi)
        :param deaths_delay_disp_sd: sd of normal prior placed over death delay dispersion (alpha / psi)
        :param cases_delay_mean_mean: mean of normal prior placed over cases delay mean
        :param cases_delay_mean_sd: sd of normal prior placed over cases delay mean
        :param cases_delay_disp_mean: mean of normal prior placed over cases delay dispersion
        :param cases_delay_disp_sd: sd of normal prior placed over cases delay dispersion
        :param deaths_truncation: maximum death delay
        :param cases_truncation: maximum reporting delay
        """
        for key, _ in kwargs.items():
            print(f'Argument: {key} not being used')

        with self.model:
            self.build_npi_prior(cm_prior, cm_prior_scale)
            self.CMReduction = pm.Deterministic('CMReduction', T.exp((-1.0) * self.CM_Alpha))

            self.HyperRVar = pm.HalfNormal(
                'HyperRVar', sigma=0.5
            )

            self.RegionR_noise = pm.Normal('RegionLogR_noise', 0, 1, shape=(self.nRs))
            self.RegionR = pm.Deterministic('RegionR', R_prior_mean + self.RegionLogR_noise * self.HyperRVar)

            self.RegionLogR = pm.math.log(self.RegionR)

            self.ActiveCMs = pm.Data('ActiveCMs', self.d.ActiveCMs)

            self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs, 1))
                    * self.ActiveCMs
            )

            self.GrowthReduction = T.sum(self.ActiveCMReduction, axis=1)

            self.ExpectedLogRCases = pm.Normal(
                'ExpectedLogRCases',
                T.reshape(self.RegionLogR, (self.nRs, 1)) - self.GrowthReduction,
                R_noise_scale,
                shape=(self.nRs, self.nDs)
            )

            self.ExpectedLogRDeaths = pm.Normal(
                'ExpectedLogRDeaths',
                T.reshape(self.RegionLogR, (self.nRs, 1)) - self.GrowthReduction,
                R_noise_scale,
                shape=(self.nRs, self.nDs)
            )

            # convert R into growth rates
            self.GI_mean = pm.Normal('GI_mean', gi_mean_mean, gi_mean_sd)
            self.GI_sd = pm.Normal('GI_sd', gi_sd_mean, gi_sd_sd)

            gi_beta = self.GI_mean / self.GI_sd ** 2
            gi_alpha = self.GI_mean ** 2 / self.GI_sd ** 2

            self.GrowthCases = gi_beta * (
                    pm.math.exp(self.ExpectedLogRCases / gi_alpha) - T.ones_like(self.ExpectedLogRCases))

            self.GrowthDeaths = gi_beta * (
                    pm.math.exp(self.ExpectedLogRDeaths / gi_alpha) - T.ones_like(self.ExpectedLogRDeaths))

            self.PsiCases = pm.HalfNormal('PsiCases', 5.)
            self.PsiDeaths = pm.HalfNormal('PsiDeaths', 5.)

            self.InitialSizeCases_log = pm.Normal('InitialSizeCases_log', 0, 50, shape=(self.nRs,))
            self.InfectedCases_log = pm.Deterministic('InfectedCases_log', T.reshape(self.InitialSizeCases_log, (
                self.nRs, 1)) + self.GrowthCases.cumsum(axis=1))

            self.InfectedCases = pm.Deterministic('InfectedCases', pm.math.exp(self.InfectedCases_log))

            self.CasesDelayMean = pm.Normal('CasesDelayMean', cases_delay_mean_mean, cases_delay_mean_sd)
            self.CasesDelayDisp = pm.Normal('CasesDelayDisp', cases_delay_disp_mean, cases_delay_disp_sd)
            cases_delay_dist = NegativeBinomialCust.dist(mu=self.CasesDelayMean, alpha=self.CasesDelayDisp)
            bins = np.arange(0, cases_truncation)
            pmf = T.exp(cases_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            reporting_delay = pmf.reshape((1, cases_truncation))

            expected_cases = C.conv2d(
                self.InfectedCases,
                reporting_delay,
                border_mode='full'
            )[:, :self.nDs]

            self.ExpectedCases = pm.Deterministic('ExpectedCases', expected_cases.reshape(
                (self.nRs, self.nDs)))

            # effectively handle missing values ourselves
            self.ObservedCases = NegativeBinomialCust(
                'ObservedCases',
                mu=self.ExpectedCases.reshape((self.nRs * self.nDs,))[self.all_observed_active],
                alpha=self.PsiCases,
                shape=(len(self.all_observed_active),),
                observed=self.d.NewCases.data.reshape((self.nRs * self.nDs,))[self.all_observed_active]
            )

            self.InitialSizeDeaths_log = pm.Normal('InitialSizeDeaths_log', 0, 50, shape=(self.nRs,))
            self.InfectedDeaths_log = pm.Deterministic('InfectedDeaths_log', T.reshape(self.InitialSizeDeaths_log, (
                self.nRs, 1)) + self.GrowthDeaths.cumsum(axis=1))

            self.InfectedDeaths = pm.Deterministic('InfectedDeaths', pm.math.exp(self.InfectedDeaths_log))

            self.DeathsDelayMean = pm.Normal('DeathsDelayMean', deaths_delay_mean_mean, deaths_delay_mean_sd)
            self.DeathsDelayDisp = pm.Normal('DeathsDelayDisp', deaths_delay_disp_mean, deaths_delay_disp_sd)
            deaths_delay_dist = NegativeBinomialCust.dist(mu=self.DeathsDelayMean, alpha=self.DeathsDelayDisp)
            bins = np.arange(0, deaths_truncation)
            pmf = T.exp(deaths_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            fatality_delay = pmf.reshape((1, deaths_truncation))

            expected_deaths = C.conv2d(
                self.InfectedDeaths,
                fatality_delay,
                border_mode='full'
            )[:, :self.nDs]

            self.ExpectedDeaths = pm.Deterministic('ExpectedDeaths', expected_deaths.reshape(
                (self.nRs, self.nDs)))

            # effectively handle missing values ourselves
            self.ObservedDeaths = NegativeBinomialCust(
                'ObservedDeaths',
                mu=self.ExpectedDeaths.reshape((self.nRs * self.nDs,))[self.all_observed_deaths],
                alpha=self.PsiDeaths,
                shape=(len(self.all_observed_deaths),),
                observed=self.d.NewDeaths.data.reshape((self.nRs * self.nDs,))[self.all_observed_deaths]
            )


class AdditiveModel(BaseCMModel):
    def build_model(self, R_prior_mean=3.28, cm_prior_scale=10,
                    gi_mean_mean=5, gi_mean_sd=1, gi_sd_mean=2, gi_sd_sd=2, growth_noise_scale=0.2,
                    deaths_delay_mean_mean=21, deaths_delay_mean_sd=1, deaths_delay_disp_mean=9, deaths_delay_disp_sd=1,
                    cases_delay_mean_mean=10, cases_delay_mean_sd=1, cases_delay_disp_mean=5, cases_delay_disp_sd=1,
                    deaths_truncation=48, cases_truncation=32, **kwargs):
        """
        Build NPI effectiveness model
        :param R_prior_mean: R_0 prior mean
        :param cm_prior_scale: NPI effectiveness prior scale. For this model, this is the concentration parameter
                                dirichlet distribution, same for all NPIs.
        :param gi_mean_mean: mean of normal prior placed over the generation interval mean
        :param gi_mean_sd: sd of normal prior placed over the generation interval mean
        :param gi_sd_mean: mean of normal prior placed over the generation interval sd
        :param gi_sd_sd: sd of normal prior placed over the generation interval sd
        :param growth_noise_scale: growth noise scale
        :param deaths_delay_mean_mean: mean of normal prior placed over death delay mean
        :param deaths_delay_mean_sd: sd of normal prior placed over death delay mean
        :param deaths_delay_disp_mean: mean of normal prior placed over death delay dispersion (alpha / psi)
        :param deaths_delay_disp_sd: sd of normal prior placed over death delay dispersion (alpha / psi)
        :param cases_delay_mean_mean: mean of normal prior placed over cases delay mean
        :param cases_delay_mean_sd: sd of normal prior placed over cases delay mean
        :param cases_delay_disp_mean: mean of normal prior placed over cases delay dispersion
        :param cases_delay_disp_sd: sd of normal prior placed over cases delay dispersion
        :param deaths_truncation: maximum death delay
        :param cases_truncation: maximum reporting delay
        """
        for key, _ in kwargs.items():
            print(f'Argument: {key} not being used')

        with self.model:
            self.AllBeta = pm.Dirichlet('AllBeta', cm_prior_scale * np.ones((self.nCMs + 1)), shape=(self.nCMs + 1,))
            self.CM_Beta = pm.Deterministic('CM_Beta', self.AllBeta[1:])
            self.Beta_hat = pm.Deterministic('Beta_hat', self.AllBeta[0])
            self.CMReduction = pm.Deterministic('CMReduction', self.CM_Beta)

            self.HyperRVar = pm.HalfNormal(
                'HyperRVar', sigma=0.5
            )

            self.RegionR_noise = pm.Normal('RegionLogR_noise', 0, 1, shape=(self.nRs), )
            self.RegionR = pm.Deterministic('RegionR', R_prior_mean + self.RegionLogR_noise * self.HyperRVar)

            self.ActiveCMs = pm.Data('ActiveCMs', self.d.ActiveCMs)

            active_cm_reduction = T.reshape(self.CM_Beta, (1, self.nCMs, 1)) * (
                    T.ones_like(self.ActiveCMs) - self.ActiveCMs)

            growth_reduction = T.sum(active_cm_reduction, axis=1) + self.Beta_hat

            self.ExpectedLogR = pm.Deterministic(
                'ExpectedLogR',
                T.log(T.exp(T.reshape(pm.math.log(self.RegionR), (self.nRs, 1))) * growth_reduction)
            )

            self.GI_mean = pm.Normal('GI_mean', gi_mean_mean, gi_mean_sd)
            self.GI_sd = pm.Normal('GI_sd', gi_sd_mean, gi_sd_sd)

            gi_beta = self.GI_mean / self.GI_sd ** 2
            gi_alpha = self.GI_mean ** 2 / self.GI_sd ** 2

            self.ExpectedGrowth = gi_beta * (pm.math.exp(self.ExpectedLogR / gi_alpha) - T.ones_like(self.ExpectedLogR))

            self.GrowthCasesNoise = pm.Normal("GrowthCasesNoise", 0, growth_noise_scale,
                                              shape=(self.nRs, self.nDs - 40))
            self.GrowthDeathsNoise = pm.Normal("GrowthDeathsNoise", 0, growth_noise_scale,
                                               shape=(self.nRs, self.nDs - 40))

            self.GrowthCases = T.inc_subtensor(self.ExpectedGrowth[:, 30:-10], self.GrowthCasesNoise)
            self.GrowthDeaths = T.inc_subtensor(self.ExpectedGrowth[:, 30:-10], self.GrowthDeathsNoise)

            self.PsiCases = pm.HalfNormal('PsiCases', 5.)
            self.PsiDeaths = pm.HalfNormal('PsiDeaths', 5.)

            self.InitialSizeCases_log = pm.Normal('InitialSizeCases_log', 0, 50, shape=(self.nRs,))
            self.InfectedCases_log = pm.Deterministic('InfectedCases_log', T.reshape(self.InitialSizeCases_log, (
                self.nRs, 1)) + self.GrowthCases.cumsum(axis=1))

            self.InfectedCases = pm.Deterministic('InfectedCases', pm.math.exp(self.InfectedCases_log))

            self.CasesDelayMean = pm.Normal('CasesDelayMean', cases_delay_mean_mean, cases_delay_mean_sd)
            self.CasesDelayDisp = pm.Normal('CasesDelayDisp', cases_delay_disp_mean, cases_delay_disp_sd)
            cases_delay_dist = NegativeBinomialCust.dist(mu=self.CasesDelayMean, alpha=self.CasesDelayDisp)
            bins = np.arange(0, cases_truncation)
            pmf = T.exp(cases_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            reporting_delay = pmf.reshape((1, cases_truncation))

            expected_cases = C.conv2d(
                self.InfectedCases,
                reporting_delay,
                border_mode='full'
            )[:, :self.nDs]

            self.ExpectedCases = pm.Deterministic('ExpectedCases', expected_cases.reshape(
                (self.nRs, self.nDs)))

            # learn the output noise for this.
            self.ObservedCases = NegativeBinomial(
                'ObservedCases',
                mu=self.ExpectedCases.reshape((self.nRs * self.nDs,))[self.all_observed_active],
                alpha=self.PsiCases,
                shape=(len(self.all_observed_active),),
                observed=self.d.NewCases.data.reshape((self.nRs * self.nDs,))[self.all_observed_active]
            )

            self.InitialSizeDeaths_log = pm.Normal('InitialSizeDeaths_log', 0, 50, shape=(self.nRs,))
            self.InfectedDeaths_log = pm.Deterministic('InfectedDeaths_log', T.reshape(self.InitialSizeDeaths_log, (
                self.nRs, 1)) + self.GrowthDeaths.cumsum(axis=1))

            self.InfectedDeaths = pm.Deterministic('InfectedDeaths', pm.math.exp(self.InfectedDeaths_log))

            self.DeathsDelayMean = pm.Normal('DeathsDelayMean', deaths_delay_mean_mean, deaths_delay_mean_sd)
            self.DeathsDelayDisp = pm.Normal('DeathsDelayDisp', deaths_delay_disp_mean, deaths_delay_disp_sd)
            deaths_delay_dist = NegativeBinomialCust.dist(mu=self.DeathsDelayMean, alpha=self.DeathsDelayDisp)
            bins = np.arange(0, deaths_truncation)
            pmf = T.exp(deaths_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            fatality_delay = pmf.reshape((1, deaths_truncation))

            expected_deaths = C.conv2d(
                self.InfectedDeaths,
                fatality_delay,
                border_mode='full'
            )[:, :self.nDs]

            self.ExpectedDeaths = pm.Deterministic('ExpectedDeaths', expected_deaths.reshape(
                (self.nRs, self.nDs)))

            # effectively handle missing values ourselves
            self.ObservedDeaths = NegativeBinomialCust(
                'ObservedDeaths',
                mu=self.ExpectedDeaths.reshape((self.nRs * self.nDs,))[self.all_observed_deaths],
                alpha=self.PsiDeaths,
                shape=(len(self.all_observed_deaths),),
                observed=self.d.NewDeaths.data.reshape((self.nRs * self.nDs,))[self.all_observed_deaths]
            )


class DifferentEffectsModel(BaseCMModel):
    def build_model(self, R_prior_mean=3.28, cm_prior_scale=10, cm_prior='skewed',
                    gi_mean_mean=5, gi_mean_sd=1, gi_sd_mean=2, gi_sd_sd=2, growth_noise_scale=0.2,
                    alpha_noise_scale=0.1, deaths_delay_mean_mean=21, deaths_delay_mean_sd=1, deaths_delay_disp_mean=9,
                    deaths_delay_disp_sd=1, cases_delay_mean_mean=10, cases_delay_mean_sd=1, cases_delay_disp_mean=5,
                    cases_delay_disp_sd=1, deaths_truncation=48, cases_truncation=32, **kwargs):
        """
        Build NPI effectiveness model
        :param R_prior_mean: R_0 prior mean
        :param cm_prior_scale: NPI effectiveness prior scale. For this model, this is the concentration parameter
                                dirichlet distribution, same for all NPIs.
        :param gi_mean_mean: mean of normal prior placed over the generation interval mean
        :param gi_mean_sd: sd of normal prior placed over the generation interval mean
        :param gi_sd_mean: mean of normal prior placed over the generation interval sd
        :param gi_sd_sd: sd of normal prior placed over the generation interval sd
        :param growth_noise_scale: growth noise scale
        :param deaths_delay_mean_mean: mean of normal prior placed over death delay mean
        :param deaths_delay_mean_sd: sd of normal prior placed over death delay mean
        :param deaths_delay_disp_mean: mean of normal prior placed over death delay dispersion (alpha / psi)
        :param deaths_delay_disp_sd: sd of normal prior placed over death delay dispersion (alpha / psi)
        :param cases_delay_mean_mean: mean of normal prior placed over cases delay mean
        :param cases_delay_mean_sd: sd of normal prior placed over cases delay mean
        :param cases_delay_disp_mean: mean of normal prior placed over cases delay dispersion
        :param cases_delay_disp_sd: sd of normal prior placed over cases delay dispersion
        :param deaths_truncation: maximum death delay
        :param cases_truncation: maximum reporting delay
        """
        with self.model:
            self.build_npi_prior(cm_prior, cm_prior_scale)

            self.CMReduction = pm.Deterministic('CMReduction', T.exp((-1.0) * self.CM_Alpha))

            self.AllCMAlpha = pm.Normal('AllCMAlpha',
                                        T.reshape(self.CM_Alpha, (1, self.nCMs)).repeat(self.nRs, axis=0),
                                        alpha_noise_scale,
                                        shape=(self.nRs, self.nCMs)
                                        )

            self.HyperRVar = pm.HalfNormal(
                'HyperRVar', sigma=0.5
            )

            self.RegionR_noise = pm.Normal('RegionLogR_noise', 0, 1, shape=(self.nRs), )
            self.RegionR = pm.Deterministic('RegionR', R_prior_mean + self.RegionLogR_noise * self.HyperRVar)

            self.ActiveCMs = pm.Data('ActiveCMs', self.d.ActiveCMs)

            active_cm_reduction = T.reshape(self.AllCMAlpha, (self.nRs, self.nCMs, 1)) * self.ActiveCMs
            growth_reduction = T.sum(active_cm_reduction, axis=1)

            self.ExpectedLogR = pm.Deterministic(
                'ExpectedLogR',
                T.reshape(pm.math.log(self.RegionR), (self.nRs, 1)) - growth_reduction,
            )

            # convert R into growth rates
            self.GI_mean = pm.Normal('GI_mean', gi_mean_mean, gi_mean_sd)
            self.GI_sd = pm.Normal('GI_sd', gi_sd_mean, gi_sd_sd)

            gi_beta = self.GI_mean / self.GI_sd ** 2
            gi_alpha = self.GI_mean ** 2 / self.GI_sd ** 2

            self.ExpectedGrowth = gi_beta * (pm.math.exp(self.ExpectedLogR / gi_alpha) - T.ones((self.nRs, self.nDs)))

            # exclude 40 days of noise, slight increase in runtime.
            self.GrowthCasesNoise = pm.Normal("GrowthCasesNoise", 0, growth_noise_scale,
                                              shape=(self.nRs, self.nDs - 40))
            self.GrowthDeathsNoise = pm.Normal("GrowthDeathsNoise", 0, growth_noise_scale,
                                               shape=(self.nRs, self.nDs - 40))

            self.GrowthCases = T.inc_subtensor(self.ExpectedGrowth[:, 30:-10], self.GrowthCasesNoise)
            self.GrowthDeaths = T.inc_subtensor(self.ExpectedGrowth[:, 30:-10], self.GrowthDeathsNoise)

            self.PsiCases = pm.HalfNormal('PsiCases', 5.)
            self.PsiDeaths = pm.HalfNormal('PsiDeaths', 5.)

            self.InitialSizeCases_log = pm.Normal('InitialSizeCases_log', 0, 50, shape=(self.nRs,))
            self.InfectedCases_log = pm.Deterministic('InfectedCases_log', T.reshape(self.InitialSizeCases_log, (
                self.nRs, 1)) + self.GrowthCases.cumsum(axis=1))
            self.InfectedCases = pm.Deterministic('InfectedCases', pm.math.exp(self.InfectedCases_log))

            self.CasesDelayMean = pm.Normal('CasesDelayMean', cases_delay_mean_mean, cases_delay_mean_sd)
            self.CasesDelayDisp = pm.Normal('CasesDelayDisp', cases_delay_disp_mean, cases_delay_disp_sd)
            cases_delay_dist = NegativeBinomialCust.dist(mu=self.CasesDelayMean, alpha=self.CasesDelayDisp)
            bins = np.arange(0, cases_truncation)
            pmf = T.exp(cases_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            reporting_delay = pmf.reshape((1, cases_truncation))

            expected_cases = C.conv2d(
                self.InfectedCases,
                reporting_delay,
                border_mode='full'
            )[:, :self.nDs]

            self.ExpectedCases = pm.Deterministic('ExpectedCases', expected_cases.reshape(
                (self.nRs, self.nDs)))

            # effectively handle missing values ourselves
            self.ObservedCases = NegativeBinomialCust(
                'ObservedCases',
                mu=self.ExpectedCases.reshape((self.nRs * self.nDs,))[self.all_observed_active],
                alpha=self.PsiCases,
                shape=(len(self.all_observed_active),),
                observed=self.d.NewCases.data.reshape((self.nRs * self.nDs,))[self.all_observed_active]
            )

            self.InitialSizeDeaths_log = pm.Normal('InitialSizeDeaths_log', 0, 50, shape=(self.nRs,))
            self.InfectedDeaths_log = pm.Deterministic('InfectedDeaths_log', T.reshape(self.InitialSizeDeaths_log, (
                self.nRs, 1)) + self.GrowthDeaths.cumsum(axis=1))
            self.InfectedDeaths = pm.Deterministic('InfectedDeaths', pm.math.exp(self.InfectedDeaths_log))

            self.DeathsDelayMean = pm.Normal('DeathsDelayMean', deaths_delay_mean_mean, deaths_delay_mean_sd)
            self.DeathsDelayDisp = pm.Normal('DeathsDelayDisp', deaths_delay_disp_mean, deaths_delay_disp_sd)
            deaths_delay_dist = NegativeBinomialCust.dist(mu=self.DeathsDelayMean, alpha=self.DeathsDelayDisp)
            bins = np.arange(0, deaths_truncation)
            pmf = T.exp(deaths_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            fatality_delay = pmf.reshape((1, deaths_truncation))

            expected_deaths = C.conv2d(
                self.InfectedDeaths,
                fatality_delay,
                border_mode='full'
            )[:, :self.nDs]

            self.ExpectedDeaths = pm.Deterministic('ExpectedDeaths', expected_deaths.reshape(
                (self.nRs, self.nDs)))

            # effectively handle missing values ourselves
            self.ObservedDeaths = NegativeBinomialCust(
                'ObservedDeaths',
                mu=self.ExpectedDeaths.reshape((self.nRs * self.nDs,))[self.all_observed_deaths],
                alpha=self.PsiDeaths,
                shape=(len(self.all_observed_deaths),),
                observed=self.d.NewDeaths.data.reshape((self.nRs * self.nDs,))[self.all_observed_deaths]
            )


class DiscreteRenewalFixedGIModel(BaseCMModel):
    """
    Discrete Renewal Model.

    This model is the same as the default, but the infection model does not convert R into g using Wallinga, but rather
    uses a discrete renewal model, adding noise on R.
    """

    def build_model(self, R_prior_mean=3.28, cm_prior_scale=10, cm_prior='skewed', R_noise_scale=0.8,
                    deaths_delay_mean_mean=21, deaths_delay_mean_sd=1, deaths_delay_disp_mean=9, deaths_delay_disp_sd=1,
                    cases_delay_mean_mean=10, cases_delay_mean_sd=1, cases_delay_disp_mean=5, cases_delay_disp_sd=1,
                    deaths_truncation=48, cases_truncation=32, gi_truncation=28, conv_padding=7, **kwargs):
        """
        Build NPI effectiveness model

        :param R_prior_mean: R_0 prior mean
        :param cm_prior_scale: NPI effectiveness prior scale
        :param cm_prior: NPI effectiveness prior type. Either 'normal', 'icl' or skewed (asymmetric laplace)
        :param R_noise_scale: multiplicative noise scale, now placed on R!
        :param deaths_delay_mean_mean: mean of normal prior placed over death delay mean
        :param deaths_delay_mean_sd: sd of normal prior placed over death delay mean
        :param deaths_delay_disp_mean: mean of normal prior placed over death delay dispersion (alpha / psi)
        :param deaths_delay_disp_sd: sd of normal prior placed over death delay dispersion (alpha / psi)
        :param cases_delay_mean_mean: mean of normal prior placed over cases delay mean
        :param cases_delay_mean_sd: sd of normal prior placed over cases delay mean
        :param cases_delay_disp_mean: mean of normal prior placed over cases delay dispersion
        :param cases_delay_disp_sd: sd of normal prior placed over cases delay dispersion
        :param deaths_truncation: maximum death delay
        :param cases_truncation: maximum reporting delay
        :param gi_truncation: truncation used for generation interval discretisation
        :param conv_padding: padding for renewal process
        """

        for key, _ in kwargs.items():
            print(f'Argument: {key} not being used')

        ep = EpidemiologicalParameters()
        gi_s = ep.generate_dist_samples(ep.generation_interval, nRVs=int(1e8), with_noise=False)
        GI = ep.discretise_samples(gi_s, gi_truncation).flatten()
        GI_rev = GI[::-1].reshape((1, 1, GI.size)).repeat(2, axis=0)

        with self.model:
            # build NPI Effectiveness priors
            self.build_npi_prior(cm_prior, cm_prior_scale)
            self.CMReduction = pm.Deterministic('CMReduction', T.exp((-1.0) * self.CM_Alpha))

            self.HyperRVar = pm.HalfNormal(
                'HyperRVar', sigma=0.5
            )

            self.RegionR_noise = pm.Normal('RegionLogR_noise', 0, 1, shape=(self.nRs), )
            self.RegionR = pm.Deterministic('RegionR', R_prior_mean + self.RegionLogR_noise * self.HyperRVar)

            self.ActiveCMs = pm.Data('ActiveCMs', self.d.ActiveCMs)

            self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs, 1))
                    * self.ActiveCMs
            )

            self.RReduction = T.sum(self.ActiveCMReduction, axis=1)

            self.ExpectedLogR = T.reshape(T.reshape(pm.math.log(self.RegionR), (self.nRs, 1)) - self.RReduction,
                                          (1, self.nRs, self.nDs)).repeat(2, axis=0)

            self.LogR = pm.Normal('LogR', self.ExpectedLogR, R_noise_scale, shape=(2, self.nRs, self.nDs))

            self.InitialSize_log = pm.Normal('InitialSizeCases_log', 0, 50, shape=(2, self.nRs))

            infected = T.zeros((2, self.nRs, self.nDs + gi_truncation))
            infected = T.set_subtensor(infected[:, :, (gi_truncation - conv_padding):gi_truncation],
                                       pm.math.exp(self.InitialSize_log.reshape((2, self.nRs, 1)).repeat(
                                           conv_padding, axis=2)))

            # R is a lognorm
            R = pm.math.exp(self.LogR)
            for d in range(self.nDs):
                val = pm.math.sum(
                    R[:, :, d].reshape((2, self.nRs, 1)) * infected[:, :, d:(d + gi_truncation)] * GI_rev,
                    axis=2)
                infected = T.set_subtensor(infected[:, :, d + gi_truncation], val)

            res = infected

            self.InfectedCases = pm.Deterministic(
                'InfectedCases',
                res[0, :, gi_truncation:].reshape((self.nRs, self.nDs))
            )

            self.InfectedDeaths = pm.Deterministic(
                'InfectedDeaths',
                res[1, :, gi_truncation:].reshape((self.nRs, self.nDs))
            )

            self.CasesDelayMean = pm.Normal('CasesDelayMean', cases_delay_mean_mean, cases_delay_mean_sd)
            self.CasesDelayDisp = pm.Normal('CasesDelayDisp', cases_delay_disp_mean, cases_delay_disp_sd)
            cases_delay_dist = NegativeBinomialCust.dist(mu=self.CasesDelayMean, alpha=self.CasesDelayDisp)
            bins = np.arange(0, cases_truncation)
            pmf = T.exp(cases_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            reporting_delay = pmf.reshape((1, cases_truncation))

            self.DeathsDelayMean = pm.Normal('DeathsDelayMean', deaths_delay_mean_mean, deaths_delay_mean_sd)
            self.DeathsDelayDisp = pm.Normal('DeathsDelayDisp', deaths_delay_disp_mean, deaths_delay_disp_sd)
            deaths_delay_dist = NegativeBinomialCust.dist(mu=self.DeathsDelayMean, alpha=self.DeathsDelayDisp)
            bins = np.arange(0, deaths_truncation)
            pmf = T.exp(deaths_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            fatality_delay = pmf.reshape((1, deaths_truncation))

            self.PsiCases = pm.HalfNormal('PsiCases', 5.)
            self.PsiDeaths = pm.HalfNormal('PsiDeaths', 5.)

            expected_cases = C.conv2d(
                self.InfectedCases,
                reporting_delay,
                border_mode='full'
            )[:, :self.nDs]

            expected_deaths = C.conv2d(
                self.InfectedDeaths,
                fatality_delay,
                border_mode='full'
            )[:, :self.nDs]

            self.ExpectedCases = pm.Deterministic('ExpectedCases', expected_cases.reshape(
                (self.nRs, self.nDs)))

            self.ExpectedDeaths = pm.Deterministic('ExpectedDeaths', expected_deaths.reshape(
                (self.nRs, self.nDs)))

            self.NewCases = pm.Data('NewCases',
                                    self.d.NewCases.data.reshape((self.nRs * self.nDs,))[
                                        self.all_observed_active])
            self.NewDeaths = pm.Data('NewDeaths',
                                     self.d.NewDeaths.data.reshape((self.nRs * self.nDs,))[
                                         self.all_observed_deaths])

            self.ObservedCases = NegativeBinomialCust(
                'ObservedCases',
                mu=self.ExpectedCases.reshape((self.nRs * self.nDs,))[self.all_observed_active],
                alpha=self.PsiCases,
                shape=(len(self.all_observed_active),),
                observed=self.NewCases
            )

            self.ObservedDeaths = NegativeBinomialCust(
                'ObservedDeaths',
                mu=self.ExpectedDeaths.reshape((self.nRs * self.nDs,))[self.all_observed_deaths],
                alpha=self.PsiDeaths,
                shape=(len(self.all_observed_deaths),),
                observed=self.NewDeaths
            )



class NegativeBinomialCust(pm.distributions.Discrete):
    R"""
    Negative binomial log-likelihood.
    The negative binomial distribution describes a Poisson random variable
    whose rate parameter is gamma distributed.
    The pmf of this distribution is
    .. math::
       f(x \mid \mu, \alpha) =
           \binom{x + \alpha - 1}{x}
           (\alpha/(\mu+\alpha))^\alpha (\mu/(\mu+\alpha))^x
    .. plot::
        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        from scipy import special
        plt.style.use('seaborn-darkgrid')
        def NegBinom(a, m, x):
            pmf = special.binom(x + a - 1, x) * (a / (m + a))**a * (m / (m + a))**x
            return pmf
        x = np.arange(0, 22)
        alphas = [0.9, 2, 4]
        mus = [1, 2, 8]
        for a, m in zip(alphas, mus):
            pmf = NegBinom(a, m, x)
            plt.plot(x, pmf, '-o', label=r'$\alpha$ = {}, $\mu$ = {}'.format(a, m))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()
    ========  ==========================
    Support   :math:`x \in \mathbb{N}_0`
    Mean      :math:`\mu`
    ========  ==========================
    Parameters
    ----------
    mu: float
        Poission distribution parameter (mu > 0).
    alpha: float
        Gamma distribution parameter (alpha > 0).
    """

    def __init__(self, mu, alpha, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu = mu = tt.as_tensor_variable(floatX(mu))
        self.alpha = alpha = tt.as_tensor_variable(floatX(alpha))
        self.mode = intX(tt.floor(mu))

    def random(self, point=None, size=None):
        r"""
        Draw random values from NegativeBinomial distribution.
        Parameters
        ----------
        point: dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size: int, optional
            Desired size of random sample (returns one sample if not
            specified).
        Returns
        -------
        array
        """
        mu, alpha = draw_values([self.mu, self.alpha], point=point, size=size)
        g = generate_samples(self._random, mu=mu, alpha=alpha,
                             dist_shape=self.shape,
                             size=size)
        g[g == 0] = np.finfo(float).eps  # Just in case

        g = np.minimum(g, 10e10)
        
        return np.asarray(stats.poisson.rvs(g)).reshape(g.shape)

    def _random(self, mu, alpha, size):
        r""" Wrapper around stats.gamma.rvs that converts NegativeBinomial's
        parametrization to scipy.gamma. All parameter arrays should have
        been broadcasted properly by generate_samples at this point and size is
        the scipy.rvs representation.
        """
        return stats.gamma.rvs(
            a=alpha,
            scale=mu / alpha,
            size=size,
        )

    def logp(self, value):
        r"""
        Calculate log-probability of NegativeBinomial distribution at specified value.
        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or theano tensor
        Returns
        -------
        TensorVariable
        """
        mu = self.mu
        alpha = self.alpha
        negbinom = bound(binomln(value + alpha - 1, value)
                         + logpow(mu / (mu + alpha), value)
                         + logpow(alpha / (mu + alpha), alpha),
                         value >= 0, mu > 0, alpha > 0)

        # Return Poisson when alpha gets very large.
        return tt.switch(tt.gt(alpha, 1e10),
                         pm.Poisson.dist(self.mu).logp(value),
                         negbinom)
