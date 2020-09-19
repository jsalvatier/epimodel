#!/usr/bin/env python
# coding: utf-8

# In[1]:


from epimodel import EpidemiologicalParameters, DefaultModel, preprocess_data
from epimodel.preprocessing.preprocessed_data import PreprocessedData
from epimodel.pymc3_models.base_model import produce_CIs
import pickle
import numpy as np
import pymc3 as pm
from pymc3.distributions import draw_values, generate_samples
import theano

import pandas as pd 


# In[2]:


import pylab as plt

# In[3]:


data = preprocess_data('notebooks/double-entry-data/double_entry_final.csv', last_day='2020-05-30', smoothing=1)
data.mask_reopenings(print_out = False)

region_info = [
    ("Andorra", "AD", "AND"),
    ("Austria", "AT", "AUT"),
    ("Albania", "AL", "ALB"),
    ("Bosnia and Herzegovina", "BA", "BIH"),
    ("Belgium", "BE", "BEL"),
    ("Bulgaria", "BG", "BGR"),
    ("Switzerland", "CH", "CHE"),
    ("Czech Republic", "CZ", "CZE"),
    ("Germany", "DE", "DEU"),
    ("Denmark", "DK", "DNK"),
    ("Estonia", "EE", "EST"),
    ("Spain", "ES", "ESP"),
    ("Finland", "FI", "FIN"),
    ("France", "FR", "FRA"),
    ("United Kingdom", "GB", "GBR"),
    ("Georgia", "GE", "GEO"),
    ("Greece", "GR", "GRC"),
    ("Croatia", "HR", "HRV"),
    ("Hungary", "HU", "HUN"),
    ("Ireland", "IE", "IRL"),
    ("Israel", "IL", "ISR"),
    ("Iceland", "IS", "ISL"),
    ("Italy", "IT", "ITA"),
    ("Lithuania", "LT", "LTU"),
    ("Latvia", "LV", "LVA"),
    ("Malta", "MT", "MLT"),
    ("Morocco", "MA", "MAR"),
    ("Mexico", "MX", "MEX"),
    ("Malaysia", "MY", "MYS"),
    ("Netherlands", "NL", "NLD"),
    ("Norway", "NO", "NOR"),
    ("New Zealand", "NZ", "NZL"),
    ("Poland", "PL", "POL"),
    ("Portugal", "PT", "PRT"),
    ("Romania", "RO", "ROU"),
    ("Serbia", "RS", "SRB"),
    ("Sweden", "SE", "SWE"),
    ("Singapore", "SG", "SGP"),
    ("Slovenia", "SI", "SVN"),
    ("Slovakia", "SK", "SVK"),
    ("South Africa", "ZA", "ZAF"),
]

region_info.sort(key=lambda x: x[0])
region_names = list([x for x, _, _ in region_info])
regions_epi = list([x for _, x, _ in region_info])
regions_threecode = list([x for _, _, x in region_info])


def eur_to_epi_code(x):
    if x in regions_threecode:
        return regions_epi[regions_threecode.index(x)]
    else:
        return "not found"




def process_euro_data(path, regular_data_path, ICL_data_path):
    data = preprocess_data('notebooks/double-entry-data/double_entry_final.csv')
    data.mask_reopenings(print_out = False)


    
    
    eur_df = pd.read_csv(path, parse_dates=["dateRep"], infer_datetime_format=True)
    eur_df['dateRep'] = pd.to_datetime(eur_df['dateRep'], utc=True)
    epi_codes = [eur_to_epi_code(cc) for cc in eur_df["countryterritoryCode"]]
    dti = pd.to_datetime(eur_df['dateRep'], utc=True)

    eur_df.index = pd.MultiIndex.from_arrays([epi_codes, dti])

    columns_to_drop = ["day", "month", "year", "countriesAndTerritories", "geoId", "popData2018", "continentExp",
                       "dateRep", "countryterritoryCode"]

    for col in columns_to_drop:
        del eur_df[col]

    eur_df = eur_df.loc[regions_epi]

    NewCases = np.zeros((len(data.Rs), len(data.Ds)))
    NewDeaths = np.zeros((len(data.Rs), len(data.Ds)))

    for r_i, r in enumerate(data.Rs):
        for d_i, d in enumerate(data.Ds):
            c_vals = eur_df.loc[r]
            if d in c_vals.index:
                NewCases[r_i, d_i] = c_vals["cases"].loc[d]
                NewDeaths[r_i, d_i] = c_vals["deaths"].loc[d]
    
    NewDeaths[NewDeaths < 0] = np.nan
    NewCases[NewCases < 0] = np.nan
    data.NewCases = np.ma.masked_invalid(NewCases.astype(theano.config.floatX))
    data.NewDeaths = np.ma.masked_invalid(NewDeaths.astype(theano.config.floatX))
    

    return data

jh_data = preprocess_data('notebooks/double-entry-data/double_entry_final.csv')
jh_data.mask_reopenings(print_out = False)
eur_data = process_euro_data("notebooks/final_data/eur_data.csv", 'notebooks/double-entry-data/double_entry_final.csv', "notebooks/final_data/ICL.csv")

# In[4]:


ep = EpidemiologicalParameters()




with DefaultModel(jh_data) as jh_model:
    jh_model.build_model(**ep.get_model_build_dict())


# In[15]:


with jh_model:
    v = jh_model.vars.copy()
    v.remove(jh_model.GrowthCasesNoise)
    v.remove(jh_model.GrowthDeathsNoise)
    
    jh_model.trace = pm.sample(1500, tune=500, cores=4, chains=4, max_treedepth=12, target_accept=0.925, trace=v, start=jh_model.test_point)
    pm.save_trace(jh_model.trace, "euro_validation_results/jh", overwrite=True)
    np.savetxt('euro_validation_results/jh_CM_Alpha.txt', jh_model.trace['CM_Alpha'])


with DefaultModel(eur_data ) as euro_model:
    euro_model.build_model(**ep.get_model_build_dict())


# In[15]:


with euro_model:
    v = euro_model.vars.copy()
    v.remove(euro_model.GrowthCasesNoise)
    v.remove(euro_model.GrowthDeathsNoise)
    
    euro_model.trace = pm.sample(1500, tune=500, cores=4, chains=4, max_treedepth=12, target_accept=0.925, trace=v, start=euro_model.test_point)
    pm.save_trace(euro_model.trace, "euro_validation_results/euro", overwrite=True)
    np.savetxt('euro_validation_results/euro_CM_Alpha.txt', euro_model.trace['CM_Alpha'])
