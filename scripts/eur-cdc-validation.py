#!/usr/bin/env python
# coding: utf-8

# In[20]:


from epimodel.pymc3_models import cm_effect
from epimodel.pymc3_models.cm_effect.datapreprocessor import DataPreprocessor, ICLDataPreprocessor
import pickle

import pandas as pd
import numpy as  np
import theano 
import scipy.signal as ss
import pymc3 as pm


# In[3]:


cm_plot_style = [
#             ("\uf7f2", "tab:red"), # hospital symbol
            ("\uf963", "black"), # mask
            ("\uf0c0", "lightgrey"), # ppl
            ("\uf0c0", "grey"), # ppl
            ("\uf0c0", "black"), # ppl
            ("\uf07a", "tab:orange"), # shop 1
            ("\uf07a", "tab:red"), # shop2
            ("\uf549", "black"), # school
            ("\uf19d", "black"), # university
            ("\uf965", "black"), # home
            ("\uf072", "grey"), # plane1
            ("\uf072", "black"), # plane2
            ("\uf238", "black"), # train
            ("\uf1b9", "black"), # car
            ("\uf641", "black") # flyer
        ]


# In[70]:


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
    dp = DataPreprocessor()
    data = dp.preprocess_data(regular_data_path)


    
    
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

    

    SmoothedNewCases = np.around(
        ss.convolve2d(NewCases, 1 / dp.N_smooth * np.ones(shape=(1, dp.N_smooth)), boundary="symm",
                      mode="same"))
    SmoothedNewDeaths = np.around(
        ss.convolve2d(NewDeaths, 1 / dp.N_smooth * np.ones(shape=(1, dp.N_smooth)), boundary="symm",
                      mode="same"))
    
    for r_i, r in enumerate(data.Rs):
        # if the country has too few deaths, ignore
        if data.Deaths[r_i, -1] < 50:
            print(f"EUR CDC Skipping smoothing {region_names[r_i]}")
            SmoothedNewDeaths[r_i, :] = NewDeaths[r_i, :]
    
    NewCases = SmoothedNewCases
    NewDeaths = SmoothedNewDeaths
    NewDeaths[NewDeaths < 0] = np.nan
    NewCases[NewCases < 0] = np.nan
    data.NewCases = np.ma.masked_invalid(NewCases.astype(theano.config.floatX))
    data.NewDeaths = np.ma.masked_invalid(NewDeaths.astype(theano.config.floatX))
    
    


    return data

dp = DataPreprocessor()
jh_data = dp.preprocess_data("../final_data/data_final.csv")
eur_data = process_euro_data("../final_data/eur_data.csv", "../final_data/data_final.csv", "../final_data/ICL.csv")


# In[33]:








# In[75]:


with cm_effect.models.CMCombined_Final(eur_data, cm_plot_style) as eur_model:
    eur_model.build_model()

with eur_model:
    eur_model.trace = pm.sample(800, cores=4, chains=4, max_treedepth=12)
    pm.save_trace(eur_model.trace, directory="small-eur")


# In[62]:
"""
with cm_effect.models.CMCombined_Final(jh_data, cm_plot_style) as jh_model:
    jh_model.build_model()
with jh_model:
    jh_model.trace = pm.sample(500, cores=4, chains=1, max_treedepth=12)

    pm.save_trace(jh_model.trace, directory="small-eur")
    """
