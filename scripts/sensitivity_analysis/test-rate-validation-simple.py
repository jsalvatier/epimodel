#!/usr/bin/env python
# coding: utf-8

# In[1]:


from epimodel import EpidemiologicalParameters, DefaultModel, preprocess_data
from epimodel.preprocessing.preprocessed_data import PreprocessedData
from epimodel.pymc3_models.base_model import produce_CIs
import numpy as np
import pymc3 as pm
import pickle 
from pymc3.distributions import draw_values, generate_samples

import pandas as pd 


# In[2]:



data = preprocess_data('notebooks/double-entry-data/double_entry_final.csv', last_day='2020-05-30', smoothing=1)
data.mask_reopenings(print_out = False)


# In[4]:


ep = EpidemiologicalParameters()


# In[5]:


with DefaultModel(data) as model_syn:
    model_syn.build_model(**ep.get_model_build_dict())


# In[6]:


with model_syn: 
    
    point = {
        'CM_Alpha' : np.array([0,0,0,0,0,.1,.9,.9,.9]),
        'InitialSizeCases_log' : -10 *np.ones((41,1)), 
        'InitialSizeDeaths_log' : -10 *np.ones((41,1)), 
        'RegionLogR_noise' : .1*np.ones(41),
        'HyperRVar' : 1
    }
    c = [model_syn.GrowthCases, 
         model_syn.GrowthDeaths, 
         model_syn.RegionR, 
         model_syn.ObservedCases, 
         model_syn.ObservedDeaths, 
         model_syn.RegionLogR_noise, 
         model_syn.InfectedCases ,
         model_syn.InfectedDeaths ,
         model_syn.ExpectedCases ,
         model_syn.ExpectedDeaths ]
    
    synthvalues = draw_values(c, point=point)
    gcases,gdeaths, regionR, ocases, odeaths, regionRnoise,icases,ideaths,ecases, edeaths = synthvalues


# In[7]:
with open("synthetic_values.pkl", 'wb+') as f:
    pickle.dump(synthvalues, f, pickle.HIGHEST_PROTOCOL)

def synthetic_prep_data(model, data, syn_deaths, syn_cases):
    
    def reverse_shape(synth, indexes, mask):
        a = np.zeros(model.nRs * model.nDs)
        a[indexes] += synth
        a = a.reshape((model.nRs, model.nDs))
        a = np.ma.masked_array(a, mask)
        return a

    
    syntheticNewDeaths = reverse_shape(syn_deaths, 
                                       model.all_observed_deaths,  
                                       data.NewDeaths.mask)
    syntheticNewCases = reverse_shape(syn_cases, 
                              model.all_observed_active,  
                              data.NewCases.mask)
    
    return PreprocessedData(
                 data.Active,
                 data.Confirmed,
                 data.ActiveCMs,
                 data.CMs,
                 data.Rs,
                 data.Ds,
                 data.Deaths,
                 syntheticNewDeaths,
                 syntheticNewCases,
                 data.RNames)



# In[8]:


adjustment = np.array([ np.random.choice([.5, 2]) for i in range(model_syn.nRs)])
with open("adjustment.pkl", 'wb+') as f:
    pickle.dump(adjustment, f, pickle.HIGHEST_PROTOCOL)


# In[9]:


synth_data_normal = synthetic_prep_data(model_syn, data, odeaths, ocases)
synth_data_adjusted = synthetic_prep_data(model_syn, data, odeaths, ocases)
synth_data_adjusted.NewCases  = synth_data_adjusted.NewCases * adjustment[:, None]


# In[ ]:



    


# In[10]:


with DefaultModel(synth_data_normal) as model_normal:
    model_normal.build_model(**ep.get_model_build_dict())


# In[18]:


with model_normal:
    v = model_normal.vars.copy()
    v.remove(model_normal.GrowthCasesNoise)
    v.remove(model_normal.GrowthDeathsNoise)
    model_normal.trace = pm.sample(3000, tune=200, cores=4, chains=4, max_treedepth=12, target_accept=0.925)
    pm.save_trace(model_normal.trace, "normal_simple", overwrite=True)


# In[19]:


with DefaultModel(synth_data_adjusted) as model_adjusted:
    model_adjusted.build_model(**ep.get_model_build_dict())


# In[20]:


with model_adjusted:
    v = model_adjusted.vars.copy()
    v.remove(model_adjusted.GrowthCasesNoise)
    v.remove(model_adjusted.GrowthDeathsNoise)
    model_adjusted.trace = pm.sample(3000, tune=200, cores=4, chains=4, max_treedepth=12, target_accept=0.925)
    pm.save_trace(model_adjusted.trace, "adjusted_simple", overwrite=True)
