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

import pandas as pd 


# In[2]:


import pylab as plt

# In[3]:


data = preprocess_data('notebooks/double-entry-data/double_entry_final.csv', last_day='2020-05-30', smoothing=1)
data.mask_reopenings(print_out = False)


# In[4]:


ep = EpidemiologicalParameters()


# In[5]:


with DefaultModel(data) as model_syn:
    model_syn.build_model(**ep.get_model_build_dict())

# In[7]:


with model_syn: 
    
    point = {
        'CM_Alpha' : np.array([.0,.05,.1,.15,.2,.25,.3,.35,.4]),
        'InitialSizeCases_log' : -7 *np.ones((41,1)), 
        'InitialSizeDeaths_log' : -7 *np.ones((41,1)), 
        #'RegionLogR_noise' : .1*np.ones(41),
        'HyperRVar' : .2
    }

    c = [
         model_syn.CM_Alpha,
         model_syn.InitialSizeCases_log,
         model_syn.RegionLogR_noise,
         model_syn.HyperRVar,
         model_syn.GrowthCases, 
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

    _,_,_,_,gcases,gdeaths, regionR, ocases, odeaths, regionRnoise,icases,ideaths,ecases, edeaths = synthvalues

    synthvalues = dict(zip([v.name for v in c], synthvalues))

# In[ ]:


with open("identifiability_results/synthetic_values_variable.pkl", 'wb+') as f:
    pickle.dump(synthvalues, f, pickle.HIGHEST_PROTOCOL)


# In[8]:


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


# In[13]:


synth_data_normal = synthetic_prep_data(model_syn, 
        preprocess_data('notebooks/double-entry-data/double_entry_final.csv', last_day='2020-05-30', smoothing=1),
        synthvalues['ObservedDeaths'],
        synthvalues['ObservedCases'])
synth_data_normal.mask_reopenings(print_out = False)


# In[14]:


with DefaultModel(synth_data_normal) as model_normal:
    model_normal.build_model(**ep.get_model_build_dict())


# In[15]:


with model_normal:
    v = model_normal.vars.copy()
    v.remove(model_normal.GrowthCasesNoise)
    v.remove(model_normal.GrowthDeathsNoise)
    
    model_normal.trace = pm.sample(2000, tune=500, cores=4, chains=4, max_treedepth=12, target_accept=0.925, trace=v, start=model_normal.test_point)
    pm.save_trace(model_normal.trace, "identifiability_results/synth", overwrite=True)
    np.savetxt('identifiability_results/synth_CM_Alpha.txt', model_normal.trace['CM_Alpha'])


