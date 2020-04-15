import pandas as pd
import numpy as np
from scipy.stats.distributions import gamma

ICL_ITD = gamma(1/0.45**2,0,18.8*0.45**2)

DEATHS_DAYS = np.array([ICL_ITD.cdf(a+1)-ICL_ITD.cdf(a) for a in range(40)])

# see heading "Align Confirmed Cases" below

def transformCases(s):
    shifted = s.diff().shift(-7)
    return shifted


# This isn't very accurate, and a better nowcast might be useful.
# However, wherever we use it we then take a convolutions with a distributions with 
# very small weight on the relevant days, so the poor fit is acceptable.

def fillforward(orig,fill=7, pred=7):
    s = orig.copy()
    data_x = np.linspace(1,pred,pred)
    data_pred = np.linspace(pred+1,pred+1+fill,fill)
    try:
        s[-fill:] = np.poly1d(np.polyfit(data_x,
                                       s[-(pred+fill):-fill],1))(data_pred)
    except ValueError:
        print('Invalid data for linear fit', s[-(pred+fill):-fill])
        # In this case, we really don't know what cases are likely to do
        s[-fill:] = np.nan
        return s
    return s

# see heading "Comparison of Expected Deaths" below


def expectedDeaths(s,fatality=0.008):
    cv = np.convolve(fillforward(s),DEATHS_DAYS,'valid')
    pw = len(s)-len(cv)
    return fatality*np.pad(cv,(pw,0),"constant",constant_values=np.nan)

## see heading "Recovered, Infectious and Exposed Classes" below


def lik_r(i,mu=0.5):
    return np.exp(-mu*i)

norm = lik_r(np.arange(1,100)).sum()

def r(i):
    return lik_r(i)/norm

def R(ti):
    ti_pad = np.pad(ti,(40,0),'constant',constant_values=0)
    cv = np.convolve(ti_pad,r(np.arange(1,42)),'valid')
    pw = len(ti)-len(cv)
    return np.pad(cv,(pw,0),"constant",constant_values=0)

norm_I = lik_r(np.arange(1,100),0.2).sum()

def inf(i):
    return lik_r(i,0.2)/norm_I

def E2I(ever_exposed):
    ee_pad = np.pad(ever_exposed,(40,0),'constant',constant_values=0)
    cv = np.convolve(ee_pad,inf(np.arange(1,42)),'valid')
    pw = len(ever_exposed)-len(cv)
    return np.pad(cv,(pw,0),"constant",constant_values=0)

# Calculate ascertainment, true infection rates, exposed and infectious classes and add as new columns
# Returns a much smaller dataframe, as 

def filtered_mean(m,indices):
    return m[indices].mean()

def ascertainment(csse_ds,fatality = 0.008):
    csse_df = csse_ds.copy()
    csse_df['New confirmed shifted'] = csse_df['Confirmed'].groupby(level=0).transform(transformCases)
    csse_df['New deaths'] = csse_df['Deaths'].groupby(level=0).transform(lambda x: x.diff())
    
    g = csse_df.groupby(level=0)
    
    csse_df['Expected deaths'] = g['New confirmed shifted'].transform(expectedDeaths,fatality=fatality)
    
    indices = csse_df['New deaths']>=5
    
    csse_df['Ascertainment'] = np.nan
    
    csse_df.loc[indices,'Ascertainment'] = pd.to_numeric(csse_df.loc[indices,'Expected deaths']
                                            /csse_df.loc[indices,'New deaths'])
    
    csse_df['New cases true'] = (csse_df['New confirmed shifted']
                                             /csse_df.groupby(level=0)['Ascertainment'].transform(filtered_mean,indices))

    g2 = csse_df.groupby(level=0)
    csse_df['Exposed'] = g2['New cases true'].transform(np.cumsum) - g2['New cases true'].transform(lambda x: np.cumsum(E2I(x)))
    csse_df['Recovered'] = g2['New cases true'].transform(lambda x: np.cumsum(R(E2I(x))))
    csse_df['Infectious'] = g2['New cases true'].transform(lambda x: np.cumsum(E2I(x))) - csse_df['Recovered']
    return csse_df