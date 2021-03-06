#!/usr/bin/env python

import numpy as np
import pandas as pd
import pickle

from scipy import stats as sps
from scipy.interpolate import interp1d

import coronadataclass as cdc


class RtComputation(object):
    """
    Compute R(t) from JHU data
    code copied and modified from Kevin Systrom, github.com/k-sys/covid-19/ 
    based on Bettencourt & Ribeiro, PLoS ONE (2008),  https://doi.org/10.1371/journal.pone.0002185
    """
    def __init__(self,**kwargs):
        self.jhu_data      = cdc.CoronaData(**kwargs, DateAsIndex = True)
        self.verbose       = kwargs.get('verbose', False)
        
        self.__skip_empty    = kwargs.get('SkipEmpty',True)
        self.__remove_negative_case_count = kwargs.get('RemoveNegativeCaseCount',True)
        self.__drop_nan      = kwargs.get('DropNaN',True)
        
        self.__R_T_MAX     = kwargs.get('RtMax',12)
        self.__R_T_SPACING = kwargs.get('RtSpacing',100)
        self.r_t_range     = np.linspace(0, self.__R_T_MAX, self.__R_T_MAX*self.__R_T_SPACING+1)
        
        self.GAMMA         = 1. / kwargs.get('SerialInterval',7.)
        
        self.sigma         = kwargs.get('Sigma',.25)
        self.sigmas        = np.linspace(1/20, 1, 20)
    
        self.hdi_list      = kwargs.get('HighestDensityIntervals',[.5,.9])
    
        self.__SmoothRt    = kwargs.get('SmoothRtWindowSize',None)
        self.__SmoothRtStd = kwargs.get('SmoothRtStdDev',None)
    
        self.__posteriors  = {}
        self.__log_likelihoods = {}
        
        self.__kwargs_for_pickle = kwargs
    
    
    
    def HighestDensityInterval(self, pmf, p = .9, debug = None):
        # If we pass a DataFrame, just call this recursively on the columns
        if(isinstance(pmf, pd.DataFrame)):
            return pd.DataFrame([self.HighestDensityInterval(pmf[col], p = p, debug = debug) for col in pmf], index = pmf.columns)
        
        cumsum     = np.cumsum(pmf.values)
        
        if np.any(np.isnan(pmf.values)) and self.verbose:
            print(debug)
        
        low_index  = np.argmin(cumsum <= (1-p)/2)
        high_index = np.argmax(cumsum >= (1+p)/2)
                
        low        = pmf.index[low_index]
        high       = pmf.index[high_index]
        
        return pd.Series([low, high], index = [f'Low_{p*100:.0f}', f'High_{p*100:.0f}'])



    def PrepareCases(self, cases, cutoff = 30):
        new_cases = cases.diff()
        if self.__remove_negative_case_count:
            new_cases[new_cases < 0] = 0
        smoothed  = new_cases.rolling(7, win_type = 'gaussian', min_periods = 1, center = True).mean(std=2).round()
        idx_start = np.searchsorted(smoothed, cutoff)
        smoothed  = smoothed.iloc[idx_start:]
        original  = new_cases.loc[smoothed.index]
        
        return original, smoothed
    
    
    
    def GetPosteriors(self, sr, sigma = 0.25):

        # (1) Calculate Lambda
        lam = sr[:-1].values * np.exp(self.GAMMA * (self.r_t_range[:, None] - 1))
        
        # (2) Calculate each day's likelihood
        likelihoods = pd.DataFrame(
            data = sps.poisson.pmf(sr[1:].values, lam),
            index = self.r_t_range,
            columns = sr.index[1:])
        
        # (3) Create the Gaussian Matrix
        process_matrix = sps.norm(loc = self.r_t_range, scale = sigma).pdf(self.r_t_range[:, None]) 

        # (3a) Normalize all rows to sum to 1
        process_matrix /= process_matrix.sum(axis=0)
        
        # (4) Calculate the initial prior
        #prior0 = sps.gamma(a=4).pdf(self.r_t_range)
        prior0 = np.ones_like(self.r_t_range)/len(self.r_t_range)
        prior0 /= prior0.sum()

        # Create a DataFrame that will hold our posteriors for each day
        # Insert our prior as the first posterior.
        posteriors = pd.DataFrame(
            index=self.r_t_range,
            columns=sr.index,
            data={sr.index[0]: prior0}
        )
        
        # We said we'd keep track of the sum of the log of the probability
        # of the data for maximum likelihood calculation.
        log_likelihood = 0.0

        # (5) Iteratively apply Bayes' rule
        for previous_day, current_day in zip(sr.index[:-1], sr.index[1:]):

            #(5a) Calculate the new prior
            current_prior = process_matrix @ posteriors[previous_day]
            
            #(5b) Calculate the numerator of Bayes' Rule: P(k|R_t)P(R_t)
            numerator = likelihoods[current_day] * current_prior
            
            #(5c) Calcluate the denominator of Bayes' Rule P(k)
            denominator = np.sum(numerator)
            
            # Execute full Bayes' Rule
            posteriors[current_day] = numerator/denominator
            
            # Add to the running sum of log likelihoods
            log_likelihood += np.log(denominator)

        return posteriors, log_likelihood
    
    
    
    def RunCV(self, countrylist = None, sigmalist = None):
        if countrylist is None: countrylist = self.countrylist
        if sigmalist is None: sigmalist = self.sigmalist
        
        self.sigmalist = sigmalist
        
        for country in countrylist:
            for sigma in sigmalist:
                self.Posteriors(country,sigma)
                
        total_log_likelihoods = np.zeros_like(sigmalist)
        # Each index of this array holds the total of the log likelihoods for
        # the corresponding index of the sigmas array.

        # Loop through each state's results and add the log likelihoods to the running total.
        for sigma in sigmalist:
            total_log_likelihoods.append(np.sum([llh for country,llh in self.log_likelihoods[sigma].items()]))

        # Select the index with the largest log likelihood total
        self.max_likelihood_index = total_log_likelihoods.argmax()

        # Select the value that has the highest log likelihood
        self.sigma = self.sigmas[self.max_likelihood_index]



    def Posteriors(self, country, sigma = None):
        if sigma is None: sigma = self.sigma
        if sigma in self.__posteriors.keys():
            if country in self.__posteriors[sigma].keys():
                return self.__posteriors[sigma][country]
        else:
            self.__posteriors[sigma] = {}
            self.__log_likelihoods[sigma] = {}
            
        cases         = self.jhu_data.CountryData(country)['Confirmed']
        new, smoothed = self.PrepareCases(cases, cutoff = 30)
        
        if len(smoothed) > 0:
            posterior, log_likelihood              = self.GetPosteriors(smoothed, sigma = sigma)
            self.__posteriors[sigma][country]      = posterior
            self.__log_likelihoods[sigma][country] = log_likelihood
            
            if self.__drop_nan:
                self.__posteriors[sigma][country].dropna(axis = 1, inplace = True)
            
            return self.__posteriors[sigma][country]
        else:
            return None

    
    
    def ProcessPosterior(self, posterior, hdi = None, debug = None):
        if hdi is None: hdi = self.hdi_list
        retDF = pd.DataFrame({'Date':posterior.idxmax().index, 'Rt':posterior.idxmax().values})
        retDF['Date'] = pd.to_datetime(retDF.Date,format = '%d/%m/%Y')
        for p in hdi:
            hdi_series = self.HighestDensityInterval(posterior, p = p, debug = debug).reset_index().drop(columns = ['Date'])
            retDF = pd.concat([retDF,hdi_series], axis = 1)
        if (not self.__SmoothRt is None) and (not self.__SmoothRtStd is None):
            return retDF.drop(retDF.index[0]).set_index('Date', drop = True).rolling(self.__SmoothRt, win_type = 'Gaussian', min_periods = 1, center = True).mean(std = self.__SmoothRtStd)
        else:
            return retDF.drop(retDF.index[0]).set_index('Date', drop = True)
        


    def CountryData(self, country = None, sigma = None, hdi = None):
        if sigma is None: sigma = self.sigma
        if hdi   is None: hdi   = self.hdi_list
        
        if country in self.countrylist:
            posterior = self.Posteriors(country = country, sigma = self.sigma)
            if not posterior is None:
                return self.ProcessPosterior(posterior, hdi = hdi, debug = country)
        
        # something above failed
        return None



    def __getattr__(self,key):
        if key == 'countrylist':
            return self.jhu_data.countrylist
        if key.replace('_',' ') in self.jhu_data.countrylist:
            return self.CountryData(country = key)



    def __getstate__(self):
        return {'kwargs':  self.__kwargs_for_pickle,
                'sigmas':  self.sigmas,
                'results': self.result,
                'rt':      self.rt,
                'mlhi':    self.max_likelihood_index
                }
    
    
    
    def __setstate__(self,state):
        self.__init__(**state['kwargs'])
        self.sigmas               = state['sigmas']
        self.results              = state['results']
        self.rt                   = state['rt']
        self.max_likelihood_index = state['mlhi']



    def __iter__(self):
        for country in self.countrylist:
            cd = self.CountryData(country = country)
            if (self.__skip_empty and not cd is None) or (not self.__skip_empty):
                yield country, cd
                
