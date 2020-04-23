#basics
import numpy as np
import pandas as pd

import os
import itertools
import datetime
import time

# statistics
import statsmodels.api as sm
import statsmodels.formula.api as smf

# datawrappers
import coronadataclass as cdc
import measureclass as mc




class CrossValidation(object):
    def __init__(self, **kwargs):
        self.__MinCaseCount    = kwargs.get('MinCases', 30) # start trajectories with at least 30 confirmed cases
        self.__MeasureMinCount = kwargs.get('MeasureMinCount',5) # at least 5 countries have implemented measure
        self.__verbose         = kwargs.get('verbose', False)
        
        # load data from DB files
        self.jhu_data          = cdc.CoronaData(**kwargs)
        self.measure_data      = mc.COVID19_measures(**kwargs)        
        self.measure_data.RemoveCountry('Diamond Princess')
        self.measure_data.RenameCountry('France (metropole)', 'France')
        self.measure_data.RenameCountry('South Korea', 'Korea, South')
        self.measure_data.RenameCountry('Czech Republic', 'Czechia')
        
        # set up internal storage
        self.__CVresults     = None
        self.__regrDF        = {}
    
    
    def GenerateRDF(self, shiftdays = 0, countrylist = None):
        if countrylist is None:
            countrylist = self.measure_data.countrylist
        
        regressionDF    = None
        measurecount    = self.measure_data.MeasureList(countrylist = countrylist, mincount = self.__MeasureMinCount, measure_level = 2)

        for country in countrylist:
            if (country in self.measure_data.countrylist) and (country in self.jhu_data.countrylist):

                observable                = self.jhu_data.CountryGrowthRates(country = country)['Confirmed'].values
                startdate, startindex     = '22/1/2020', 0
                if not self.__MinCaseCount is None:
                    startdate, startindex = self.jhu_data.DateAtCases(country = country, cases = self.__MinCaseCount, outputformat = '%d/%m/%Y', return_index = True)
                    observable            = observable[startindex:]

                if not self.__maxlen is None:
                    observable            = observable[:np.min([self.__maxlen,len(observable) + 1])]
                    
                obslen                    = len(observable)
                
                DF_country = self.measure_data.ImplementationTable(country           = country,
                                                                   measure_level     = 2,
                                                                   startdate         = startdate,
                                                                   enddate           = self.jhu_data.FinalDate(country),
                                                                   shiftdays         = shiftdays,
                                                                   maxlen            = obslen,
                                                                   clean_measurename = True)
                # remove measures not in list
                for measurename in DF_country.columns:
                    if measurename not in measurecount.keys():
                        DF_country.drop(labels = measurename, axis = 'columns')
                
                DF_country['Country']     = country
                DF_country['Observable']  = observable

                if regressionDF is None:
                    regressionDF = DF_country
                else:
                    regressionDF = pd.concat([regressionDF,DF_country], ignore_index = True, sort = False)
    
        # not implemented measures should be NaN values, set them to 0
        regressionDF.fillna(0, inplace = True)
        
        return regressionDF
            
    
    def RegressionDF(self,shiftdays = 0):
        if not shiftdays in self.__regrDF.keys():
            self.__regrDF[shiftdays] = self.GenerateRDF(shiftdays = shiftdays)
        
        return self.__regrDF[shiftdays]
    
        
    def RunCV(self, shiftdaylist = [0], alphalist = [1e-5], verbose = None, countrywise_crossvalidation = True, crossvalcount = 10, outputheader = {}):
        if verbose is None: verbose = self.__verbose
        
        for shiftdays, alpha in itertools.product(shiftdaylist,alphalist):
            if verbose: print('{:3d} {:.6f} {:>15s}'.format(shiftdays,alpha, 'computing'), end = '\r', flush = True)
            
            measurelist       = list(self.RegressionDF(shiftdays).columns)
            measurelist.remove('Observable')
            measurelist.remove('Country')
            
            formula           = 'Observable ~ C(Country) + ' + ' + '.join(measurelist)
            
            if not countrywise_crossvalidation:
                # assign samples to each of the crossvalidation chunks
                datalen       = len(self.RegressionDF(shiftdays))
                chunklen      = np.ones(crossvalcount,dtype = np.int) * (datalen // crossvalcount)
                chunklen[:datalen%crossvalcount] += 1
                samples       = np.random.permutation(np.concatenate([i*np.ones(chunklen[i],dtype = np.int) for i in range(crossvalcount)]))
            else:
                countrylist   = list(self.RegressionDF(shiftdays)['Country'].unique())
                crossvalcount = len(countrylist)
                samples       = np.concatenate([i * np.ones(len(self.RegressionDF(shiftdays)[self.RegressionDF(shiftdays)['Country'] == countrylist[i]])) for i in range(crossvalcount)])
            
            for xv_index in range(crossvalcount):
                trainidx      = (samples != xv_index)
                testidx       = (samples == xv_index)
                trainmodel    = smf.ols(formula = formula, data = self.RegressionDF(shiftdays)[trainidx])
                testmodel     = smf.ols(formula = formula, data = self.RegressionDF(shiftdays)[testidx])
            
                trainresults  = trainmodel.fit_regularized(alpha = alpha, L1_wt = 1)

                test_params   = []
                for paramname in testmodel.exog_names:
                    if paramname in trainresults.params.keys():
                        test_params.append(trainresults.params[paramname])
                    else:
                        test_params.append(0)

                obs_train     = np.array(trainmodel.endog)
                obs_test      = np.array(testmodel.endog)
                pred_train    = trainmodel.predict(trainresults.params)
                pred_test     = testmodel.predict(test_params)
                    
                # store results in dict
                result_dict                     = {'shiftdays': shiftdays, 'alpha': alpha}
                if countrywise_crossvalidation:
                    result_dict['Iteration']    = countrylist[xv_index]
                else:
                    result_dict['Iteration']    = xv_index
                result_dict['Loglike Training'] = trainmodel.loglike(trainresults.params)
                result_dict['Loglike Test']     = testmodel.loglike(np.array(test_params))
                result_dict['R2 Training']      = 1 - np.sum((obs_train - pred_train)**2)/np.sum((obs_train - np.mean(obs_train))**2)
                result_dict['R2 Test']          = 1 - np.sum((obs_test - pred_test)**2)/np.sum((obs_test - np.mean(obs_test))**2)
                result_dict['RSS Training']     = np.sum((obs_train - pred_train)**2)
                result_dict['RSS Test']         = np.sum((obs_test - pred_test)**2)

                result_dict.update({k:v for k,v in trainresults.params.items()})
                
                # append dict to df
                if self.__CVresults is None:
                    self.__CVresults = pd.DataFrame({k:np.array([v]) for k,v in result_dict.items()})
                else:
                    self.__CVresults = self.__CVresults.append(result_dict, ignore_index = True)
            
            if verbose: print('{:3d} {:.6f} {:>15s}'.format(shiftdays,alpha, datetime.datetime.now().strftime('%H:%M:%S')))
            

    def StoreResults(self, store_file = None, reset = False):
        try:        self.__CVresults.to_csv(store_file)
        except:     pass
        if reset:   self.__CVresults = None

        
    def __getattr__(self,key):
        if key == 'CVresults':
            return self.__CVresults

