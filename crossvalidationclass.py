#basics
import numpy as np
import pandas as pd

import os
import itertools
import datetime
import time
import textwrap

# plotting
import matplotlib
import matplotlib.pyplot as plt

# statistics
import statsmodels.api as sm
import statsmodels.formula.api as smf

# datawrappers
import coronadataclass as cdc
import measureclass as mc




class CrossValidation(object):
    def __init__(self, **kwargs):
        self.__MinCaseCount             = kwargs.get('MinCases', 30) # start trajectories with at least 30 confirmed cases
        self.__MeasureMinCount          = kwargs.get('MeasureMinCount',5) # at least 5 countries have implemented measure
        self.__verbose                  = kwargs.get('verbose', True)
        self.__cvres_filename           = kwargs.get('CVResultsFilename',None)
        self.__external_observable_file = kwargs.get('ExternalObservableFile',None)
        self.__external_indicators_file = kwargs.get('ExternalIndicatorsFile',None)
        self.__observable_name          = kwargs.get('ObservableName','Confirmed')
        self.__maxlen                   = kwargs.get('MaxObservableLength',None)
        self.__finaldate                = kwargs.get('FinalDate',None)
        self.__finaldatefile            = kwargs.get('FinalDateFile',None)
        self.__finaldatefrommeasureDB   = kwargs.get('FinalDateFromDB',False)
        self.__extendfinaldateshiftdays = kwargs.get('FinalDateExtendWithShiftdays',False)
        self.colornames                 = kwargs.get('ColorNames',None)
        
        
        # load data from DB files
        self.jhu_data          = cdc.CoronaData(**kwargs)
        self.measure_data      = mc.COVID19_measures(**kwargs)        
        self.measure_data.RemoveCountry('Diamond Princess')
        self.measure_data.RenameCountry('South Korea', 'Korea, South')
        self.measure_data.RenameCountry('Czech Republic', 'Czechia')
        self.measure_data.RenameCountry('Republic of Ireland', 'Ireland')
        self.measure_data.RenameCountry('Taiwan', 'Taiwan*')
        
        
        self.__UseExternalObs            = False
        if not self.__external_observable_file is None:
            self.__UseExternalObs        = True
            self.__ExternalObservables   = pd.read_csv(self.__external_observable_file)
        
        self.__UseExternalIndicators     = False
        if not self.__external_indicators_file is None:
            self.__UseExternalIndicators = True
            self.__ExternalIndicators    = pd.read_csv(self.__external_indicators_file, index_col = 'Country')
            self.ExternalIndicatorsNames = pd.DataFrame({'Indicator':list(self.__ExternalIndicators.columns)})
            self.ExternalIndicatorsNames.index = [self.measure_data.CleanUpMeasureName(indicator) for indicator in self.ExternalIndicatorsNames['Indicator']]
        
        # set up internal storage
        self.CVresults                   = None
        self.__regrDF                    = {}
    
        if not self.__cvres_filename is None:
            self.LoadCVResults(filename  = self.__cvres_filename)
    
        #self.colornames = ['#f563e2','#609cff','#00bec4','#00b938','#b79f00','#f8766c', '#75507b'] # Amelie's color scheme
        if self.colornames is None:
            self.colornames              = [cn.upper() for cn in matplotlib.colors.TABLEAU_COLORS.keys() if (cn.upper() != 'TAB:WHITE' and cn.upper() != 'TAB:GRAY')]
        self.L1colors                    = {L1name:self.colornames[i % len(self.colornames)] for i,L1name in enumerate(self.measure_data.MeasureList(mincount = 1).sort_values(by = 'Measure_L1')['Measure_L1'].unique())}
        self.L1colors['Country Effects'] = '#babdb6'
        
        
        if not self.__finaldatefile is None:
            self.__FinalDateCountries = pd.read_csv(self.__finaldatefile)
        
        self.finalModels                 = []
        self.finalResults                = []
        self.finalCV                     = None
        self.finalParameters             = []
        
        self.__kwargs_for_pickle         = kwargs
        
        
    
    def addDF(self, df = None, new = None):
        if not new is None:
            if df is None:
                return new
            else:
                return pd.concat([df,new], ignore_index = True, sort = False)
        else:
            return df



    def HaveCountryData(self, country = None):
        if self.__UseExternalObs:
            if country in self.__ExternalObservables['country'].unique():
                return True
        else:
            if country in self.jhu_data.countrylist:
                return True
        return False
    
    
    
    def GetObservable(self, country, shiftdays = None):
        if not self.__UseExternalObs:
            observable = self.jhu_data.CountryGrowthRates(country = country)[self.__observable_name].values 
            
            startdate                 = self.jhu_data.DateStart(country = country)
            if not self.__MinCaseCount is None:
                startdate, startindex = self.jhu_data.DateAtCases(country = country, cases = self.__MinCaseCount, outputformat = '%d/%m/%Y', return_index = True)
                observable            = observable[startindex:]

        else:
            # import from Nils' files
            c_index               = np.array(self.__ExternalObservables['country'] == country)
            observable            = self.__ExternalObservables[c_index]['Median(R)'].values[1:] # first entry is usually 0
            startdate             = (datetime.datetime.strptime(self.__ExternalObservables[c_index]['startdate'].values[0],'%Y-%m-%d') + datetime.timedelta(days = 1)).strftime('%d/%m/%Y')            
            startindex            = (datetime.datetime.strptime(startdate,'%d/%m/%Y') - datetime.datetime.strptime('22/1/2020','%d/%m/%Y')).days
        
        startdate_dt = datetime.datetime.strptime(startdate,'%d/%m/%Y')
        possible_end_dates = [startdate_dt + datetime.timedelta(days = len(observable) - 1)]
        
        shiftdays_dt = datetime.timedelta(days = 0)
        if not shiftdays is None:
            shiftdays_dt = datetime.timedelta(days = int(shiftdays))
        
        if not self.__maxlen is None:
            possible_end_dates.append(startdate_dt + datetime.timedelta(days = self.__maxlen))

        if not self.__finaldate is None:
            possible_end_dates.append(datetime.datetime.strptime(self.__finaldate,'%d/%m/%Y') + shiftdays_dt)
            
        if not self.__finaldatefrommeasureDB is None:
            possible_end_dates.append(datetime.datetime.strptime(self.measure_data.FinalDates(countrylist = [country])['Date'].values[0],'%d/%m/%Y') + shiftdays_dt)
        
        enddate = np.min(possible_end_dates).strftime('%d/%m/%Y')
        
        obslen = (datetime.datetime.strptime(enddate,'%d/%m/%Y') - datetime.datetime.strptime(startdate,'%d/%m/%Y')).days + 1

        if obslen > 0:
            observable = observable[:obslen]
            return observable, startdate, enddate
        else:
            return None,None,None
    
    
    def GenerateRDF(self, shiftdays = 0, countrylist = None):
        if countrylist is None:
            countrylist = self.measure_data.countrylist
        
        regressionDF    = None
        measurelist    = self.measure_data.MeasureList(mincount = self.__MeasureMinCount, measure_level = 2, enddate = self.__finaldate)

        for country in countrylist:
            if (country in self.measure_data.countrylist) and self.HaveCountryData(country):
                
                extend_shiftdays = None
                if self.__extendfinaldateshiftdays: extend_shiftdays = shiftdays
                
                observable, startdate, enddate = self.GetObservable(country, shiftdays = extend_shiftdays)
                
                if not observable is None:
                    DF_country = self.measure_data.ImplementationTable( country           = country,
                                                                        measure_level     = 2,
                                                                        startdate         = startdate,
                                                                        enddate           = enddate,
                                                                        shiftdays         = shiftdays,
                                                                        clean_measurename = True)
                    # remove measures not in list
                    for measurename in DF_country.columns:
                        if measurename not in measurelist.index:
                            DF_country.drop(labels = measurename, axis = 'columns')
                    
                    DF_country['Country']     = str(country)
                    DF_country['Observable']  = observable
                    
                    if self.__UseExternalIndicators:
                        for indicator in self.ExternalIndicatorsNames.iterrows():
                            if country in self.__ExternalIndicators.index:
                                DF_country[indicator[0]] = self.__ExternalIndicators.loc[country,indicator[1][0]]
                            else:
                                DF_country[indicator[0]] = 0

                    regressionDF = self.addDF(regressionDF,DF_country)

        # not implemented measures should be NaN values, set them to 0
        regressionDF.fillna(0, inplace = True)
        
        return regressionDF
            
    
    
    def RegressionDF(self,shiftdays = 0):
        if not shiftdays in self.__regrDF.keys():
            self.__regrDF[shiftdays] = self.GenerateRDF(shiftdays = shiftdays)
        return self.__regrDF[shiftdays]
    
    
    
    def SingleParamCV(self, shiftdays = None, alpha = None, crossvalcount = None, outputheader = {}, L1_wt = 1):
        
        curResDF          = None
        measurelist       = list(self.RegressionDF(shiftdays).columns)
        measurelist.remove('Observable')
        measurelist.remove('Country')
        
        formula           = 'Observable ~ C(Country) + ' + ' + '.join(measurelist)
        
        countrylist       = list(self.RegressionDF(shiftdays)['Country'].unique())
        
        if crossvalcount is None:            crossvalcount = len(countrylist)
        if crossvalcount > len(countrylist): crossvalcount = len(countrylist)
            
        countrylen        = len(countrylist)
        chunklen          = np.ones(crossvalcount, dtype = np.int) * (countrylen // crossvalcount)
        chunklen[:countrylen % crossvalcount] += 1
        # sample_countries should be a list (of the same length as countrylist) with values corresponding to which test group the country is assigned
        sample_countries  = np.random.permutation(np.concatenate([i * np.ones(chunklen[i],dtype = np.int) for i in range(crossvalcount)]))
        # extend this list to the whole dataset
        samples           = np.concatenate([s * np.ones(len(self.RegressionDF(shiftdays)[self.RegressionDF(shiftdays)['Country'] == countrylist[i]]), dtype = np.int) for i,s in enumerate(sample_countries)])

        for xv_index in range(crossvalcount):
            testcountries = [countrylist[i] for i,s in enumerate(sample_countries) if s == xv_index]
            trainidx      = (samples != xv_index)
            testidx       = (samples == xv_index)
            trainmodel    = smf.ols(formula = formula, data = self.RegressionDF(shiftdays)[trainidx])
            testmodel     = smf.ols(formula = formula, data = self.RegressionDF(shiftdays)[testidx])
        
            trainresults  = trainmodel.fit_regularized(alpha = alpha, L1_wt = L1_wt)

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
            result_dict                          = {'shiftdays': shiftdays, 'alpha': alpha}
            result_dict['Test Countries']        = '; '.join([str(c) for c in testcountries])
            result_dict['Test Sample Size']      = np.sum([len(self.RegressionDF(shiftdays)[self.RegressionDF(shiftdays)['Country'] == country]) for country in testcountries])
            result_dict['Training Sample Size']  = len(self.RegressionDF(shiftdays)) - result_dict['Test Sample Size']
            result_dict['Loglike Training']      = trainmodel.loglike(trainresults.params)
            result_dict['Loglike Test']          = testmodel.loglike(np.array(test_params))
            result_dict['R2 Training']           = 1 - np.sum((obs_train - pred_train)**2)/np.sum((obs_train - np.mean(obs_train))**2)
            result_dict['R2 Test']               = 1 - np.sum((obs_test - pred_test)**2)/np.sum((obs_test - np.mean(obs_test))**2)
            result_dict['RSS Training']          = np.sum((obs_train - pred_train)**2)
            result_dict['RSS Test']              = np.sum((obs_test - pred_test)**2)

            result_dict.update({k:v for k,v in trainresults.params.items()})
            
            curResDF = self.addDF(curResDF,pd.DataFrame({k:np.array([v]) for k,v in result_dict.items()}))
        
        return curResDF
    
    
        
    def RunCV(self, shiftdaylist = [0], alphalist = [1e-5], verbose = None, crossvalcount = None, outputheader = {}, L1_wt = 1):
        if verbose is None: verbose = self.__verbose
        if L1_wt != 1:
            outputheader.update({'L1_wt':L1_wt})
            
        for shiftdays, alpha in itertools.product(shiftdaylist,alphalist):
            if verbose: print('{:3d} {:.6f} {:>15s}'.format(shiftdays,alpha, 'computing'), end = '\r', flush = True)
            
            curResDF         = self.SingleParamCV(shiftdays = shiftdays, alpha = alpha, crossvalcount = crossvalcount, outputheader = outputheader, L1_wt = L1_wt)
            self.CVresults = self.addDF(self.CVresults,curResDF)
            
            if verbose: print('{:3d} {:.6f} {:>15s}'.format(shiftdays,alpha, datetime.datetime.now().strftime('%H:%M:%S')))
        
        # automatically store latest CrossValidation run
        self.CVresults.to_csv('latest_CVrun.csv')


    def SaveCVResults(self, filename = None, reset = False):
        if (not filename is None) and (not self.CVresults is None):
            self.CVresults.to_csv(filename)
            if self.__verbose: print('# saving CV results as "{}"'.format(filename))
        if reset:   self.CVresults = None



    def LoadCVResults(self, filename, reset = True):
        if os.path.exists(filename):
            if self.__verbose:print('# loading CV results from "{}"'.format(filename))
            if reset:   self.CVresults = pd.read_csv(filename,index_col = 0)
            else:       self.CVresults = self.addDF(self.CVresults,pd.read_csv(filename,index_col = 0))
        else:
            raise IOError



    def ProcessCVresults(self, CVresults = None):
        if CVresults is None: CVresults = self.CVresults.copy(deep = True)
        CVresults['alpha'] = CVresults['alpha'].map('{:.6e}'.format) # convert alpha to a format that can be grouped properly
        CVresults =  CVresults.groupby(['shiftdays','alpha'], as_index = False).agg(
            { 'Loglike Test':['mean','std'],
              'Loglike Training':['mean','std'],
              'R2 Test': ['mean','std'],
              'R2 Training': ['mean','std'],
              'RSS Training' : ['sum'],
              'RSS Test': ['sum'],
              'Test Sample Size':['sum'],
              'Training Sample Size':['sum']
            })
        CVresults.columns = [ 'shiftdays','alpha',
                              'Loglike Test','Loglike Test Std',
                              'Loglike Training','Loglike Training Std',
                              'R2 Test','R2 Test Std',
                              'R2 Training','R2 Training Std',
                              'RSS Training Sum',
                              'RSS Test Sum',
                              'Test Sample Size',
                              'Training Sample Size'
                            ]
        CVresults['RSS per datapoint Training'] = CVresults['RSS Training Sum']/CVresults['Training Sample Size']
        CVresults['RSS per datapoint Test'] = CVresults['RSS Test Sum']/CVresults['Test Sample Size']
        
        CVresults['alpha'] = CVresults['alpha'].astype(np.float64) # return to numbers
        CVresults.sort_values(by = ['shiftdays','alpha'], inplace = True)
        return CVresults



    def ComputeFinalModels(self, modelparameters = [(6,1e-3)], L1_wt = 1, crossvalcount = None):
        self.finalModels     = []
        self.finalResults    = []
        self.finalCV         = None
        self.finalParameters = []
        
        for i, (shiftdays, alpha) in enumerate(modelparameters):
            self.finalParameters.append((shiftdays,alpha))
            
            finalCV      = self.SingleParamCV(shiftdays = shiftdays, alpha = alpha, outputheader = {'modelindex':i}, crossvalcount = crossvalcount)
            self.finalCV = self.addDF(self.finalCV, finalCV)
            
            measurelist  = list(self.RegressionDF(shiftdays).columns)
            measurelist.remove('Observable')
            measurelist.remove('Country')
            formula      = 'Observable ~ C(Country) + ' + ' + '.join(measurelist)
            
            self.finalModels.append(smf.ols(data = self.RegressionDF(shiftdays), formula = formula))
            self.finalResults.append(self.finalModels[i].fit_regularized(alpha = alpha, L1_wt = L1_wt))
    

    
    def FinalMeasureEffects(self, drop_zeros = False, rescale = True, include_countries = False, additional_columns = []):
        if not self.finalCV is None:
            finalCVrelative                = self.finalCV.copy(deep = True).drop(columns = 'Test Countries', axis = 0).fillna(0)
            if rescale: finalCVrelative    = finalCVrelative.divide(self.finalCV['Intercept'],axis = 0)
            finalCVrelative                = finalCVrelative.quantile([.5,.025,.975]).T
            finalCVrelative.columns        = ['median', 'low', 'high']
            if drop_zeros: finalCVrelative = finalCVrelative[(finalCVrelative['median'] != 0) | (finalCVrelative['low'] != 0) | (finalCVrelative['high'] != 0)]
            
            if len(additional_columns) > 0:
                try:
                    finalCVadditional             = self.finalCV.copy(deep = True).drop(columns = 'Test Countries', axis = 0).fillna(0)
                    if rescale: finalCVadditional = finalCVadditional.divide(self.finalCV['Intercept'],axis = 0)
                    finalCVadditional             = finalCVadditional.apply(additional_columns).T
                    finalCVadditional.columns     = additional_columns
                    
                    finalCVrelative        = finalCVrelative.merge(finalCVadditional, left_index = True, right_index = True, how = 'left')
                except:
                    pass
            
            fCV_withNames                  = self.measure_data.MeasureList(mincount = self.__MeasureMinCount, enddate = self.__finaldate, measure_level = 2).merge(finalCVrelative, how = 'inner', left_index = True, right_index = True).drop(columns = 'Countries with Implementation', axis = 0)
            
            if include_countries:
                countryDF                  = pd.DataFrame({'Measure_L2':[country[13:].split(']')[0] for country in finalCVrelative.index if country[:3] == 'C(C']})
                countryDF['Measure_L1']    = 'Country Effects'
                countryDF.index            = [country for country in finalCVrelative.index if country[:3] == 'C(C']
                fCV_withCountries          = countryDF.merge(finalCVrelative, how = 'inner', left_index = True, right_index = True)
                fCV_withNames              = self.addDF(fCV_withNames,fCV_withCountries)
            
            fCV_withNames.sort_values(by   = ['median','high','low'], inplace = True)
            return fCV_withNames
        else:
            return None


        

    # ************************************************************************************
    # ** plotting output 
    # ************************************************************************************


    def PlotTrajectories(self, filename = 'trajectories.pdf', columns = 2):
        modelcount  = len(self.finalModels)
        if modelcount > 0:
            countrylist = [country[13:].strip(']') for country in self.finalModels[0].data.xnames if country[:10] == 'C(Country)']
            ycount = len(countrylist) // columns
            if len(countrylist) % columns != 0:
                ycount += 1
            
            fig,axes = plt.subplots(ycount, columns, figsize = (15,6.*ycount/columns))
            ax = axes.flatten()
            for j,country in enumerate(countrylist):
                for m,(model,results) in enumerate(zip(self.finalModels,self.finalResults)):
                    country_mask = np.array(self.RegressionDF(self.finalParameters[m][0])['Country'] == country)
                    if m == 0:
                        ax[j].plot(model.endog[country_mask], lw = 5, label = 'data')
                    ax[j].plot(results.predict()[country_mask], lw = 2, linestyle = '--', label = '({:d}, {:.2f})'.format(self.finalParameters[m][0],np.log10(self.finalParameters[m][1])))
                ax[j].annotate(country,[5,0.42], ha = 'left', fontsize = 15)
                ax[j].set_ylim([0,0.5])
                ax[j].set_xlim([0,60])
                ax[j].legend()
            fig.tight_layout()
            fig.savefig(filename)
       
    
    
    def PlotCVresults(self, filename = 'CVresults.pdf', shiftdayrestriction = None, ylim = (0,1), figsize = (15,6)):
        processedCV = self.ProcessCVresults().sort_values(by = 'alpha')
        
        fig,axes = plt.subplots(1,2,figsize = figsize, sharey = True)
        ax = axes.flatten()
        
        shiftdaylist = np.array(processedCV['shiftdays'].unique(), dtype = np.int)
        shiftdaylist.sort()
        
        if shiftdayrestriction is None:
            shiftdayrestriction = shiftdaylist
        
        for shiftdays in shiftdaylist:
            if shiftdays in shiftdayrestriction:
                s_index = (processedCV['shiftdays'] == shiftdays).values
                alphalist = processedCV[s_index]['alpha']
                ax[0].plot(alphalist, processedCV[s_index]['RSS Test Sum']/processedCV[s_index]['Test Sample Size'],         label = 's = {}'.format(shiftdays), lw = 3, alpha = .8)
                ax[1].plot(alphalist, processedCV[s_index]['RSS Training Sum']/processedCV[s_index]['Training Sample Size'], label = 's = {}'.format(shiftdays), lw = 3, alpha = .8)
        
        for i in range(2):
            ax[i].legend()
            ax[i].set_xlabel(r'Penalty parameter $\alpha$')
            ax[i].set_xscale('log')
            ax[i].grid()
        ax[0].set_ylim(ylim)
        
        ax[0].set_ylabel('RSS/datapoint Test')
        ax[1].set_ylabel('RSS/datapoint Training')
        
        fig.tight_layout()
        fig.savefig(filename)
    
    
    
    def PlotCVAlphaSweep(self, shiftdays = None, filename = 'crossval_evaluation.pdf', country_effects = False, measure_effects = True, ylim = (-1,1), figsize = (15,10),verticallines = []):
        if isinstance(shiftdays,int):
            shiftdaylist = [shiftdays]
        elif isinstance(shiftdays,(tuple,list,np.ndarray)):
            shiftdaylist = shiftdays
        elif shiftdays is None:
            shiftdaylist = list(self.CVresults['shiftdays'].unique())
            
        fig,axes = plt.subplots(len(shiftdaylist),1,figsize=figsize)
        ax_index = 0
        
        grouped_parameters = self.measure_data.MeasureList(mincount = self.__MeasureMinCount, enddate = self.__finaldate).merge(self.CVresults.drop(columns = ['Test Countries']).divide(self.CVresults['Intercept'],axis = 0).T,left_index=True,right_index=True,how='inner').T.merge(self.CVresults[['shiftdays','alpha']],left_index=True,right_index=True,how='inner').fillna(0)
        grouped_parameters['alpha'] = grouped_parameters['alpha'].map('{:.6e}'.format)
        grouped_parameters = grouped_parameters.groupby(by = ['shiftdays','alpha'],as_index=False)
    
        median_measures = grouped_parameters.quantile(.5)
        low_measures    = grouped_parameters.quantile(.025)
        high_measures   = grouped_parameters.quantile(.975)

        median_measures['alpha'] = median_measures['alpha'].astype(np.float64)
        low_measures['alpha']    = low_measures['alpha'].astype(np.float64)
        high_measures['alpha']   = high_measures['alpha'].astype(np.float64)

        median_measures.sort_values(by = ['shiftdays','alpha'], inplace = True)
        low_measures.sort_values(by = ['shiftdays','alpha'], inplace = True)
        high_measures.sort_values(by = ['shiftdays','alpha'], inplace = True)

        measuredict = {index:l1name for index,l1name in self.measure_data.MeasureList(mincount = self.__MeasureMinCount, enddate = self.__finaldate)['Measure_L1'].items()}
        
        if country_effects:
            countrycolor    = '#777777'
            countrylist     = pd.DataFrame({'Country':[country for country in restrictedCV.columns if country[:3] == 'C(C']})

            grouped_country = countrylist.merge(self.CVresults.drop(columns = ['Test Countries']).divide(self.CVresults['Intercept'],axis = 0).T,left_index=True,right_index=True,how='inner').T.merge(self.CVresults[['shiftdays','alpha']],left_index=True,right_index=True,how='inner').fillna(0)
            grouped_country['alpha'] = grouped_country['alpha'].map('{:.6e}'.format)
            grouped_country = grouped_country.groupby(by = ['shiftdays','alpha'],as_index=False)

            median_country  = grouped_country.quantile(.5)
            low_country     = grouped_country.quantile(.025)
            high_country    = grouped_country.quantile(.975)

            median_country['alpha'] = median_country['alpha'].astype(np.float64)
            low_country['alpha']    = low_country['alpha'].astype(np.float64)
            high_country['alpha']   = high_country['alpha'].astype(np.float64)

            median_country.sort_values(by = ['shiftdays','alpha'],inplace = True)
            low_country.sort_values(by = ['shiftdays','alpha'],inplace = True)
            high_country.sort_values(by = ['shiftdays','alpha'],inplace = True)


        for shiftdays in shiftdaylist:
            if shiftdays in self.CVresults['shiftdays']:
                
                if len(shiftdaylist) == 1:
                    ax = axes
                else:
                    ax = axes[ax_index]
                    ax_index += 1

                if country_effects:
                    s_index = (median_country['shiftdays'] == shiftdays)
                    alphalist = median_country[s_index]['alpha'].values
                    for country in [c for c in median_country.columns if c != 'shiftdays' and c != 'alpha']:
                        ax.plot(alphalist,median_country[country].values,c = countrycolor,lw = .5)
                        ylow  = np.array(low_country[country].values,dtype=np.float)
                        yhigh = np.array(high_country[country].values,dtype=np.float)
                        ax.fill_between(alphalist,y1 = ylow,y2=yhigh,color = countrycolor,alpha = .05)

                s_index = (median_measures['shiftdays'] == shiftdays)
                alphalist = median_measures[s_index]['alpha'].values

                for measure in [m for m in median_measures.columns if m != 'shiftdays' and m != 'alpha']:
                    color = self.L1colors[measuredict[measure]]
                    ax.plot(alphalist,median_measures[s_index][measure].values, c = color, lw = 2)
                    ylow  = np.array(low_measures[s_index][measure].values,dtype=np.float)
                    yhigh = np.array(high_measures[s_index][measure].values,dtype=np.float)
                    ax.fill_between(alphalist,y1 = ylow,y2=yhigh,color = color,alpha = .05)

                legendhandles = [matplotlib.lines.Line2D([0],[0],c = value,label = key,lw=2) for key,value in self.L1colors.items()]
                if country_effects:
                    legendhandles += [matplotlib.lines.Line2D([0],[0],c = countrycolor,label = 'Country Effects',lw=.5)]
                
                for alpha in verticallines:
                    ax.vline(x,zorder = 0, lw = 2, alpha = .5, c = '#000000')
                
                ax.legend(handles = legendhandles )
                ax.set_xlabel(r'Penalty Parameter $\alpha$')
                ax.set_ylabel(r'Relative Effect Size')
                ax.annotate('shiftdays = ${:d}$'.format(shiftdays),[np.power(np.min(alphalist),.97)*np.power(np.max(alphalist),0.03),np.max(ylim)*.9])
                ax.set_ylim(ylim)
                ax.set_xscale('log')            
        
        fig.tight_layout()
        fig.savefig(filename)
    
    

    #def PlotMeasureListValues(self, filename = 'measurelist_values.pdf'):
        #def significanceColor(beta):
            #if beta   >  0.00: return 'red'
            #elif beta == 0.00: return 'lightgray'
            #else:              return 'black'

        ## amelies colorscheme...                 
        #colornames  = ['gray','#f563e2','#609cff','#00bec4','#00b938','#b79f00','#f8766c', '#75507b']

        ## collect measure names for labels
        #measurelist = self.measure_data.MeasureList(mincount = self.__MeasureMinCount, measure_level = 2, enddate = self.__finaldate)
        #countrylist = [country[13:].strip(']') for country in self.finalModels[0].data.xnames if country[:10] == 'C(Country)']
        #modelcount  = len(self.finalModels)
        #intercept   = [self.finalResults[m].params['Intercept'] for m in range(modelcount)]

        ## internal counters to determine position to plot
        #ypos = 0
        #groupcolor = 0

        ## define positions for various elements
        #label_x        = 1
        #label_x_header = .6
        #value_x        = 12
        #value_dx       = 2
        #boxalpha       = .15
        
        ## start plot
        #fig,ax = plt.subplots(figsize = (14,23))
        
        ## country effects
        #ax.annotate('Country specific effects',[label_x_header, len(countrylist)], c = colornames[groupcolor], weight = 'bold' )
        #background = plt.Rectangle([label_x - .6, ypos - .65], value_x + (modelcount-1)*2 + .6, len(countrylist) + 1.8, fill = True, fc = colornames[groupcolor], alpha = boxalpha, zorder = 10)
        #ax.add_patch(background)
        #for country in countrylist[::-1]:
            #ax.annotate(country, [label_x, ypos], c= colornames[groupcolor])
            #for m in range(modelcount):
                #beta_val = self.finalResults[m].params['C(Country)[T.{}]'.format(country)] / intercept[m]
                #ax.annotate('{:6.0f}%'.format(beta_val*100),[value_x + m * value_dx, ypos], c = significanceColor(beta_val), ha = 'right')
            #ypos += 1
        #groupcolor += 1
        #ypos+=2 

        ## measure effects
        #for l1 in L1names[::-1]:
            #ax.annotate(l1,[label_x_header, ypos + len(measure_level_dict[l1])], c = colornames[groupcolor], weight = 'bold')
            #L2names = list(measure_level_dict[l1].keys())
            #L2names.sort()
            
            #background = plt.Rectangle([label_x - .6, ypos - .65], value_x + 2*(modelcount-1) + .6, len(measure_level_dict[l1]) + 1.8, fill = True, fc = colornames[groupcolor], alpha = boxalpha, zorder = 10)
            #ax.add_patch(background)
            
            #for l2 in L2names[::-1]:
                #ax.annotate(l2,[label_x,ypos],c = colornames[groupcolor])
                #for m in range(modelcount):
                    #beta_val = self.finalResults[m].params[measure_level_dict[l1][l2]] / intercept[m]
                    #ax.annotate('{:6.0f}%'.format(beta_val*100),[value_x + m * value_dx, ypos], c = significanceColor(beta_val), ha = 'right')
                #ypos+=1
            #ypos+=2
            #groupcolor += 1

        ## header
        #ax.annotate(r'shiftdays $s$',[label_x,ypos+1])
        #ax.annotate(r'Penality parameter $\log_{10}(\alpha)$',[label_x,ypos])
        #for m in range(modelcount):
            #ax.annotate('{}'.format(self.finalParameters[m][0]),               [value_x + m*value_dx,ypos+1],ha='right')
            #ax.annotate('{:.1f}'.format(np.log10(self.finalParameters[m][1])), [value_x + m*value_dx,ypos],ha='right')        

        ## adjust and save
        #ax.set_xlim([0,value_x + 2 * modelcount])
        #ax.set_ylim([-1,ypos+2])
        #ax.axis('off')
        #fig.savefig(filename)
        


    def PlotMeasureListSorted(self, filename = 'measurelist_sorted.pdf', drop_zeros = False, figsize = (15,30), labelsize = 40, blacklines = [0], graylines = [-30,-20,-10,10], border = 2, title = '', textbreak = 40, include_countries = False, rescale = True):
        # get plotting area
        minplot      = np.min(blacklines + graylines)
        maxplot      = np.max(blacklines + graylines)

        # function to plot one row in DF
        def PlotRow(ax, ypos = 1, values = None, color = '#ffffff', boxalpha = .2, textbreak = 40):
            count_labels = len(values) - 3
            ax.plot(values['median'],[ypos], c = self.L1colors[values[0]], marker = 'D')
            ax.plot([values['low'],values['high']],[ypos,ypos], c = self.L1colors[values[0]], lw = 2)
            background = plt.Rectangle([1e-2 * (minplot - border - count_labels * labelsize), ypos - .4], 1e-2*(count_labels*labelsize + maxplot + border - minplot), .9, fill = True, fc = color, alpha = boxalpha, zorder = 10)
            ax.add_patch(background)
            for i in range(count_labels):
                ax.annotate(textwrap.shorten(str(values[i]), width = textbreak), [1e-2*(minplot - (count_labels - i) * labelsize), ypos - .1])

        # setup
        measure_effects = self.FinalMeasureEffects(drop_zeros = drop_zeros, include_countries = include_countries, rescale = rescale)
        
        # actual plotting including vertical lines
        fig,ax = plt.subplots(figsize = figsize)
        for j,(index,values) in enumerate(measure_effects.iterrows()):
            PlotRow(ax, ypos = -j,values = values, color = self.L1colors[values[0]], textbreak = textbreak)
        for x in blacklines:
            ax.plot([1e-2 * x,1e-2 * x],[0.7,-j-0.5], lw = 2, c = 'black',zorder = -2)
            ax.annotate('{:.0f}%'.format(x),[1e-2*x,0.9],fontsize = 12, c = 'gray', ha = 'center')
        for x in graylines:
            ax.plot([1e-2 * x,1e-2 * x],[0.6,-j-0.4], lw = 1, c = 'gray',zorder = -2)
            ax.annotate('{:.0f}%'.format(x),[1e-2*x,0.9],fontsize = 12, c = 'gray', ha = 'center')
        
        # format output
        if title != '':
            ax.annotate(title,[1e-2 * (-(len(measure_effects.columns) - 3) * labelsize + minplot),1.2], fontsize = 12, weight = 'bold')
        ax.set_xlim([1e-2 * (-(len(measure_effects.columns) -3 ) * labelsize - 2*border + minplot), 1e-2 * (maxplot+border)])
        ax.set_ylim([-j-2,1.8])
        ax.axis('off')
        fig.savefig(filename)



    # ************************************************************************************
    # ** python stuff
    # ************************************************************************************

    # make CrossValidation object pickleable ...
    # (getstate, setstate) interacts with (pickle.dump, pickle.load)
    def __getstate__(self):
        return {'kwargs':          self.__kwargs_for_pickle,
                'CVresults':       self.CVresults,
                'finalModels':     self.finalModels,
                'finalResults':    self.finalResults,
                'finalCV':         self.finalCV,
                'finalParameters': self.finalParameters}
    
    def __setstate__(self,state):
        self.__init__(**state['kwargs'])
        self.CVresults      = state['CVresults']
        self.finalModels      = state['finalModels']
        self.finalResults     = state['finalResults']
        self.finalCV          = state['finalCV']
        self.finalParameters  = state['finalParameters']
                
