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
        self.__MinCaseCount    = kwargs.get('MinCases', 30) # start trajectories with at least 30 confirmed cases
        self.__MeasureMinCount = kwargs.get('MeasureMinCount',5) # at least 5 countries have implemented measure
        self.__verbose         = kwargs.get('verbose', False)
        self.__cvres_filename  = kwargs.get('CVResultsFilename',None)
        
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
    
        if not self.__cvres_filename is None:
            self.LoadCVResults(filename = self.__cvres_filename)
    
        self.__kwargs_for_pickle = kwargs
    
    
    
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


                regressionDF = self.addDF(regressionDF,DF_country)

        # not implemented measures should be NaN values, set them to 0
        regressionDF.fillna(0, inplace = True)
        
        return regressionDF
            
    
    
    def RegressionDF(self,shiftdays = 0):
        if not shiftdays in self.__regrDF.keys():
            self.__regrDF[shiftdays] = self.GenerateRDF(shiftdays = shiftdays)
        return self.__regrDF[shiftdays]
    
    
    
    def SingleParamCV(self, shiftdays = None, alpha = None, countrywise_crossvalidation = True, crossvalcount = 10, outputheader = {}, L1_wt = 1):
        
        curResDF          = None
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
            
            curResDF = self.addDF(curResDF,pd.DataFrame({k:np.array([v]) for k,v in result_dict.items()}))
        
        return curResDF
    
    
        
    def RunCV(self, shiftdaylist = [0], alphalist = [1e-5], verbose = None, countrywise_crossvalidation = True, crossvalcount = 10, outputheader = {}, L1_wt = 1):
        if verbose is None: verbose = self.__verbose
        if L1_wt != 1:
            outputheader.update({'L1_wt':L1_wt})
            
        for shiftdays, alpha in itertools.product(shiftdaylist,alphalist):
            if verbose: print('{:3d} {:.6f} {:>15s}'.format(shiftdays,alpha, 'computing'), end = '\r', flush = True)
            
            curResDF         = self.SingleParamCV(shiftdays = shiftdays, alpha = alpha, countrywise_crossvalidation = countrywise_crossvalidation, crossvalcount = crossvalcount, outputheader = outputheader, L1_wt = L1_wt)
            self.__CVresults = self.addDF(self.__CVresults,curResDF)
            
            if verbose: print('{:3d} {:.6f} {:>15s}'.format(shiftdays,alpha, datetime.datetime.now().strftime('%H:%M:%S')))
            


    def SaveCVResults(self, filename = None, reset = False):
        if (not filename is None) and (not self.__CVresults is None):
            self.__CVresults.to_csv(filename)
            if self.__verbose: print('# saving CV results as "{}"'.format(filename))
        if reset:   self.__CVresults = None



    def LoadCVResults(self, filename, reset = True):
        if os.path.exists(filename):
            if self.__verbose:print('# loading CV results from "{}"'.format(filename))
            if reset:   self.__CVresults = pd.read_csv(filename)
            else:       self.__CVresults = self.addDF(pd.read_csv(filename))
        else:
            raise IOError


    def ProcessCVresults(self, CVresults = None):
        if CVresults is None: CVresults = self.__CVresults
        CVresults =  CVresults.groupby(['shiftdays','alpha'], as_index = False).agg(
            { 'Loglike Test':['mean','std'],
              'Loglike Training':['mean','std'],
              'R2 Test': ['mean','std'],
              'R2 Training': ['mean','std'],
              'RSS Training' : ['sum'],
              'RSS Test': ['sum']
            })
        CVresults.columns = [ 'shiftdays','alpha',
                              'Loglike Test','Loglike Test Std',
                              'Loglike Training','Loglike Training Std',
                              'R2 Test','R2 Test Std',
                              'R2 Training','R2 Training Std',
                              'RSS Training Sum',
                              'RSS Test Sum'
                            ]
        CVresults.sort_values(by = ['shiftdays','alpha'], inplace = True)
        return CVresults



    def ComputeFinalModels(self, modelparameters = [(6,1e-3)], L1_wt = 1):
        self.finalModels     = []
        self.finalResults    = []
        self.finalCV         = None
        self.finalParameters = []
        
        for i, (shiftdays, alpha) in enumerate(modelparameters):
            self.finalParameters.append((shiftdays,alpha))
            
            finalCV      = self.SingleParamCV(shiftdays = shiftdays, alpha = alpha, outputheader = {'modelindex':i})
            self.finalCV = self.addDF(self.finalCV, finalCV)
            
            measurelist  = list(self.RegressionDF(shiftdays).columns)
            measurelist.remove('Observable')
            measurelist.remove('Country')
            formula      = 'Observable ~ C(Country) + ' + ' + '.join(measurelist)
            
            self.finalModels.append(smf.ols(data = self.RegressionDF(shiftdays), formula = formula))
            self.finalResults.append(self.finalModels[i].fit_regularized(alpha = alpha, L1_wt = L1_wt))
            


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
       
    
    
    def PlotCVresults(self, filename = 'CVresults.pdf', shiftdayrestriction = None):
        processedCV = self.ProcessCVresults().sort_values(by = 'alpha')
        totalrescale = 1./(len(self.__CVresults['Iteration'].unique()) - 1)
        
        
        fig,axes = plt.subplots(1,2,figsize = (15,6), sharey = True)
        ax = axes.flatten()
        
        shiftdaylist = np.array(processedCV['shiftdays'].unique(), dtype = np.int)
        shiftdaylist.sort()
        
        if shiftdayrestriction is None:
            shiftdayrestriction = shiftdaylist
        
        for shiftdays in shiftdaylist:
            if shiftdays in shiftdayrestriction:
                s_index = (processedCV['shiftdays'] == shiftdays).values
                alphalist = processedCV[s_index]['alpha']
                #ax[0].plot(alphalist, processedCV[s_index]['Loglike Test'],     label = 's = {}'.format(shiftdays))
                #ax[1].plot(alphalist, processedCV[s_index]['Loglike Training'], label = 's = {}'.format(shiftdays))
                ax[0].plot(alphalist, processedCV[s_index]['RSS Test Sum'],     label = 's = {}'.format(shiftdays), lw = 3, alpha = .8)
                ax[1].plot(alphalist, processedCV[s_index]['RSS Training Sum']*totalrescale,    label = 's = {}'.format(shiftdays), lw = 3, alpha = .8)
        
        for i in range(2):
            ax[i].legend()
            ax[i].set_xlabel(r'Penalty parameter $\alpha$')
            ax[i].set_xscale('log')
            ax[i].grid()
        ax[0].set_ylim([0,50])
        
        #ax[0].set_ylabel('Log Likelihood Test')
        #ax[1].set_ylabel('Log Likelihood Training')
        ax[0].set_ylabel('RSS Test (sum over crossval)')
        ax[1].set_ylabel('RSS Training (sum over crossval)')
        
        fig.tight_layout()
        fig.savefig(filename)
    
    

    def PlotMeasureListValues(self, filename = 'measurelist_values.pdf'):
        def significanceColor(beta):
            if beta   >  0.00: return 'red'
            elif beta == 0.00: return 'lightgray'
            else:              return 'black'

        # amelies colorscheme...                 
        colornames  = ['gray','#f563e2','#609cff','#00bec4','#00b938','#b79f00','#f8766c', '#75507b']

        # collect measure names for labels
        measurelist = self.measure_data.MeasureList(mincount = self.__MeasureMinCount, extend_measure_names = True, measure_level = 2)
        countrylist = [country[13:].strip(']') for country in self.finalModels[0].data.xnames if country[:10] == 'C(Country)']
        modelcount  = len(self.finalModels)
        intercept   = [self.finalResults[m].params['Intercept'] for m in range(modelcount)]

        # convert shortened measure names backward
        measure_level_dict = {}
        for mn in measurelist.keys():
            l1,l2 = mn.split(' -- ')
            if not l1 in measure_level_dict.keys():
                measure_level_dict[l1] = {}
            measure_level_dict[l1][l2] = self.measure_data.CleanUpMeasureName(l2)
        L1names = list(measure_level_dict.keys())
        L1names.sort()

        # internal counters to determine position to plot
        ypos = 0
        groupcolor = 0

        # define positions for various elements
        label_x        = 1
        label_x_header = .6
        value_x        = 12
        value_dx       = 2
        boxalpha       = .15
        
        # start plot
        fig,ax = plt.subplots(figsize = (14,23))
        
        # country effects
        ax.annotate('Country specific effects',[label_x_header, len(countrylist)], c = colornames[groupcolor], weight = 'bold' )
        background = plt.Rectangle([label_x - .6, ypos - .65], value_x + (modelcount-1)*2 + .6, len(countrylist) + 1.8, fill = True, fc = colornames[groupcolor], alpha = boxalpha, zorder = 10)
        ax.add_patch(background)
        for country in countrylist[::-1]:
            ax.annotate(country, [label_x, ypos], c= colornames[groupcolor])
            for m in range(modelcount):
                beta_val = self.finalResults[m].params['C(Country)[T.{}]'.format(country)] / intercept[m]
                ax.annotate('{:6.0f}%'.format(beta_val*100),[value_x + m * value_dx, ypos], c = significanceColor(beta_val), ha = 'right')
            ypos += 1
        groupcolor += 1
        ypos+=2 

        # measure effects
        for l1 in L1names[::-1]:
            ax.annotate(l1,[label_x_header, ypos + len(measure_level_dict[l1])], c = colornames[groupcolor], weight = 'bold')
            L2names = list(measure_level_dict[l1].keys())
            L2names.sort()
            
            background = plt.Rectangle([label_x - .6, ypos - .65], value_x + 2*(modelcount-1) + .6, len(measure_level_dict[l1]) + 1.8, fill = True, fc = colornames[groupcolor], alpha = boxalpha, zorder = 10)
            ax.add_patch(background)
            
            for l2 in L2names[::-1]:
                ax.annotate(l2,[label_x,ypos],c = colornames[groupcolor])
                for m in range(modelcount):
                    beta_val = self.finalResults[m].params[measure_level_dict[l1][l2]] / intercept[m]
                    ax.annotate('{:6.0f}%'.format(beta_val*100),[value_x + m * value_dx, ypos], c = significanceColor(beta_val), ha = 'right')
                ypos+=1
            ypos+=2
            groupcolor += 1

        # header
        ax.annotate(r'shiftdays $s$',[label_x,ypos+1])
        ax.annotate(r'Penality parameter $\log_{10}(\alpha)$',[label_x,ypos])
        for m in range(modelcount):
            ax.annotate('{}'.format(self.finalParameters[m][0]),               [value_x + m*value_dx,ypos+1],ha='right')
            ax.annotate('{:.1f}'.format(np.log10(self.finalParameters[m][1])), [value_x + m*value_dx,ypos],ha='right')        

        # adjust and save
        ax.set_xlim([0,value_x + 2 * modelcount])
        ax.set_ylim([-1,ypos+2])
        ax.axis('off')
        fig.savefig(filename)
        


    def GetMeasureEffects(self, drop_zeros = False, rescale = True):
        if not self.finalCV is None:
            finalCVrelative                = self.finalCV[self.finalCV.columns[10:]]
            if rescale: finalCVrelative    = finalCVrelative.divide(self.finalCV['Intercept'],axis = 0)
            finalCVrelative                = finalCVrelative.quantile([.5,.025,.975]).T
            finalCVrelative.columns        = ['median', 'low', 'high']            
            if drop_zeros: finalCVrelative = finalCVrelative[(finalCVrelative['median'] != 0) | (finalCVrelative['low'] != 0) | (finalCVrelative['high'] != 0)]
            fCV_withNames                  = self.measure_data.GetMeasureNames().merge(finalCVrelative, how = 'inner', left_index = True, right_index = True)
            fCV_withNames.sort_values(by   = ['median','high'], inplace = True)
            
            return fCV_withNames
        else:
            return None


        
    def PlotMeasureListSorted(self, filename = 'measurelist_sorted.pdf', drop_zeros = False, figsize = (15,30)):
        l1_pos      = -120
        l2_pos      = -80
        blacklines  = [-30,-20,-10,0,10]
        graylines   = []
        border      = 2
        def PlotRow(ax, ypos = 1, values = None, color = '#ffffff', boxalpha = .2, columns = (None,None), textlen = 40):
            ax.plot(values['median'],[ypos], c = measurecolors[values[columns[0]]], marker = 'D')
            ax.plot([values['low'],values['high']],[ypos,ypos], c = measurecolors[values[columns[0]]], lw = 2)
            background = plt.Rectangle([1e-2 * (l1_pos - border), ypos - .4], 1e-2*(-l1_pos + np.max(blacklines + graylines)+border), .9, fill = True, fc = color, alpha = boxalpha, zorder = 10)
            ax.add_patch(background)
            ax.annotate(textwrap.shorten(values[columns[0]], width = textlen), [1e-2*l1_pos,ypos-.1])
            if columns[0] != columns[1]: ax.annotate(textwrap.shorten(values[columns[1]], width = textlen), [1e-2*l2_pos,ypos-.1])

        measure_effects = self.GetMeasureEffects(drop_zeros = drop_zeros)
        if len(measure_effects.columns) >= 5:
            headerkey, labelkey = measure_effects.columns[[0,-4]]
        else:
            headerkey = labelkey = measure_effects.columns[0]
            
        colornames      = ['#f563e2','#609cff','#00bec4','#00b938','#b79f00','#f8766c', '#75507b']
        colornames      = [cn for cn in matplotlib.colors.TABLEAU_COLORS.keys() if (cn.upper() != 'TAB:WHITE' and cn.upper() != 'TAB:GRAY')]
        measurecolors   = {l1:colornames[i] for i,l1 in enumerate(measure_effects[headerkey].unique())}

        
        fig,ax = plt.subplots(figsize = figsize)

        for j,(index,values) in enumerate(measure_effects.iterrows()):
            PlotRow(ax, ypos = -j,values = values, color = measurecolors[values['Measure_L1']], columns = (headerkey, labelkey))
        for x in blacklines:
            ax.plot([1e-2 * x,1e-2 * x],[0.7,-j-0.5], lw = 2, c = 'black',zorder = -2)
            ax.annotate('{:.0f}%'.format(x),[1e-2*x,0.9],fontsize = 12, c = 'gray', ha = 'center')
        for x in graylines:
            ax.plot([1e-2 * x,1e-2 * x],[0.6,-j-0.4], lw = 1, c = 'gray',zorder = -2)
            ax.annotate('{:.0f}%'.format(x),[1e-2*x,0.9],fontsize = 12, c = 'gray', ha = 'center')
        
        ax.set_xlim([1e-2 * (l1_pos - border), 1e-2 * (blacklines[-1]+0.5*border)])
        ax.set_ylim([-j-2,1.6])
        ax.axis('off')
        fig.savefig(filename)



    def addDF(self, df = None, new = None):
        if not new is None:
            if df is None:
                return new
            else:
                return pd.concat([df,new], ignore_index = True, sort = False)
        else:
            return df
    

        
    def __getattr__(self,key):
        if key == 'CVresults':
            return self.__CVresults



    # make CrossValidation object pickleable ...
    # (getstate, setstate) interacts with (pickle.dump, pickle.load)
    def __getstate__(self):
        return {'kwargs':          self.__kwargs_for_pickle,
                'CVresults':       self.__CVresults,
                'finalModels':     self.finalModels,
                'finalResults':    self.finalResults,
                'finalCV':         self.finalCV,
                'finalParameters': self.finalParameters}
    
    def __setstate__(self,state):
        self.__init__(**state['kwargs'])
        self.__CVresults      = state['CVresults']
        self.finalModels      = state['finalModels']
        self.finalResults     = state['finalResults']
        self.finalCV          = state['finalCV']
        self.finalParameters  = state['finalParameters']
                
