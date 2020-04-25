#basics
import numpy as np
import pandas as pd

import os
import itertools
import datetime
import time

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
    
    
    
    def SingleParamCV(self, shiftdays = None, alpha = None, countrywise_crossvalidation = True, crossvalcount = 10, outputheader = {}):
        
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
            
            curResDF = self.addDF(curResDF,pd.DataFrame({k:np.array([v]) for k,v in result_dict.items()}))
        
        return curResDF
    
    
        
    def RunCV(self, shiftdaylist = [0], alphalist = [1e-5], verbose = None, countrywise_crossvalidation = True, crossvalcount = 10, outputheader = {}):
        if verbose is None: verbose = self.__verbose
        
        for shiftdays, alpha in itertools.product(shiftdaylist,alphalist):
            if verbose: print('{:3d} {:.6f} {:>15s}'.format(shiftdays,alpha, 'computing'), end = '\r', flush = True)
            
            curResDF         = self.SingleParamCV(shiftdays = shiftdays, alpha = alpha, countrywise_crossvalidation = countrywise_crossvalidation, crossvalcount = crossvalcount, outputheader = outputheader)
            self.__CVresults = self.addDF(self.__CVresults,curResDF)
            
            if verbose: print('{:3d} {:.6f} {:>15s}'.format(shiftdays,alpha, datetime.datetime.now().strftime('%H:%M:%S')))
            


    def SaveCVResults(self, filename = None, reset = False):
        try:        self.__CVresults.to_csv(filename)
        except:     pass
        if reset:   self.__CVresults = None



    def LoadCVResults(self, filename, reset = True):
        if reset:   self.__CVresults = pd.read(filename)
        else:       self.__CVresults = self.addDF(pd.read(filename))



    def ComputeFinalModels(self, modelparameters = [(6,1e-3)]):
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
            self.finalResults.append(self.finalModels[i].fit_regularized(alpha = alpha, L1_wt = 1))
            


    def PlotTrajectories(self, filename = 'trajectories.pdf'):
        modelcount  = len(self.finalModels)
        if modelcount > 0:
            countrylist = [country[13:].strip(']') for country in self.finalModels[0].data.xnames if country[:10] == 'C(Country)']
            ycount = len(countrylist) // 2 + len(countrylist) % 2
            
            fig,axes = plt.subplots(ycount, 2, figsize = (15,3*ycount))
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
            l1,l2 = mn.split(' - ')
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
        

        
    def PlotMeasureListSorted(self, filename = 'measurelist_sorted.pdf'):
        def modelname(index):
            return 'm{} ({},{})'.format(index,self.finalParameters[index][0],self.finalParameters[index][1])

        # collect measure names for labels
        measurelist = self.measure_data.MeasureList(extend_measure_names = True, measure_level = 2)
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

        colornames    = ['#f563e2','#609cff','#00bec4','#00b938','#b79f00','#f8766c', '#75507b']
        measurecolors = {l1:col for l1,col in zip(measure_level_dict.keys(),colornames)}
        measurecolors['Returning to normal life'] = '#eeff00'

        inverse_mld   = {}
        for l1 in measure_level_dict.keys():
            for l2 in measure_level_dict[l1].keys():
                inverse_mld[measure_level_dict[l1][l2]] = (l1,l2)

        modelDF       = pd.DataFrame(columns=[modelname(m) for m in range(modelcount)])

        allparamnames = list(set([mn for m in range(modelcount) for mn in self.finalResults[m].params.keys() if mn[:3] != 'C(C']))
        allparamnames.remove('Intercept')

        for paramname in allparamnames:
            paramvaluedict = {}
            for m in range(modelcount):
                paramvaluedict[modelname(m)] = float(self.finalResults[m].params[paramname]/intercept[m])
            modelDF.loc[paramname] = paramvaluedict
            
        modelDF['average'] = modelDF.mean(numeric_only = True, axis = 1)
        modelDF.sort_values(by = 'average', inplace = True, ascending = False)

        l1_pos = -3
        l2_pos = -2.5

        def plotbox(ax, ypos = 1, label = '', color = '#ffffff', boxalpha = .2, header = False):
            background = plt.Rectangle([l2_pos - .05, ypos - .4], -l2_pos + 1.05, .9, fill = True, fc = color, alpha = boxalpha, zorder = 10)
            ax.add_patch(background)
            if not header:
                ax.annotate(label, [l2_pos,ypos-.1])
            else:
                ax.annotate(label, [l2_pos,ypos-.1], c = color, weight = 'bold')

        finalCVrelative = self.finalCV[self.finalCV.columns[10:]].divide(self.finalCV['Intercept'],axis = 0).quantile([.5,.025,.975]).T

        
        averaged_beta = finalCVrelative.loc[list(modelDF.index),:]
        #print(finalCVrelative)
        averaged_beta.columns = ['mean','low','high']
        averaged_beta.sort_values(by = 'mean', axis = 0, inplace = True,ascending = False)

        betascaling = 10/3.
        fig,ax = plt.subplots(figsize = (15,30))
        for j,(index,values) in enumerate(averaged_beta.iterrows()):
            #print(index)
            if index in inverse_mld.keys():
                plotbox(ax,ypos = j,label = inverse_mld[index][1], color = measurecolors[inverse_mld[index][0]])
                ax.plot(averaged_beta['mean'][index]*betascaling,[j], c = measurecolors[inverse_mld[index][0]], marker = 'D')
                ax.plot([betascaling*averaged_beta['low'][index],betascaling*averaged_beta['high'][index]],[j,j], c = measurecolors[inverse_mld[index][0]], lw = 3)

        ax.set_xlim([-3.1,1.2])

        for o,vertline in enumerate(np.arange(10)*betascaling/10.):
            if vertline != 0 and -1 <= vertline  <= 1:
                ax.plot([vertline,vertline],[-.7,j+.7],c = 'lightgray',zorder = -3,lw =1)
                ax.plot([-vertline,-vertline],[-.7,j+.7],c = 'lightgray',zorder = -3,lw =1)
                ax.annotate('-{:d}%'.format(o*10),[-vertline,j+1.5],fontsize = 12,c='gray',ha='center')
                ax.annotate('{:d}%'.format(o*10),[vertline,j+1.5],fontsize = 12,c='gray',ha='center')
        ax.plot([-1,-1],[-1,j+1], lw = 2, c = 'black',zorder = -2)
        ax.plot([0,0],[-1,j+1], lw = 2, c = 'black', zorder = -2)
        ax.plot([1,1],[-1,j+1], lw = 2, c = 'black', zorder = -2)
        ax.annotate('0%',[0,j+1.5],fontsize = 12, c='gray',ha='center')

        for k,l1 in enumerate(sorted(measure_level_dict.keys())[::-1]):
            plotbox(ax,ypos = j + k + 3, label = l1, color = measurecolors[l1], header = True)

        ax.set_ylim([-2,j+k+4])
            
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
                
