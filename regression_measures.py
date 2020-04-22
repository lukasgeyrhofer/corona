import numpy as np
import pandas as pd
import datetime
import re
import os,glob

# statistics
import statsmodels.api as sm
import statsmodels.formula.api as smf


def date2vector(implementdate, start = '22/1/20', end = None, shiftdays = 0):
    # generate vector of 0s and 1s when measure is implemented or not
    starttime     = datetime.datetime.strptime(start,         '%d/%m/%y')
    if end is None:
        endtime   = datetime.datetime.today()
    else:
        endtime   = datetime.datetime.strptime(end,           '%d/%m/%y')
    implementtime = datetime.datetime.strptime(implementdate, '%d/%m/%Y')
    
    totaldays   = (endtime       - starttime).days
    measuredays = (implementtime - starttime).days
    
    vec         = np.zeros(totaldays)
    vec[min(measuredays+shiftdays,len(vec)-1):] = 1
    
    return vec


def ConvertDateFormat(date):
    m,d,y = date.split('/')
    return '{:02d}/{:02d}/{:02d}'.format(int(d),int(m),int(y))


def CleanUpMeasureName(measurename):
    # regression model formula can't contain special characters
    return ''.join([mn.capitalize() for mn in measurename.replace(',','').replace('-','').replace('/','').split(' ')])


def GetMeasureIDs(measure_data = None, countrylist = None, measure_level = 2, mincount = None, extend_measure_names = False):
    if countrylist is None:
        countrylist = measure_data.countrylist # use ALL countries
    
    measurelist = {}
    
    # get all restrictions from countries
    for country in countrylist:
        country_measures = measure_data.CountryData(country, measure_level = 2, extend_measure_names = extend_measure_names)
        for measurename, initialdata in country_measures.items():
            if not measurename in measurelist.keys():
                measurelist[measurename] = 0
            measurelist[measurename] += 1
    
    if not mincount is None:
        # rebuild dict with restrictions
        measurelist = {k:v for k,v in measurelist.items() if v >= mincount}

    return measurelist


def SmoothTrajectories3(traj):
    if len(traj) > 3:
        newtraj       = np.zeros(len(traj))
        newtraj[0]    = (             2 * traj[0]    + traj[1] )/3.
        newtraj[1:-1] = (traj[0:-2] + 2 * traj[1:-1] + traj[2:])/4.
        newtraj[-1]   = (traj[-2]   + 2 * traj[-1]             )/3.
        return newtraj
    else:
        return traj

    
def SmoothTrajectoriesFreqCutoff(traj, freqcutoff = .1, b = .1):
    N = int(np.ceil((4 / b)))
    if not N % 2: N += 1
    n = np.arange(N)

    sinc_func = np.sinc(2 * freqcutoff * (n - (N - 1) / 2.))
    window = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 0.08 * np.cos(4 * np.pi * n / (N - 1))
    sinc_func = sinc_func * window
    sinc_func = sinc_func / np.sum(sinc_func)

    new_traj = np.convolve(traj, sinc_func)
    return new_traj[(N-1)//2:(N-1)//2 + len(traj)]
    


def GetCountryObservables(jhu_data = None, countrylist = None, data = 'Confirmed', startcases = None, maxlen = None, smooth_stddev = None):
    if countrylist is None:
        countrylist = jhu_data.countrylist
        
    trajectories = {}
    for country in [c for c in countrylist if c in jhu_data.countrylist]:
        if not smooth_stddev is None:
            ctraj     = jhu_data.CountryData(country, windowsize = 21, window_stddev = smooth_stddev)[data].values
        else:
            ctraj     = jhu_data.CountryData(country)[data].values
            
        dlctraj       = np.diff(np.log(ctraj))
        dlctraj       = np.nan_to_num(dlctraj)
        dlctraj[dlctraj > 1e200] = 0
        
        starttraj     = 0
        if not startcases is None:
            starttraj = np.argmax(ctraj >= startcases)
            dlctraj   = dlctraj[starttraj:]
        
        endtraj       = len(dlctraj)
        if not maxlen is None:
            endtraj   = np.min([starttraj + maxlen + 1, len(dlctraj)])

        trajectories[country] = {}
        trajectories[country]['startdate']  = ConvertDateFormat(jhu_data.CountryData(country)['Date'][starttraj])
        trajectories[country]['Observable'] = dlctraj[starttraj:endtraj]
    
    return trajectories
    
    
def GetRegressionDF(jhu_data = None, measure_data = None, countrylist = None, measure_level = 2, shiftdays = 0, verbose = False, maxlen = None, smooth_stddev = None):
    # construct pd.DataFrame used for regression
    
    # get trajectories and measure list for all countries in 'countrylist'
    trajectories         = GetCountryObservables(jhu_data = jhu_data, countrylist = countrylist, data = 'Confirmed', startcases = 30, maxlen = maxlen, smooth_stddev = smooth_stddev)
    measureIDs           = measure_data.MeasureList(countrylist = countrylist, measure_level = 2, mincount = 5)
    cleaned_measurelist  = {CleanUpMeasureName(mn):count for mn,count in measureIDs.items()}
    regressionDF         = None
    
    if verbose:
        print(measureIDs)
    
    for country in trajectories.keys():
        if country in measure_data.countrylist:

            # ********************************************
            # change observable to regress here:
            observable                  = trajectories[country]['Observable']
            obslen                      = len(observable)
            # ********************************************
            
            DF_country = measure_data.ImplementationTable(country           = country,
                                                          measure_level     = 2,
                                                          startdate         = trajectories[country]['startdate'],
                                                          shiftdays         = shiftdays,
                                                          maxlen            = obslen,
                                                          clean_measurename = True)
            
            for measurename in DF_country.columns:
                if measurename not in measureIDs.keys():
                    DF_country.drop(labels = measurename, axis = 'columns')
            
            DF_country['Country']    = country
            DF_country['Observable'] = observable

            
            if not (np.isnan(DF_country['Observable']).any() or np.isinf(DF_country['Observable']).any()):

                if regressionDF is None:
                    regressionDF = DF_country
                else:
                    regressionDF = pd.concat([regressionDF,DF_country], ignore_index = True, sort = False)
    
    # not implemented measures should be NaN values, set them to 0
    regressionDF.fillna(0, inplace = True)
    
    return regressionDF



def GetCountryMasks(regrDF):
    countrylist = list(regrDF['Country'].unique())
    maskdict = {}
    for country in countrylist:
        mask = list(regrDF['Country'] == country)
        maskdict[country] = mask
    return maskdict



def CrossValidation(data = None, drop_cols = None, outputheader = None, alpha = 1e-5, alphacountry = None, crossvalcount = 10, countrycrossvalidation = False):
    # output df
    result_DF = None
    
    if not countrycrossvalidation:
        # assign samples to each of the crossvalidation chunks
        datalen = len(data)
        chunklen   = np.ones(crossvalcount,dtype = np.int) * (datalen // crossvalcount)
        chunklen[:datalen%crossvalcount] += 1
        samples    = np.random.permutation(
                        np.concatenate(
                            [i*np.ones(chunklen[i],dtype = np.int) for i in range(crossvalcount)]
                        ))
    else:
        countrylist = list(data['Country'].unique())
        crossvalcount = len(countrylist)
        samples = np.concatenate([i * np.ones(len(data[data['Country'] == countrylist[i]])) for i in range(crossvalcount)])

    # generate formula of model directly from columns in DataFrame
    measurelist = list(data.columns)
    measurelist.remove('Observable')
    measurelist.remove('Country')
    formula = 'Observable ~ C(Country) + ' + ' + '.join(measurelist)
    
    # iterate over all chunks
    for xv_index in range(crossvalcount):
        # generate training and test models
        trainidx = (samples != xv_index)
        testidx  = (samples == xv_index)
        trainmodel = smf.ols(formula = formula, data = data[trainidx], drop_cols = drop_cols)
        testmodel  = smf.ols(formula = formula, data = data[testidx],  drop_cols = drop_cols)
    
        # if no alphacountry value is given, assume same penalty (alpha) for all paramters
        if alphacountry is None:
            results = trainmodel.fit_regularized(alpha = alpha, L1_wt = 1)
        else:
            # otherwise, penalize measures and countries differently
            # no penality for the 'Intercept'
            alphavec = np.zeros(len(trainmodel.exog_names))
            for i,exogname in enumerate(trainmodel.exog_names):
                if exogname[:10] == 'C(Country)': alphavec[i] = alphacountry
                elif exogname == 'Intercept':     alphavec[i] = 0
                else:                             alphavec[i] = alpha
            results = trainmodel.fit_regularized(alpha = alphavec, L1_wt = 1)

        # generate list of test params
        # random sampling could have discarded some of these parameters in the test case
        test_params = []
        for paramname in testmodel.exog_names:
            if paramname in results.params.keys():
                test_params.append(results.params[paramname])
            else:
                test_params.append(0)

        obs_train   = np.array(trainmodel.endog)
        obs_test    = np.array(testmodel.endog)
        pred_train  = trainmodel.predict(results.params)
        pred_test   = testmodel.predict(test_params)
            
        # store results first in dict
        result_dict = {}
        if outputheader is not None:
            result_dict.update(outputheader)
        
        if countrycrossvalidation:
            result_dict['Iteration']    = countrylist[xv_index]
        else:
            result_dict['Iteration']    = xv_index
        
        result_dict['Loglike Training'] = trainmodel.loglike(results.params)
        result_dict['Loglike Test']     = testmodel.loglike(np.array(test_params))

        result_dict['R2 Training']      = 1 - np.sum((obs_train - pred_train)**2)/np.sum((obs_train - np.mean(obs_train))**2)
        result_dict['R2 Test']          = 1 - np.sum((obs_test - pred_test)**2)/np.sum((obs_test - np.mean(obs_test))**2)

        result_dict['RSS Training']     = np.sum((obs_train - pred_train)**2)
        result_dict['RSS Test']         = np.sum((obs_test - pred_test)**2)

        result_dict.update({k:v for k,v in results.params.items()})
        
        # append dict to df
        if result_DF is None:
            result_DF = pd.DataFrame({k:np.array([v]) for k,v in result_dict.items()})
        else:
            result_DF = result_DF.append(result_dict, ignore_index = True)
    return result_DF








