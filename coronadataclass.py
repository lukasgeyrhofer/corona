import numpy as np
import pandas as pd
import os
import datetime


class CoronaData(object):
    '''
    ***********************************************************
    **  wrapper for reading data from official repositories  **
    **  github.com/lukasgeyrhofer/corona/                    **
    ***********************************************************

    main functionality:
    
    * can iterate over data-object, eg:
            data = CoronaData()
            for countryname, countrydata in data:
                // do stuff
                // countrydata is pandas-object with time series
    
    * access pandas-object directly, eg:
            data = Coronadata()
            then 'data.Austria' or 'data.Germany' returns the pandas objects for the respective countries
            
    * pandas-object has following columns:
        Dates
        Confirmed
        Deaths
        Recovered
    
    * added automatic update of data with official Johns Hopkins University repository (github.com/CSSEGISandData/COVID-19/)
      data is stored locally, such that not every instance of CoronaData class accesses the github data
    
    '''
    
    
    def __init__(self,**kwargs):

        self.BASE_URL     = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/'
        self.CONFIRMED    = 'time_series_covid19_confirmed_global.csv'
        self.DEATH        = 'time_series_covid19_deaths_global.csv'
        self.RECOVERED    = 'time_series_covid19_recovered_global.csv'

        self.CONFIRMED_US = 'time_series_covid19_confirmed_US.csv'
        self.DEATHS_US    = 'time_series_covid19_deaths_US.csv'


        self.__data                = {}
        self.__maxtrajectorylength = 0
        
        self.__smooth_windowsize = kwargs.get('SmoothWindowSize', None)
        self.__smooth_stddev     = kwargs.get('SmoothStdDev', None)
        self.__resolve_US_states = kwargs.get('resolve_US_states', False)
        self.__keep_US           = kwargs.get('keep_US', False)
        
        self.__output_dateformat = kwargs.get('DateFormat', '%d/%m/%Y')
        self.__input_dateformat  = '%m/%d/%y'
        
        self.__date_as_index     = kwargs.get('DateAsIndex', False)
        
        if kwargs.get('download_data',False):
            self.DownloadData()
            
        self.LoadData()



    def LoadData(self):
        
        if not os.path.exists(self.CONFIRMED) or not os.path.exists(self.DEATH) or not os.path.exists(self.RECOVERED) or (self.__resolve_US_states and not os.path.exists(self.CONFIRMED_US)) or (self.__resolve_US_states and not os.path.exists(self.DEATHS_US)):
            self.DownloadData()
        
        self.__data_confirmed = pd.read_csv(self.CONFIRMED)
        self.__data_death     = pd.read_csv(self.DEATH)
        self.__data_recovered = pd.read_csv(self.RECOVERED)
        
        self.__countrylist = list(self.__data_confirmed['Country/Region'].unique())
        self.__countrylist.sort()
        
        for country in self.__countrylist:
            tmp_dates     = self.ConvertDates(self.__data_confirmed.columns[5:], inputformat = self.__input_dateformat)
            tmp_confirmed = np.array(((self.__data_confirmed[self.__data_confirmed['Country/Region'] == country].groupby('Country/Region').sum()).T)[3:], dtype = np.int).flatten()
            tmp_deaths    = np.array(((self.__data_death    [self.__data_death    ['Country/Region'] == country].groupby('Country/Region').sum()).T)[3:], dtype = np.int).flatten()
            tmp_recovered = np.array(((self.__data_recovered[self.__data_recovered['Country/Region'] == country].groupby('Country/Region').sum()).T)[3:], dtype = np.int).flatten()
            
            self.AddCountryData(country, tmp_dates, tmp_confirmed, tmp_deaths, tmp_recovered)

        if self.__resolve_US_states:
            
            self.__data_confirmed_us = pd.read_csv(self.CONFIRMED_US, index_col = 0)
            self.__data_deaths_us    = pd.read_csv(self.DEATHS_US, index_col = 0)
            
            statelist = list(self.__data_confirmed_us['Province_State'].unique())
            
            for state in statelist:
                tmp_dates_us     = self.ConvertDates(self.__data_confirmed_us.columns[11:], inputformat = self.__input_dateformat)
                tmp_confirmed_us = np.array(((self.__data_confirmed_us[self.__data_confirmed_us['Province_State'] == state].groupby('Province_State').sum()).T)[5:],dtype = np.int).flatten()
                tmp_deaths_us    = np.array(((self.__data_confirmed_us[self.__data_confirmed_us['Province_State'] == state].groupby('Province_State').sum()).T)[5:],dtype = np.int).flatten()
                tmp_recovered_us = np.zeros_like(tmp_confirmed_us)
                
                self.AddCountryData('US - {}'.format(state), tmp_dates, tmp_confirmed_us, tmp_deaths_us, tmp_recovered_us)
                self.countrylist.append('US - {}'.format(state))
            
            if not self.__keep_US:
                del self.__data['US']
                self.__countrylist.remove('US')
            self.__countrylist.sort()
            
            

    def AddCountryData(self,countryname, dates, confirmed, deaths, recovered):
        self.__data[countryname] = pd.DataFrame({ 'Date': dates, 'Confirmed': confirmed, 'Deaths': deaths, 'Recovered': recovered})
        if len(dates) > self.__maxtrajectorylength:
            self.__maxtrajectorylength = len(dates)



    def DownloadData(self):
        # download data and store locally
        data_confirmed = pd.read_csv(self.BASE_URL + self.CONFIRMED)
        data_deaths    = pd.read_csv(self.BASE_URL + self.DEATH)
        data_recovered = pd.read_csv(self.BASE_URL + self.RECOVERED)
        
        data_confirmed.to_csv(self.CONFIRMED)
        data_deaths.to_csv(self.DEATH)
        data_recovered.to_csv(self.RECOVERED)

        if self.__resolve_US_states:
            data_confirmed_us = pd.read_csv(self.BASE_URL + self.CONFIRMED_US)
            data_deaths_us    = pd.read_csv(self.BASE_URL + self.DEATHS_US)
            
            data_confirmed_us.to_csv(self.CONFIRMED_US)
            data_deaths_us.to_csv(self.DEATHS_US)
        
        
    

    
    def ConvertDates(self, datelist, outputformat = None, inputformat = None):
        if outputformat is None: outputformat = self.__output_dateformat
        if inputformat  is None: inputformat  = self.__output_dateformat
        return np.array([datetime.datetime.strptime(date, inputformat).strftime(outputformat) for date in datelist])



    def DateAtCases(self, country, cases = 1, column = 'Confirmed', return_index = False, outputformat = None):
        if outputformat is None: outputformat = self.__output_dateformat
        cd = self.CountryData(country)
        index = int(np.argmin(cd[column].values <= cases))
        casetime = datetime.datetime.strptime(self.DateStart(country),self.__output_dateformat) + datetime.timedelta(days = index)
        if return_index:
            return datetime.datetime.strftime(casetime,outputformat),index
        else:
            return datetime.datetime.strftime(casetime,outputformat)


    
    def DateStart(self, country, outputformat = None):
        if country in self.__countrylist:
            return self.__data[country]['Date'].values[0]
        else:
            return None



    def DateFinal(self, country, outputformat = None):
        if country in self.countrylist:
            return self.__data[country]['Date'].values[-1]
        else:
            return None



    def CountryData(self, country, windowsize = None, stddev = None, dateasindex = None):
        if windowsize  is None: windowsize  = self.__smooth_windowsize
        if stddev      is None: stddev      = self.__smooth_stddev
        if dateasindex is None: dateasindex = self.__date_as_index
        if country in self.__countrylist:
            if (not windowsize is None) and (not stddev is None):
                if stddev > 0:
                    returnDF = self.__data[country].rolling(window = windowsize, win_type = 'gaussian', min_periods = 1, center = True).mean(std = stddev)
                    returnDF['Date'] = self.__data[country]['Date'].values
                    if dateasindex:
                        return returnDF.set_index('Date', drop = True)
                    else:
                        return returnDF
            if dateasindex:
                return self.__data[country].set_index('Date', drop = True)
            else:
                return self.__data[country]
        else:
            return None



    def CountryGrowthRates(self, country = None, windowsize = None, stddev = None, dateasindex = None):
        def GrowthRate(trajectory):
            storewarnings = np.seterr(invalid = 'ignore')
            growthrate = np.diff(np.log(trajectory))
            np.seterr(**storewarnings)
            growthrate = np.nan_to_num(growthrate)
            growthrate[growthrate > 1e300] = 0            
            return growthrate

        if windowsize  is None: windowsize  = self.__smooth_windowsize
        if stddev      is None: stddev      = self.__smooth_stddev
        if dateasindex is None: dateasindex = self.__date_as_index

        if country in self.__countrylist:
            returnDF = self.__data[country].apply({k:GrowthRate for k in self.__data[country].columns if k != 'Date'}) #.join(self.__data[country].Date)
            
            if (not windowsize is None) and (not stddev is None):
                if stddev > 0:
                    returnDF = returnDF.rolling(window = windowsize, win_type = 'gaussian', min_periods = 1, center = True).mean(std = stddev).join(self.__data[country].Date)
                    returnDF['Date'] = self.__data[country]['Date'].values[1:]
            if dateasindex:
                return returnDF.set_index('Date', drop = True)
            else:
                return returnDF
        else:
            return None



    def __getattr__(self,key):
        if key.lower() == 'countrylist':
            return self.__countrylist
        elif key.replace('_',' ') in self.__countrylist:
            return self.CountryData(country = key.replace('_',' '))
        else:
            raise KeyError

        
    
    def __len__(self):
        return self.__maxtrajectorylength

    
    
    def __iter__(self):
        for country in self.__countrylist:
            yield country, self.__data[country]




if __name__ == "__main__":
    # download data when file is called from cli
    data = CoronaData(download_data = True)
