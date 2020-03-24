import numpy as np
import pandas as pd
import os

from dataflows import Flow, load, unpivot, find_replace, set_type, dump_to_path, update_resource, join, add_computed_field, delete_fields

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
    
    * for countries with multiple provinces (ie US with states), can lump everything together
      with option 'group_by_country = True' (True by default)
      if false, generate entries with 'COUNTRY_PROVINCE' with its own pandas-object
    
    * added automatic update of data with official Johns Hopkins University repository (github.com/CSSEGISandData/COVID-19/)
      data is stored locally, such that not every instance of CoronaData class accesses the github data
    
    '''
    
    
    def __init__(self,**kwargs):

        self.BASE_URL  = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/'
        self.CONFIRMED = 'time_series_covid19_confirmed_global.csv'
        self.DEATH     = 'time_series_covid19_deaths_global.csv'


        self.__datafile            = kwargs.get('datafile','time-series-19-covid-combined.csv')
        self.__group_by_country    = kwargs.get('group_by_country',True)
        self.__data                = {}
        self.__maxtrajectorylength = 0
        
        if kwargs.get('download_data',False):
            self.DownloadData()
            
        self.LoadData()


    def LoadData(self, group_by_country = None):
        if group_by_country is None:
            group_by_country = self.__group_by_country
        
        if not os.path.exists(self.CONFIRMED) or not os.path.exists(self.DEATH):
            self.DownloadData()
        
        self.__data_confirmed = pd.read_csv(self.CONFIRMED)
        self.__data_death     = pd.read_csv(self.DEATH)
        
        self.__countrylist = list(self.__data_confirmed['Country/Region'].unique())
        self.__countrylist.sort()
        
        for country in self.__countrylist:
            tmp_dates     = np.array(  self.__data_confirmed.columns[5:])
            tmp_confirmed = np.array(((self.__data_confirmed[self.__data_confirmed['Country/Region'] == country].groupby('Country/Region').sum()).T)[3:]).flatten()
            tmp_deaths    = np.array(((self.__data_death    [self.__data_death    ['Country/Region'] == country].groupby('Country/Region').sum()).T)[3:]).flatten()

            self.AddCountryData(country, tmp_dates, tmp_confirmed, tmp_deaths)


    def AddCountryData(self,countryname, dates, confirmed, deaths):
        self.__data[countryname] = pd.DataFrame({ 'Date': dates, 'Confirmed': confirmed, 'Deaths': deaths})
        if len(dates) > self.__maxtrajectorylength:
            self.__maxtrajectorylength = len(dates)


    def DownloadData(self):
        # download data and store locally
        data_confirmed = pd.read_csv(self.BASE_URL + self.CONFIRMED)
        data_deaths    = pd.read_csv(self.BASE_URL + self.DEATH)
        
        data_confirmed.to_csv(self.CONFIRMED)
        data_deaths.to_csv(self.DEATH)
        


    def __getattr__(self,key):
        if key.lower() == 'countrylist':
            return self.__countrylist
        elif key.replace('_',' ') in self.__countrylist:
            return self.__data[key]
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
