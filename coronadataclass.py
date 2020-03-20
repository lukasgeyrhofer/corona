import numpy as np
import pandas as pd

class CoronaData(object):
    '''
    ***************************************************
    wrapper for reading data from official repositories
    ***************************************************
    
    main functionality:
    
    * can iterate over data-object, eg:
            data = Coronadata()
            for countryname, countrydata in data:
                // do stuff
                // countrydata is pandas-object with time series
    
    * access pandas-object directly, eg:
            data = Coronadata()
            then 'data.Austria' or 'data.Germany' returns the pandas objects for the respective countries
            
            
    * pandas-object has following columns:
        Dates
        Confirmed
        Recovered
        Deaths
    
    * for countries with multiple provinces (ie US with states), can lump everything together
      with option 'group_by_country = True' (True by default)
      if false, generate entries with 'COUNTRY_PROVINCE' with its own pandas-object
    
    '''
    
    
    def __init__(self,**kwargs):
        
        self.__datafile            = kwargs.get('datafile','../data/datasets_covid-19/time-series-19-covid-combined.csv')
        self.__group_by_country    = kwargs.get('group_by_country',True)
        self.__data                = {}
        self.__maxtrajectorylength = 0
        
        self.LoadData()

    def LoadData(self, filename = None, group_by_country = None):
        if filename is None:
            filename = self.__datafile
        if group_by_country is None:
            group_by_country = self.__group_by_country
            
        self.__tempalldata = pd.read_csv(filename)
        self.__countrylist = list(set(self.__tempalldata['Country/Region']))
        self.__countrylist.sort()
        
        for country in self.__countrylist:
            tmp_data      = self.__tempalldata[self.__tempalldata['Country/Region'] == country]
            
            province_list = list(set(tmp_data['Province/State']))
            
            if len(province_list) > 1:
                if group_by_country:
                    tmp_data      = tmp_data.groupby('Date').sum()
                    
                    tmp_dates     = np.array([d for d in tmp_data.groupby('Date').groups.keys()])
                    tmp_total     = np.array(tmp_data['Confirmed'], dtype = np.int)
                    tmp_recovered = np.array(tmp_data['Recovered'], dtype = np.int)
                    tmp_deaths    = np.array(tmp_data['Deaths'],    dtype = np.int)

                    self.AddCountryData(country,tmp_dates, tmp_total, tmp_recovered, tmp_deaths)

                else:
                    for province in province_list:
                        tmp_data_prov      = tmp_data[tmp_data['Province/State'] == province]
                        tmp_dates_prov     = np.array(tmp_data_prov['Date'])
                        tmp_total_prov     = np.array(tmp_data_prov['Confirmed'], dtype = np.int)
                        tmp_recovered_prov = np.array(tmp_data_prov['Recovered'], dtype = np.int)
                        tmp_deaths_prov    = np.array(tmp_data_prov['Deaths'],    dtype = np.int)
                        
                        self.AddCountryData(country + '_' + province, tmp_dates_prov, tmp_total_prov, tmp_recovered_prov, tmp_deaths_prov)
            else:
                tmp_dates     = np.array(tmp_data['Date'])
                tmp_total     = np.array(tmp_data['Confirmed'], dtype = np.int)
                tmp_recovered = np.array(tmp_data['Recovered'], dtype = np.int)
                tmp_deaths    = np.array(tmp_data['Deaths'],    dtype = np.int)
                
                self.AddCountryData(country,tmp_dates, tmp_total, tmp_recovered, tmp_deaths)


    def AddCountryData(self,countryname, dates, confirmed, recovered, deaths):
        self.__data[countryname] = pd.DataFrame({ 'Date': dates, 'Confirmed': confirmed, 'Recovered': recovered, 'Deaths': deaths})
        if len(dates) > self.__maxtrajectorylength:
            self.__maxtrajectorylength = len(dates)


    def __getattr__(self,key):
        if key.lower() == 'countrylist':
            return self.__countrylist
        elif key in self.__countrylist:
            return self.__data[key]
        else:
            raise KeyError
    
    
    def __len__(self):
        return self.__maxtrajectorylength
    
    
    def __iter__(self):
        for country in self.__countrylist:
            yield country, self.__data[country]
