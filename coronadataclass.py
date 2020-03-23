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
        Recovered
        Deaths
    
    * for countries with multiple provinces (ie US with states), can lump everything together
      with option 'group_by_country = True' (True by default)
      if false, generate entries with 'COUNTRY_PROVINCE' with its own pandas-object
    
    * added automatic update of data with official Johns Hopkins University repository (github.com/CSSEGISandData/COVID-19/)
      with a script found on github.com/datasets/covid-19/
      data is stored locally, such that not every instance of CoronaData class accesses the github data
    
    '''
    
    
    def __init__(self,**kwargs):

        self.BASE_URL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/'
        self.CONFIRMED = 'time_series_19-covid-Confirmed.csv'
        self.DEATH = 'time_series_19-covid-Deaths.csv'
        self.RECOVERED = 'time_series_19-covid-Recovered.csv'


        
        self.__datafile            = kwargs.get('datafile','time-series-19-covid-combined.csv')
        self.__group_by_country    = kwargs.get('group_by_country',True)
        self.__data                = {}
        self.__maxtrajectorylength = 0
        
        if kwargs.get('download_data',False):
            self.DownloadData()
            
        self.LoadData()




    def LoadData(self, filename = None, group_by_country = None):
        if filename is None:
            filename = self.__datafile
            
        if group_by_country is None:
            group_by_country = self.__group_by_country
        
        if not os.path.exists(filename):
            self.DownloadData(datafile = filename)
        
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





    def DownloadData(self, datafile = 'time-series-19-covid-combined.csv'):
        """
        copied from github.com/datasets/covid-19/process.py
        """
        def to_normal_date(row):
            old_date = row['Date']
            month, day, year = row['Date'].split('-')
            day = f'0{day}' if len(day) == 1 else day
            month = f'0{month}' if len(month) == 1 else month
            row['Date'] = '-'.join([day, month, year])

        unpivoting_fields = [
            { 'name': '([0-9]+\/[0-9]+\/[0-9]+)', 'keys': {'Date': r'\1'} }
        ]

        extra_keys = [{'name': 'Date', 'type': 'string'} ]
        extra_value = {'name': 'Case', 'type': 'number'}

        Flow(
            load(f'{self.BASE_URL}{self.CONFIRMED}'),
            load(f'{self.BASE_URL}{self.RECOVERED}'),
            load(f'{self.BASE_URL}{self.DEATH}'),
            unpivot(unpivoting_fields, extra_keys, extra_value),
            find_replace([{'name': 'Date', 'patterns': [{'find': '/', 'replace': '-'}]}]),
            to_normal_date,
            set_type('Date', type='date', format='%d-%m-%y', resources=None),
            set_type('Case', type='number', resources=None),
            join(
                source_name='time_series_19-covid-Confirmed',
                source_key=['Province/State', 'Country/Region', 'Date'],
                source_delete=True,
                target_name='time_series_19-covid-Deaths',
                target_key=['Province/State', 'Country/Region', 'Date'],
                fields=dict(Confirmed={
                    'name': 'Case',
                    'aggregate': 'first'
                })
            ),
            join(
                source_name='time_series_19-covid-Recovered',
                source_key=['Province/State', 'Country/Region', 'Date'],
                source_delete=True,
                target_name='time_series_19-covid-Deaths',
                target_key=['Province/State', 'Country/Region', 'Date'],
                fields=dict(Recovered={
                    'name': 'Case',
                    'aggregate': 'first'
                })
            ),
            add_computed_field(
                target={'name': 'Deaths', 'type': 'number'},
                operation='format',
                with_='{Case}'
            ),
            delete_fields(['Case']),
            update_resource('time_series_19-covid-Deaths', name='time-series-19-covid-combined', path = datafile),
            dump_to_path()
        ).results()[0]









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
