import numpy as np
import pandas as pd
import os

class COVID19_measures(object):
    '''
    ***************************************************
    **  wrapper for COVID19 measures tracked by CSH  **
    **  github.com/lukasgeyrhofer/corona/            **
    ***************************************************

    
    data can be found at
    https://github.com/amel-github/covid19-interventionmeasures
    compiled by Desvars-Larrive et al (2020)
    
    read table of measures from CSV file,
    and download data from github if not present or forced
    
    main usage:
    ( with default options )
    ***************************************************
    
        data = COVID19_measures( download_data        = False,
                                 measure_level        = 2,
                                 only_first_dates     = False,
                                 unique_dates         = True,
                                 extend_measure_names = False )
    
        for countryname, measuredata in data:
            // do stuff with measuredata
    
    ***************************************************
    
    measuredata is dictionary:
        keys: name of measures 
        values: list of dates when implemented
    
    only return measures that correspond to 'measure_level' = [1 .. 4]
    if 'only_first_dates == True' only return date of first occurence of measure for this level, otherwise whole list
    if 'extend_measure_dates == True' keys are changed to include all names of all levels of measures
    if 'unique_dates == True' remove duplicate days in list of values

    ***************************************************        
    '''
    
    def __init__(self,**kwargs):
        self.DATAFILE             = 'COVID19_non-pharmaceutical-interventions.csv'
        self.BASEURL              = 'https://raw.githubusercontent.com/amel-github/covid19-interventionmeasures/master/'
        
        # set default values of options
        self.__downloaddata       = kwargs.get('download_data',        False )
        self.__measurelevel       = kwargs.get('measure_level',        2     )
        self.__onlyfirstdates     = kwargs.get('only_first_dates',     False )
        self.__uniquedates        = kwargs.get('unique_dates',         True  )
        self.__extendmeasurenames = kwargs.get('extend_measure_names', False )
        self.__countrycodes       = kwargs.get('country_codes',        False )
        
        # can switch internal declaration of countries completely to the ISO3C countrycodes
        # no full names of countries can be used then
        if self.__countrycodes:
            self.__countrycolumn  = 'iso3c'
        else:
            self.__countrycolumn  = 'Country'
        
        self.ReadData()
    
    
    def DownloadData(self):
        tmpdata = pd.read_csv(self.BASEURL + self.DATAFILE, sep = ',', quotechar = '"', encoding = 'latin-1')
        tmpdata.to_csv(self.DATAFILE)
    
    
    def ReadData(self):
        if not os.path.exists(self.DATAFILE) or self.__downloaddata:
            self.DownloadData()

        self.__data        = pd.read_csv(self.DATAFILE, sep = ',', quotechar = '"', encoding = 'latin-1')
        self.__countrylist = list(self.__data[self.__countrycolumn].unique())
    
    
    def CountryData(self, country = None, measure_level = None, only_first_dates = None, unique_dates = None, extend_measure_names = None):
        if country in self.__countrylist:
            
            if measure_level is None:        measure_level        = self.__measurelevel
            if only_first_dates is None:     only_first_dates     = self.__onlyfirstdates
            if unique_dates is None:         unique_dates         = self.__uniquedates
            if extend_measure_names is None: extend_measure_names = self.__extendmeasurenames
            
            countrydata           = self.__data[self.__data[self.__countrycolumn] == country]
            if measure_level >= 2:
                for ml in range(2,measure_level+1):
                    # fill columns with previous measure levels, if empty (otherwise the empty fields generate errors)
                    countrydata['Measure_L{:d}'.format(ml)] = countrydata['Measure_L{:d}'.format(ml)].fillna(countrydata['Measure_L{:d}'.format(ml-1)])
            
            # make new column, which will be grouped below
            if extend_measure_names:
                countrydata['MN'] = countrydata[['Measure_L{:d}'.format(ml+1) for ml in range(measure_level)]].agg(' - '.join, axis = 1)
            else:
                countrydata['MN'] = countrydata['Measure_L{:d}'.format(measure_level)]
            
            # drop all entries which don't have date associated
            countrydata           = countrydata[countrydata['Date'].notna()]
            mgdata                = countrydata.groupby(by = 'MN')['Date']
            
            if unique_dates:
                mgdata            = mgdata.apply(set)
            
            # rebuild as dict
            mgdata                = dict((k.strip(),v) for k,v in dict(mgdata.apply(list)).items())
            if only_first_dates:
                mgdata            = dict((k,[min(v)]) for k,v in mgdata.items())
                
            return mgdata
        else:
            return None
    
    
    def RawData(self,country):
        if country in self.__countrylist:
                return self.__data[self.__data[self.__countrycolumn] == country].reset_index()
    
    
    def FindMeasure(self, country, measure_name, measure_level):
        cd = self.CountryData(country, measure_level = measure_level)
        if measure_name in cd.keys():
            return cd[measure_name][0]
        else:
            return None
    
    
    def __getattr__(self,key):
        if key in self.__countrylist:
            return self.GetCountryData(country = key)
        elif key == 'countrylist':
            return self.__countrylist
        
    
    def __iter__(self):
        for country in self.__countrylist:
            yield country,self.CountryData(country = country)
        
    
