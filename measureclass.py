import numpy as np
import pandas as pd
import os

class COVID19_measures(object):
    '''
    read table of measures from CSV file
    
    main usage:
    
    data = COVID19_measures(datafile = DATAFILE)
    
    if datafile not provided, download from CSH github
    
    for countryname, measuredata in data:
        // do stuff with measuredata
    
    measuredata is dictionary:
        keys: name of measures 
        values: list of dates when implemented
    
    only return measures that correspond to 'measure_level' = [1 .. 4]
    if 'only_first_dates == True' only return date of first occurence of measure for this level, otherwise whole list
    if 'extend_measure_dates == True' keys are changed to include all names of all levels of measures
    if 'unique_dates == True' remove duplicate days in list of values
        
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
        
        self.ReadData()
    
    
    def DownloadData(self):
        tmpdata = pd.read_csv(self.BASEURL + self.DATAFILE, sep = ',', quotechar = '"', encoding = 'latin-1')
        tmpdata.to_csv(self.DATAFILE)
    
    def ReadData(self):
        if not os.path.exists(self.DATAFILE) or self.__downloaddata:
            self.DownloadData()

        self.__data        = pd.read_csv(self.DATAFILE, sep = ',', quotechar = '"', encoding = 'latin-1')
        self.__countrylist = list(self.__data['Country'].unique())
    
    
    def GetCountryData(self, country = None, measure_level = None, only_first_dates = None, unique_dates = None, extend_measure_names = None):
        if country in self.__countrylist:
            
            if measure_level is None:        measure_level        = self.__measurelevel
            if only_first_dates is None:     only_first_dates     = self.__onlyfirstdates
            if unique_dates is None:         unique_dates         = self.__uniquedates
            if extend_measure_names is None: extend_measure_names = self.__extendmeasurenames
            
            
            groupkey = 'Measure_L{:d}'.format(measure_level)
            cdata    = self.__data[self.__data['Country'] == country]
            rdata    = cdata.groupby(by = groupkey)['Date']

            if unique_dates:
                rdata = rdata.apply(set)

            rdata = dict(rdata.apply(list))

            if only_first_dates:
                rdata = dict((k,[min(v)]) for k,v in rdata.items())

            if extend_measure_names and measure_level > 1:
                tmprdata = dict()
                for k,v in rdata.items():
                    measure_names = []
                    for i in range(1,measure_level):                    
                        measure_names.append( (cdata[cdata['Measure_L{}'.format(measure_level)] == k])['Measure_L{}'.format(i)].tolist()[0] )
                    measure_names.append(k)
                    tmprdata[' - '.join(measure_names)] = v
                rdata = tmprdata
                
            return rdata
        else:
            return None
    
    
    def __getattr__(self,key):
        if key in self.__countrylist:
            return self.GetCountryData(country = key)
        elif key == 'countrylist':
            return self.__countrylist
        
    
    def __iter__(self):
        for country in self.__countrylist:
            yield country,self.GetCountryData(country = country)
        
    
