import numpy as np
import pandas as pd


class COVID19_measures(object):
    '''
    read table of measures from CSV file
    
    main usage:
    
    data = COVID19_measures(datafile = DATAFILE)
    
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
    
        self.__datafile           = kwargs.get('datafile','../data/COVID19_measures_clean.csv')
        self.__measurelevel       = kwargs.get('measure_level',1)
        self.__onlyfirstdates     = kwargs.get('only_first_dates',True)
        self.__uniquedates        = kwargs.get('unique_dates',True)
        self.__extendmeasurenames = kwargs.get('extend_measure_names',False)
        
        self.ReadData()
    
    
    def ReadData(self):
        self.__data        = pd.read_csv(self.__datafile,sep = ';')
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
    
    
    def __iter__(self):
        for country in self.__countrylist:
            yield country,self.GetCountryData(country = country)
        
    
