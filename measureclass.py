import numpy as np
import pandas as pd


class COVID19_measures(object):
    def __init__(self,**kwargs):
    
        self.__datafile           = kwargs.get('datafile','../data/COVID19_measures_clean.csv')
        self.__measurelevel       = kwargs.get('measure_level',1)
        self.__onlyfirstdates     = kwargs.get('only_first_dates',True)
        self.__uniquedates        = kwargs.get('unique_dates',True)
        self.__extendmeasurenames = kwargs.get('extend_measure_names',False)
        
        self.ReadData()
    
    
    def SetMeasureLevel(self, level = 1):
        self.__measurelevel = level
    
    
    def ReadData(self):
        self.__data        = pd.read_csv(self.__datafile,sep = ';')
        self.__countrylist = list(self.__data['Country'].unique())
    
    
    def GetCountryData(self, country = None, measure_level = 1, only_first_dates = True):
        cdata = self.__data[self.__data['Country'] == country]
        groupkey = 'Measure_L{:d}'.format(measure_level)
        rdata = cdata.groupby(by = groupkey)['Date']
        if self.__uniquedates:
            rdata = rdata.apply(set)
        rdata = dict(rdata.apply(list))
        if only_first_dates:
            rdata = dict((k,[min(v)]) for k,v in rdata.items())
        if (self.__extendmeasurenames) and (measure_level > 1):
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
            yield country,self.GetCountryData(country = country, measure_level = self.__measurelevel, only_first_dates = self.__onlyfirstdates)
        
    
