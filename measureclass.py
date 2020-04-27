import numpy as np
import pandas as pd
import os
import datetime
import re

class COVID19_measures(object):
    '''
    ***************************************************
    **  wrapper for COVID19 measures tracked by CSH  **
    **  github.com/lukasgeyrhofer/corona/            **
    ***************************************************

    two sources for implemented measures possible:
    
    * CSH
       https://github.com/amel-github/covid19-interventionmeasures
       compiled by Desvars-Larrive et al (2020), CC-BY-SA 4.0
    * Oxford
       https://www.bsg.ox.ac.uk/research/research-projects/coronavirus-government-response-tracker
       https://ocgptweb.azurewebsites.net/CSVDownload
       complied by Hale, Webster, Petherick, Phillips, Kira (2020), CC-BY-SA 4.0
       
    read table of measures from CSV file,
    and download data from github if not present or forced
    
    main usage:
    ( with default options )
    ***************************************************
    
        data = COVID19_measures( datasource           = 'CSH',
                                 download_data        = False,
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
        
        # set default values of options
        self.__downloaddata       = kwargs.get('download_data',        False )
        self.__measurelevel       = kwargs.get('measure_level',        2     )
        self.__onlyfirstdates     = kwargs.get('only_first_dates',     False )
        self.__uniquedates        = kwargs.get('unique_dates',         True  )
        self.__extendmeasurenames = kwargs.get('extend_measure_names', False )
        self.__countrycodes       = kwargs.get('country_codes',        False )
        self.__dateformat         = kwargs.get('dateformat',           '%d/%m/%Y')
        self.__removedcountries   = []
        
        self.__datasource         = (kwargs.get('datasource','CSH')).upper()
        self.__datasourceinfo     = {   'CSH':    {'dateformat':          '%d/%m/%Y',
                                                   'Country':             'Country',
                                                   'CountryCodes':        'iso3c',
                                                   'MaxMeasureLevel':     4,
                                                   'DownloadURL':         'https://raw.githubusercontent.com/amel-github/covid19-interventionmeasures/master/COVID19_non-pharmaceutical-interventions.csv',
                                                   'DatafileName':        'COVID19_non-pharmaceutical-interventions.csv',
                                                   'DatafileReadOptions': {'sep': ',', 'quotechar': '"', 'encoding': 'latin-1'}},
                                        'OXFORD': {'dateformat':          '%Y%m%d',
                                                   'Country':             'CountryName',
                                                   'CountryCodes':        'CountryCode',
                                                   'MaxMeasureLevel':     1,
                                                   'DownloadURL':         'https://ocgptweb.azurewebsites.net/CSVDownload',
                                                   'DatafileName':        'OxCGRT_Full.csv',
                                                   'DatafileReadOptions': {}}
                                    }
        
        # can switch internal declaration of countries completely to the ISO3C countrycodes
        # no full names of countries can be used then
        if self.__countrycodes:   self.__countrycolumn  = self.__datasourceinfo[self.__datasource]['CountryCodes']
        else:                     self.__countrycolumn  = self.__datasourceinfo[self.__datasource]['Country']

        if self.__datasource in self.__datasourceinfo.keys():
            self.ReadData()
        else:
            raise NotImplementedError



    def DownloadData(self):
        tmpdata = pd.read_csv(self.__datasourceinfo[self.__datasource]['DownloadURL'],**self.__datasourceinfo[self.__datasource]['DatafileReadOptions'])
        tmpdata.to_csv(self.__datasourceinfo[self.__datasource]['DatafileName'])


    
    def convertDate(self,datestr):
        return datetime.datetime.strptime(str(datestr),self.__datasourceinfo[self.__datasource]['dateformat']).strftime(self.__dateformat)


    
    def ReadData(self):
        if not os.path.exists(self.__datasourceinfo[self.__datasource]['DatafileName']) or self.__downloaddata:
            self.DownloadData()
        
        readdata           = pd.read_csv(self.__datasourceinfo[self.__datasource]['DatafileName'],**self.__datasourceinfo[self.__datasource]['DatafileReadOptions'])
        self.__countrylist = list(readdata[self.__countrycolumn].unique())
        
        if self.__datasource == 'CSH':
            # store CSV directly as data
            self.__data    = readdata
    
        elif self.__datasource == 'OXFORD':
            # construct list of measures from DB column names
            # naming scheme is 'S[NUMBER]_NAME'
            # in addition to columns 'S[NUMBER]_IsGeneral' and 'S[NUMBER]_Notes' for more info
            measurecolumns = []
            for mc in readdata.columns:
                if not re.search('^S\d+\_',mc) is None:
                    if mc[-7:].lower() != 'general' and mc[-5:].lower() != 'notes':
                        measurecolumns.append(mc)
            
            # reconstruct same structure of CSH DB bottom up
            self.__data    = None
            for country in self.__countrylist:
                countrydata = readdata[readdata[self.__countrycolumn] == country]
                for mc in measurecolumns:
                    for date in countrydata[countrydata[mc].diff() > 0]['Date']:
                        db_entry_dict = {self.__countrycolumn: country, 'Date': self.convertDate(date), 'Measure_L1': mc}
                        if self.__data is None:
                            self.__data = pd.DataFrame({k:np.array([v]) for k,v in db_entry_dict.items()})
                        else:
                            self.__data = self.__data.append(db_entry_dict, ignore_index = True)
        
        else:
            NotImplementedError
            
    
    
    def RemoveCountry(self, country = None):
        if country in self.__countrylist:
            self.__removedcountries.append(country)
            self.__countrylist.remove(country)
    
    
    
    def RenameCountry(self, country = None, newname = None):
        if country in self.__countrylist:
            self.__countrylist.remove(country)
            self.__countrylist.append(newname)
            self.__countrylist.sort()
            self.__data.replace(to_replace = country, value = newname, inplace = True)
    
    
    
    def SortDates(self,datelist):
        tmp_datelist = list(datelist[:])
        tmp_datelist.sort(key = lambda x:datetime.datetime.strptime(x,self.__dateformat))
        return tmp_datelist
    
    
    
    def CountryData(self, country = None, measure_level = None, only_first_dates = None, unique_dates = None, extend_measure_names = None):
        if country in self.__countrylist:
            
            if measure_level is None:        measure_level        = self.__measurelevel
            if only_first_dates is None:     only_first_dates     = self.__onlyfirstdates
            if unique_dates is None:         unique_dates         = self.__uniquedates
            if extend_measure_names is None: extend_measure_names = self.__extendmeasurenames
            
            if measure_level > self.__datasourceinfo[self.__datasource]['MaxMeasureLevel']: measure_level = self.__datasourceinfo[self.__datasource]['MaxMeasureLevel']

            countrydata           = self.__data[self.__data[self.__countrycolumn] == country].copy(deep = True)
            if measure_level >= 2:
                for ml in range(2,measure_level+1):
                    # fill columns with previous measure levels, if empty (otherwise the empty fields generate errors)
                    countrydata.loc[:,'Measure_L{:d}'.format(ml)] = countrydata['Measure_L{:d}'.format(ml)].fillna(countrydata['Measure_L{:d}'.format(ml-1)])
            
            # make new column, which will be grouped below
            if extend_measure_names:
                countrydata.insert(1, 'MN', np.array(countrydata[['Measure_L{:d}'.format(ml+1) for ml in range(measure_level)]].agg(' -- '.join, axis = 1)), True)
            else:
                countrydata.insert(1, 'MN',np.array(countrydata['Measure_L{:d}'.format(measure_level)]), True)
            
            # drop all entries which don't have date associated
            countrydata           = countrydata[countrydata['Date'].notna()]
            mgdata                = countrydata.groupby(by = 'MN')['Date']
            
            if unique_dates:
                mgdata            = mgdata.apply(set)
            
            # rebuild as dict
            mgdata                = {k.strip():self.SortDates(v) for k,v in dict(mgdata.apply(list)).items()}
            if only_first_dates:
                mgdata            = {k:[v[0]] for k,v in mgdata.items()}
                
            return mgdata
        else:
            return None
    
    
    
    def RawData(self,country):
        if country in self.__countrylist:
                return self.__data[self.__data[self.__countrycolumn] == country].reset_index()
    
    

    def date2vector(self, implementdate, start = '22/1/2020', end = None, shiftdays = 0, maxlen = None, datefmt = '%d/%m/%Y'):
        # generate vector of 0s and 1s when measure is implemented or not
        starttime     = datetime.datetime.strptime(start,         datefmt)
        if end is None:
            endtime   = datetime.datetime.today()
        else:
            endtime   = datetime.datetime.strptime(end,           datefmt)
        implementtime = datetime.datetime.strptime(implementdate, datefmt)
        
        totaldays   = (endtime       - starttime).days
        measuredays = (implementtime - starttime).days
        
        vec         = np.zeros(totaldays)
        vec[min(measuredays+shiftdays,len(vec)-1):] = 1
        
        if not maxlen is None:
            vec     = vec[:min(maxlen,len(vec))]
        
        return vec



    def CleanUpMeasureName(self, measurename = '', clean_up = True):
        if isinstance(measurename,str) and clean_up:
                return ''.join([mn.capitalize() for mn in measurename.replace('/','').replace(',','').replace('-',' ').replace('_',' ').split(' ')])
        return measurename



    def ImplementationTable(self, country, measure_level = None, startdate = '22/1/20', enddate = None, shiftdays = 0, maxlen = None, clean_measurename = False):
        if country in self.__countrylist:
            countrydata = self.CountryData(country = country, measure_level = measure_level, only_first_dates = True)
            return pd.DataFrame(    {   self.CleanUpMeasureName(measurename, clean_up = clean_measurename):
                                        self.date2vector(implemented[0], start = startdate, end = enddate, shiftdays = shiftdays, maxlen = maxlen)
                                        for measurename, implemented in countrydata.items()
                                    } )
        else:
            return None

    
    
    def MeasureList(self, countrylist = None, measure_level = None, mincount = None, extend_measure_names = None, clean_measurename = False):
        if extend_measure_names is None:    extend_measure_neames = self.__extendmeasurenames
        if countrylist is None:             countrylist = self.__countrylist # use ALL countries
        measurelist = {}
        # get all restrictions from countries
        for country in countrylist:
            country_measures = self.CountryData(country, measure_level = measure_level, extend_measure_names = extend_measure_names)
            for measurename, initialdata in country_measures.items():
                if not measurename in measurelist.keys():
                    measurelist[measurename] = 0
                measurelist[measurename] += 1
        if not mincount is None:
            # rebuild dict with restrictions
            measurelist = {self.CleanUpMeasureName(k,clean_measurename):v for k,v in measurelist.items() if v >= mincount}
        return measurelist
    
    
    
    def FindMeasure(self, country, measure_name, measure_level):
        cd = self.CountryData(country, measure_level = measure_level)
        if measure_name in cd.keys():
            return cd[measure_name][0]
        else:
            return None
    
    
    def GetMeasureNames(self, measure_level = None, cleaned_measurelist = None):
        if measure_level is None: measure_level = self.__measurelevel
        if measure_level > self.__datasourceinfo[self.__datasource]['MaxMeasureLevel']: measure_level = self.__datasourceinfo[self.__datasource]['MaxMeasureLevel']

        measurenameDF = pd.DataFrame(self.__data[['Measure_L{:d}'.format(ml+1) for ml in range(measure_level)]]).replace(np.nan,'',regex=True)
        measurenameDF.index = list(self.__data['Measure_L{}'.format(measure_level)].apply(self.CleanUpMeasureName))
        measurenameDF.columns = ['Measure_L{:d}'.format(ml+1) for ml in range(measure_level)]
        measurenameDF.drop_duplicates(inplace = True)

        return measurenameDF
    
    
    def __getattr__(self,key):
        if key in self.__countrylist:
            return self.CountryData(country = key)
        elif key == 'countrylist':
            return self.__countrylist
    
    
    
    def __iter__(self):
        for country in self.__countrylist:
            yield country,self.CountryData(country = country)
        
    
