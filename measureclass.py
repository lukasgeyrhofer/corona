import numpy as np
import pandas as pd
import os
import datetime
import re
import urllib.request
import zipfile

class COVID19_measures(object):
    '''
    ***************************************************
    **  wrapper for COVID19 measures                 **
    **  github.com/lukasgeyrhofer/corona/            **
    ***************************************************

    multiple sources for implemented measures possible:
    
    * CSH
       https://github.com/amel-github/covid19-interventionmeasures
       compiled by Desvars-Larrive et al (2020), CC-BY-SA 4.0
    * Oxford
       https://www.bsg.ox.ac.uk/research/research-projects/coronavirus-government-response-tracker
       https://ocgptweb.azurewebsites.net/CSVDownload
       complied by Hale, Webster, Petherick, Phillips, Kira (2020), CC-BY-SA 4.0
    * ACAPS
       https://www.acaps.org/covid19-government-measures-dataset
       info@acaps.org
       README: https://www.acaps.org/sites/acaps/files/key-documents/files/acaps_covid-19_government_measures_dataset_readme.pdf
       
    read table of measures from CSV file,
    and download data from github if not present or forced
    
    main usage:
    ( with default options )
    ***************************************************
        # initialize dataset    
        measure_data = COVID19_measures( datasource           = 'CSH',
                                         download_data        = False,
                                         measure_level        = 2,
                                         only_first_dates     = False,
                                         unique_dates         = True,
                                         extend_measure_names = False )


        # dataset is iterable
        
        for countryname, measuredata in data:
            # do stuff with measuredata
            # measuredata is dictionary:
            #   keys: name of measures 
            #   values: list of dates when implemented


        # obtain DF with columns of implemented measures (0/1) over timespan as index
        # only 'country' is required option, defaults as below
        
        imptable = measure_data.ImplementationTable( country       = 'Austria',
                                                     measure_level = 2,
                                                     startdata     = '22/1/2020',
                                                     enddate       = None,  # today
                                                     shiftdays     = 0 )
    
    ***************************************************
    
    options at initialization:
     * only return measures that correspond to 'measure_level' = [1 .. 4]
     * if 'only_first_dates == True' only return date of first occurence of measure for this level, otherwise whole list
     * if 'extend_measure_dates == True' keys are changed to include all names of all levels of measures
     * if 'unique_dates == True' remove duplicate days in list of values

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
        
        self.__datasource         = kwargs.get('datasource','CSH').upper()
        self.__datasourceinfo     = {   'CSH':    {'dateformat':          '%Y-%m-%d',
                                                   'Country':             'Country',
                                                   'CountryCodes':        'iso3c',
                                                   'MaxMeasureLevel':     4,
                                                   'DownloadURL':         'https://raw.githubusercontent.com/amel-github/covid19-interventionmeasures/master/COVID19_non-pharmaceutical-interventions_version2_utf8.csv',
                                                   'DatafileName':        'COVID19_non-pharmaceutical-interventions.csv',
                                                   'DatafileReadOptions': {'sep': ',', 'quotechar': '"', 'encoding': 'latin-1'}},
                                        'OXFORD': {'dateformat':          '%Y%m%d',
                                                   'Country':             'CountryName',
                                                   'CountryCodes':        'CountryCode',
                                                   'MaxMeasureLevel':     1,
                                                   'DownloadURL':         'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv',
                                                   'DatafileName':        'OxCGRT_latest.csv',
                                                   'DatafileReadOptions': {}},
                                        'ACAPS':  {'dateformat':          '%d/%m/%Y',
                                                   'Country':             'COUNTRY',
                                                   'CountryCodes':        'ISO',
                                                   'MaxMeasureLevel':     2,
                                                   'DownloadURL':         'https://www.acaps.org/sites/acaps/files/resources/files/acaps_covid19_goverment_measures_dataset.xlsx',
                                                   'DatafileName':        'ACAPS_covid19_measures.xlxs',
                                                   'DatafileReadOptions': {'sheet_name':'Database'}},
                                        'WHOPHSM':{'dateformat':          '%d/%m/%Y',
                                                   'Country':             'country_territory_area',
                                                   'CountryCodes':        'iso',
                                                   'DownloadURL':         'https://www.who.int/docs/default-source/documents/phsm/phsm-who.zip',
                                                   'DatafileName':        'who_phsm.csv',
                                                   'MaxMeasureLevel':      2,
                                                   'DatafileReadOptions': {'encoding':'latin-1'}}
                                    }

        self.__update_dsinfo = kwargs.get('datasourceinfo',None)
        if not self.__update_dsinfo is None:
            self.__datasourceinfo[self.__datasource].update(self.__update_dsinfo)
        
        # can switch internal declaration of countries completely to the ISO3C countrycodes
        # no full names of countries can be used then
        if self.__countrycodes:   self.__countrycolumn  = self.__datasourceinfo[self.__datasource]['CountryCodes']
        else:                     self.__countrycolumn  = self.__datasourceinfo[self.__datasource]['Country']



        if self.__datasource in self.__datasourceinfo.keys():
            self.ReadData()
        else:
            raise NotImplementedError



    def DownloadData(self):
        urllib.request.urlretrieve(self.__datasourceinfo[self.__datasource]['DownloadURL'],self.__datasourceinfo[self.__datasource]['DatafileName'])

        # download for WHO PHSM comes as zipfile. need to extract file first
        if self.__datasource == 'WHOPHSM':
            who_archive = zipfile.ZipFile('phsm-who.zip')
            for fileinfo in who_archive.infolist():
                if '.csv' in fileinfo.filename:
                    who_archive.extract(fileinfo)
                    who_filename = fileinfo.filename
            os.rename(who_filename,self.__datasourceinfo['WHOPHSM']['DatafileName'])


    
    def convertDate(self,datestr, inputformat = None, outputformat = None):
        if inputformat is None: inputformat = self.__datasourceinfo[self.__datasource]['dateformat']
        if outputformat is None: outputformat = self.__dateformat
        return datetime.datetime.strptime(str(datestr),inputformat).strftime(outputformat)



    def filetype(self, datasource = None):
        return str(os.path.splitext(self.__datasourceinfo[datasource]['DatafileName'])[1]).strip('.').upper()



    
    
    
    def ReadData(self):
        def CleanWHOName(name):
            return name.replace('nan -- ','').replace(' -- nan','')
        
        if not os.path.exists(self.__datasourceinfo[self.__datasource]['DatafileName']) or self.__downloaddata:
            self.DownloadData()
        
        if self.filetype(self.__datasource) == 'CSV':
            readdata       = pd.read_csv(self.__datasourceinfo[self.__datasource]['DatafileName'],**self.__datasourceinfo[self.__datasource]['DatafileReadOptions'])
        elif self.filetype(self.__datasource) == 'XLXS':
            readdata       = pd.read_excel(self.__datasourceinfo[self.__datasource]['DatafileName'], **self.__datasourceinfo[self.__datasource]['DatafileReadOptions'])
        else:
            raise NotImplementedError
        self.__countrylist = list(readdata[self.__countrycolumn].unique())
        
        if self.__datasource == 'CSH':
            # store CSV directly as data
            self.__data    = readdata.copy(deep = True)
            self.__data['Date'] = self.__data['Date'].apply(self.convertDate)
    
        elif self.__datasource == 'OXFORD':
            # construct list of measures from DB column names
            # naming scheme is 'S[NUMBER]_NAME'
            # in addition to columns 'S[NUMBER]_IsGeneral' and 'S[NUMBER]_Notes' for more info
            measurecolumns = []
            for mc in readdata.columns:
                if not re.search('^[CEH]\d+\_',mc) is None:
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
        
        elif self.__datasource == 'ACAPS':
            self.__data = readdata[[self.__countrycolumn,'DATE_IMPLEMENTED','CATEGORY','MEASURE']].copy(deep = True)
            self.__data.columns = [self.__countrycolumn,'Date', 'Measure_L1', 'Measure_L2']
            self.__data.dropna(inplace = True)
            self.__data['Date'] = self.__data['Date'].dt.strftime(self.__dateformat)
        
        elif self.__datasource == 'WHOPHSM':
            self.__data = readdata[[self.__countrycolumn,'date_start','who_category']].copy(deep = True)
            self.__data.columns = [self.__countrycolumn,'Date','Measure_L1']
            self.__data['Measure_L2'] = (readdata['who_subcategory'].astype(str) + ' -- ' + readdata['who_measure'].astype(str)).apply(CleanWHOName)
            # some cleanup, might not be enough
            self.__data.dropna(subset = ['Date'], inplace = True)
            self.__data.drop(self.__data[self.__data['Measure_L2'] == 'nan'].index, inplace = True)
            self.__data.drop(self.__data[self.__data['Measure_L2'] == 'unkown -- unknown'].index, inplace = True)
            self.__data['Date'] = self.__data['Date'].apply(self.convertDate)
        
        else:
            NotImplementedError
            
    
    
    def RemoveCountry(self, country = None):
        if country in self.__countrylist:
            self.__countrylist.remove(country)
            self.__data = self.__data[self.__data[self.__countrycolumn] != country]
    
    
    
    def RenameCountry(self, country = None, newname = None):
        if country in self.__countrylist:
            self.__countrylist.remove(country)
            self.__countrylist.append(newname)
            self.__countrylist.sort()
            self.__data.replace(to_replace = {self.__countrycolumn: country}, value = newname, inplace = True)
    
    
    
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
    
    

    def dates2vector(self, implementdate, start = '22/1/2020', end = None, shiftdays = 0, maxlen = None, datefmt = '%d/%m/%Y', only_pulse = False, binary_output = False):
        # generate vector of 0s and 1s when measure is implemented or not
        # or, when 'only_pulse == True', then output 1 only at dates of implementation
        starttime     = datetime.datetime.strptime(start,         datefmt)
        if end is None:
            endtime   = datetime.datetime.today()
        else:
            endtime   = datetime.datetime.strptime(end,           datefmt)
        implementlist = [datetime.datetime.strptime(date, datefmt) for date in self.SortDates(implementdate)]

        totaldays   = (endtime - starttime).days + 1
        vec         = np.zeros(totaldays)

        if only_pulse:
            for implementtime in implementlist:
                measuredays = (implementtime - starttime).days
                if 0 <= measuredays+shiftdays < len(vec):
                    vec[measuredays+shiftdays] = 1
        else:
            measuredays = (implementlist[0] - starttime).days
            if 0 <= measuredays + shiftdays < len(vec):
                vec[measuredays+shiftdays:] = 1
            
        if not maxlen is None:
            vec     = vec[:min(maxlen,len(vec))]
        
        if binary_output:
            vec = np.array(vec,dtype=np.bool)
        
        return vec



    def CleanUpMeasureName(self, measurename = '', clean_up = True):
        if isinstance(measurename,str) and clean_up:
                return ''.join([mn.capitalize() for mn in measurename.replace('/','').replace(',','').replace('-',' ').replace('_',' ').split(' ')])
        return measurename



    def ImplementationTable(self, country, measure_level = None, startdate = '22/1/2020', enddate = None, shiftdays = 0, maxlen = None, clean_measurename = True, only_pulse = False, binary_output = False, extend_measure_names = False):
        if country in self.__countrylist:
            countrydata  = self.CountryData(country = country, measure_level = measure_level, only_first_dates = False, extend_measure_names = extend_measure_names)
            ret_imptable = pd.DataFrame( { self.CleanUpMeasureName(measurename, clean_up = clean_measurename):
                                           self.dates2vector(implemented, start = startdate, end = enddate, shiftdays = shiftdays, maxlen = maxlen, only_pulse = only_pulse, binary_output = binary_output)
                                           for measurename, implemented in countrydata.items() } )
            ret_imptable.index = [(datetime.datetime.strptime(startdate,'%d/%m/%Y') + datetime.timedelta(days = i)).strftime(self.__dateformat) for i in range(len(ret_imptable))]
            return ret_imptable

        else:
            return None

    
    
    def FindMeasure(self, country, measure_name, measure_level):
        cd = self.CountryData(country, measure_level = measure_level)
        if measure_name in cd.keys():
            return cd[measure_name][0]
        else:
            return None
    
    
    
    def MeasureList(self, countrylist = None, measure_level = None, mincount = None, enddate = None):
        if measure_level is None: measure_level = self.__measurelevel
        if measure_level > self.__datasourceinfo[self.__datasource]['MaxMeasureLevel']: measure_level = self.__datasourceinfo[self.__datasource]['MaxMeasureLevel']

        if enddate is None: enddate = datetime.datetime.today().strftime(self.__dateformat)
        mheaders = ['Measure_L{:d}'.format(ml+1) for ml in range(measure_level)]
        measurenameDF = pd.DataFrame(self.__data[mheaders + [self.__countrycolumn,'Date']]).replace(np.nan,'',regex=True)
        measurenameDF.drop(measurenameDF[measurenameDF['Date'].astype(np.datetime64) <=  np.datetime64(self.convertDate(enddate,inputformat = self.__dateformat,outputformat='%Y-%m-%d'))].index,inplace=True)
        measurenameDF.drop('Date',axis = 'columns', inplace = True)
        measurenameDF.drop_duplicates(inplace=True)
        if not countrylist is None: measurenameDF = measurenameDF[measurenameDF[self.__countrycolumn].isin(countrylist)]
        measurenameDF = measurenameDF.groupby(by = mheaders, as_index=False).count()
        measurenameDF.columns = mheaders + ['Countries with Implementation']
        measurenameDF.index = list(measurenameDF['Measure_L{:d}'.format(measure_level)].apply(self.CleanUpMeasureName))

        if not mincount is None: measurenameDF = measurenameDF[measurenameDF['Countries with Implementation'] >= mincount]

        return measurenameDF
    
    
    
    def __getattr__(self,key):
        if key in self.__countrylist:
            return self.CountryData(country = key)
        elif key == 'countrylist':
            return self.__countrylist
        elif key == 'rawdata':
            return self.__data  
    
    
    
    def __iter__(self):
        for country in self.__countrylist:
            yield country,self.CountryData(country = country)
        
    
