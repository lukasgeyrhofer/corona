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
        self.__resolve_US_states  = kwargs.get('resolve_US_states',    False)
        self.__removedcountries   = []
        
        self.__max_date_check     = 40 # how many days back from today
        
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
                                                   'DownloadURL':         'https://www.acaps.org/sites/acaps/files/resources/files/acaps_covid19_government_measures_dataset.xlsx',
                                                   'DatafileName':        'ACAPS_covid19_measures.xlsx',
                                                   'DatafileReadOptions': {'sheet_name':'Database'}},
                                        'WHOPHSM':{'dateformat':          '%d/%m/%Y',
                                                   'Country':             'country_territory_area',
                                                   'CountryCodes':        'iso',
                                                   'DownloadURL':         'https://www.who.int/docs/default-source/documents/phsm/{DATE}-phsm-who-int.zip',
                                                   'DownloadURL_dateformat': '%Y%m%d',
                                                   'DownloadFilename':    'who_phsm.zip',
                                                   'DatafileName':        'who_phsm.xlsx',
                                                   'MaxMeasureLevel':      2,
                                                   'DatafileReadOptions': {'encoding':'latin-1'}},
                                        'CORONANET':{'dateformat':        '%Y-%m-%d',
                                                   'DownloadURL':         'http://coronanet-project.org/data/coronanet_release.csv',
                                                   'DatafileName':        'coronanet_release.csv',
                                                   'MaxMeasureLevel':     3,
                                                   'Country':             'country',
                                                   'DatafileReadOptions': {}},
                                        'HITCOVID':{'dateformat':         '%Y-%m-%d',
                                                   'DownloadURL':         'https://github.com/HopkinsIDD/hit-covid/raw/master/data/hit-covid-longdata.csv',
                                                   'DatafileName':        'hit-covid-longdata.csv',
                                                   'MaxMeasureLevel':     2,
                                                   'Country':             'country_name',
                                                   'DatafileReadOptions': {}}
                                    }

        if not self.__datasource in self.__datasourceinfo.keys():
            raise NotImplementedError('Implemented databases: [' + ', '.join('{}'.format(dbname) for dbname in self.__datasourceinfo.keys()) + ']')
        
        self.__update_dsinfo = kwargs.get('datasourceinfo',None)
        if not self.__update_dsinfo is None:
            self.__datasourceinfo[self.__datasource].update(self.__update_dsinfo)
        
        # can switch internal declaration of countries completely to the ISO3C countrycodes
        # no full names of countries can be used then
        if self.__countrycodes:
            self.__countrycolumn  = self.__datasourceinfo[self.__datasource]['CountryCodes']
            self.__USname         = 'USA'
        else:
            self.__countrycolumn  = self.__datasourceinfo[self.__datasource]['Country']
            self.__USname         = 'United States of America'
        
        # after setting all options and parameters: load data
        self.ReadData()


    def URLexists(self, url):
        request = urllib.request.Request(url)
        request.get_method = lambda: 'HEAD'
        try:
            urllib.request.urlopen(request)
            return True
        except urllib.request.HTTPError:
            return False


    def filetype(self, datasource = None, filename = None):
        if filename is None:
            filename = self.__datasourceinfo[datasource]['DatafileName']
        return os.path.splitext(filename)[1].strip('.').upper()
    
    
    def DownloadData(self):
        # if cannot directly download CSV files, but need to download something else first
        if 'DownloadFilename' in self.__datasourceinfo[self.__datasource].keys():
            download_savefile = self.__datasourceinfo[self.__datasource]['DownloadFilename']
        else:
            download_savefile = self.__datasourceinfo[self.__datasource]['DatafileName']
        
        # if download filename contains a date, try different dates, starting from today, with max (self.__max_date_check) days back
        if '{DATE}' in self.__datasourceinfo[self.__datasource]['DownloadURL']:
            d = 0
            while not self.URLexists(self.__datasourceinfo[self.__datasource]['DownloadURL'].format(DATE = (datetime.datetime.today() - datetime.timedelta(days = d)).strftime(self.__datasourceinfo[self.__datasource]['DownloadURL_dateformat']))):
                d += 1
                if d >= self.__max_date_check:
                    break
            download_url = self.__datasourceinfo[self.__datasource]['DownloadURL'].format(DATE = (datetime.datetime.today() - datetime.timedelta(days = d)).strftime(self.__datasourceinfo[self.__datasource]['DownloadURL_dateformat']))
        else:
            download_url = self.__datasourceinfo[self.__datasource]['DownloadURL']
        
        # download actual data
        urllib.request.urlretrieve(download_url, download_savefile)

        # download for WHO PHSM comes as zipfile. need to extract file first
        if self.__datasource == 'WHOPHSM':
            who_archive = zipfile.ZipFile(download_savefile)
            who_filename = None
            for fileinfo in who_archive.infolist():
                if self.filetype(filename = fileinfo.filename) in ['CSV','XLSX']:
                    who_archive.extract(fileinfo)
                    who_filename = fileinfo.filename
            if not who_filename is None:
                os.rename(who_filename, self.__datasourceinfo['WHOPHSM']['DatafileName'])
            else:
                raise IOError('did not find appropriate files in ZIP archive')

    
    def convertDate(self, datestr, inputformat = None, outputformat = None):
        if inputformat is None: inputformat = self.__datasourceinfo[self.__datasource]['dateformat']
        if outputformat is None: outputformat = self.__dateformat
        if isinstance(datestr, (list, tuple, np.ndarray, pd.Series)):
                          return [self.convertDate(x, inputformat = inputformat, outputformat = outputformat) for x in datestr]
        return datetime.datetime.strptime(str(datestr),inputformat).strftime(outputformat)


    def ReadData(self):
        def CleanWHOName(name):
            return name.replace('nan -- ','').replace(' -- nan','')
        
        if not os.path.exists(self.__datasourceinfo[self.__datasource]['DatafileName']) or self.__downloaddata:
            self.DownloadData()
        
        if self.filetype(datasource = self.__datasource) == 'CSV':
            readdata       = pd.read_csv(self.__datasourceinfo[self.__datasource]['DatafileName'],**self.__datasourceinfo[self.__datasource]['DatafileReadOptions'])
        elif self.filetype(datasource = self.__datasource) == 'XLSX':
            readdata       = pd.read_excel(self.__datasourceinfo[self.__datasource]['DatafileName'], **self.__datasourceinfo[self.__datasource]['DatafileReadOptions'])
        else:
            raise NotImplementedError
        
        # set a preliminary countrylist, is updated again at the end
        self.__countrylist = list(readdata[self.__countrycolumn].unique())
        
        
        # individual loading code for the different databases
        # internal structure of the data: data.columns = [self.__countrycolumn, 'Date', 'Measure_L1', 'Measure_L2', ... ]
        
        if self.__datasource == 'CSH':
            # store CSV directly as data
            self.__data    = readdata.copy(deep = True)
            self.__data['Date'] = self.__data['Date'].apply(self.convertDate)
            for i in range(self.__datasourceinfo['CSH']['MaxMeasureLevel']):
                self.__data['Measure_L{:d}'.format(i+1)] = self.__data['Measure_L{:d}'.format(i+1)].str.strip()
            
            # treat each US state as individual country
            if self.__resolve_US_states:
                # first, rename states to 'US - STATENAME'
                self.__data[self.__countrycolumn] = np.where(self.__data[self.__countrycolumn] == self.__USname, 'US - ' + self.__data['State'], self.__data[self.__countrycolumn])
                
                nationwide_id = 'US - United States of America'
                us_states = list(self.__data[self.__data[self.__countrycolumn].str.startswith('US - ')][self.__countrycolumn].unique())
                us_states.remove(nationwide_id)
                
                # copy all nationwide measures for all states
                for us_state in us_states:
                    for index, measure_item in self.__data[self.__data[self.__countrycolumn] == nationwide_id].iterrows():
                        self.__data = self.__data.append(measure_item.replace({self.__countrycolumn:nationwide_id}, value = us_state), ignore_index = True)
                
                # remove nationwide measures
                self.__data.drop(self.__data[self.__data[self.__countrycolumn] == nationwide_id].index, inplace = True)
                
    
        elif self.__datasource == 'OXFORD':
            # construct list of measures from DB column names
            # naming scheme is '[CEH][NUMBER]_NAME'
            # in addition to columns '[CEH][NUMBER]_IsGeneral' and '[CEH][NUMBER]_Notes' for more info
            measurecolumns = []
            for mc in readdata.columns:
                if not re.search('^[CEH]\d+\_',mc) is None:
                    if mc[-7:].lower() != 'general' and mc[-5:].lower() != 'notes' and mc[-4:].lower() != 'flag':
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
            self.__data['Date'] = self.__data['Date'].dt.strftime(self.__dateformat)
        
        
        elif self.__datasource == 'CORONANET':
            self.__data = readdata[[self.__countrycolumn, 'date_start', 'type', 'type_sub_cat', 'type_text']].copy(deep = True)
            self.__data.columns = [self.__countrycolumn,'Date', 'Measure_L1', 'Measure_L2', 'Measure_L3']
            self.__data['Date'] = self.__data['Date'].apply(self.convertDate)
            # general measures seem to have no L2 description, thus if empty, copy L1
            self.__data['Measure_L2'].fillna(self.__data['Measure_L1'], inplace = True)
            
                
        elif self.__datasource == 'HITCOVID':
            self.__data = readdata[[self.__countrycolumn, 'date_of_update', 'intervention_group', 'intervention_name']].copy(deep = True)
            self.__data.columns = [self.__countrycolumn, 'Date', 'Measure_L1', 'Measure_L2']
            self.__data.dropna(inplace = True)
            self.__data['Data'] = self.__data['Date'].apply(self.convertDate)
            
        
        else:
            NotImplementedError

        # update countrylist with potential changes during load
        self.__countrylist = list(self.__data[self.__countrycolumn].unique())

    
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
                return ''.join([mn.capitalize() for mn in measurename.replace('/','').replace(',','').replace('.','').replace('-',' ').replace('_',' ').replace('(',' ').replace(')',' ').split(' ')])
        return measurename



    def ImplementationTable(self, country, measure_level = None, startdate = '22/1/2020', enddate = None, shiftdays = 0, maxlen = None, clean_measurename = True, only_pulse = False, binary_output = False, extend_measure_names = False, mincount = None):
        if country in self.__countrylist:
            countrydata  = self.CountryData(country = country, measure_level = measure_level, only_first_dates = False, extend_measure_names = extend_measure_names)
            ret_imptable = pd.DataFrame( { self.CleanUpMeasureName(measurename, clean_up = clean_measurename):
                                           self.dates2vector(implemented, start = startdate, end = enddate, shiftdays = shiftdays, maxlen = maxlen, only_pulse = only_pulse, binary_output = binary_output)
                                           for measurename, implemented in countrydata.items() } )
            ret_imptable.index = [(datetime.datetime.strptime(startdate,'%d/%m/%Y') + datetime.timedelta(days = i)).strftime(self.__dateformat) for i in range(len(ret_imptable))]
            # check to only return Measures that are in measurelist (which funnels mincount)
            measurelist = self.MeasureList(measure_level = measure_level, mincount = mincount)
            ret_imptable = ret_imptable[ret_imptable.columns[ret_imptable.columns.isin(measurelist.index)]]
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
        measurenameDF = pd.DataFrame(self.__data[mheaders + [self.__countrycolumn,'Date']]).replace(np.nan, '', regex = True)
        measurenameDF.drop(measurenameDF[measurenameDF['Date'].apply(lambda x:datetime.datetime.strptime(x,self.__dateformat) > datetime.datetime.strptime(enddate,self.__dateformat))].index, inplace = True)
        measurenameDF.drop('Date', axis = 'columns', inplace = True)
        measurenameDF.drop_duplicates(inplace = True)
        if not countrylist is None: measurenameDF = measurenameDF[measurenameDF[self.__countrycolumn].isin(countrylist)]
        measurenameDF = measurenameDF.groupby(by = mheaders, as_index=False).count()
        measurenameDF.columns = mheaders + ['Countries with Implementation']
        measurenameDF.index = list(measurenameDF['Measure_L{:d}'.format(measure_level)].apply(self.CleanUpMeasureName))

        if not mincount is None: measurenameDF = measurenameDF[measurenameDF['Countries with Implementation'] >= mincount]

        return measurenameDF
    
    
    
    def FinalDates(self, countrylist = None):
        def LastDate(datelist):
            return self.SortDates(datelist)[-1]
        
        if countrylist is None: countrylist = self.__countrylist
        finaldatesDF = self.__data[[self.__countrycolumn,'Date']].groupby(by = self.__countrycolumn, as_index = False).agg({'Date':LastDate})
        return finaldatesDF[finaldatesDF[self.__countrycolumn].isin(countrylist)].set_index(self.__countrycolumn, drop = True)
    
    
    
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
        
    
