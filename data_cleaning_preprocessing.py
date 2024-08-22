# Python Libraries -- Data Manipualtion & Miscellneous
import pandas as pd
import numpy as np
import json

# Regular Expression & date
import re
from dateutil import parser

# Missing Values, Encodings, Ouliers Smoothning
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from scipy.stats.mstats import winsorize

# Module Import
import config

# Multithreading
import threading

"""Handle Values formatted AS [digits][alphabet] -- Example 15Kg,$15,'min 15' (High Risk Function)"""
def handle_value_formatted_as_digits_alphabet(value : object) -> pd.to_numeric:
        
    """Handle Values formatted AS [digits][alphabet] -- Example 15Kg,$15,'min 15' (High Risk to use it)

    This Function is used to fetch numeric value greedily form any object/str 

    Parameters
    ----------
    value : object

    Returns
    -------
    value : pandas.to_numeric
        possible int/float numeric value.
    """
    value =str(value)
    # Check if value starts with a number only and doesn't contain [digit][alphabet][digit]
    if re.match(r'^\d+', value) and not re.search(r'\d+(\.\d+)?[^\d\.]+\d+(\.\d+)?', value): 
            try:
                    value = re.findall(r'^[+-]?\d*\.?\d+',value)
                    return pd.to_numeric(value[0])
            except:
                    pass

    # check for '$15','min 15'
    if  re.match(r'[^\d\.]+\d+(\.\d+)?',value) and not re.search(r'[^\d\.]+\d+(\.\d+)?[^\d\.]+', value): 
            try:
                    value = re.search(r'\d+(\.\d+)?',value)
                    if value:
                            extracted_value = value.group()
                    return pd.to_numeric(extracted_value)
            except:
                    pass      
    return value



"""Timestamp Conversion"""
def convert_timestamp(ts : object) -> pd.Timestamp:
    """Timestamp convert

    object (different formats) to pandas Timestamp 

    Parameters
    ----------
    ts : object

    Returns
    -------
    value : datetime64[ns]
       
    """
    # Define the formats for each timestamp
    formats = ['%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d:%H:%M:%S.%f','%d-%m-%Y %H:%M','%m/%d/%Y %H:%M','%d-%m-%Y']
    for fmt in formats:
        try:
            ts = pd.to_datetime(ts, format=fmt)
        except Exception as e:
            pass
    return ts

"""Column Name Cleaning"""
def remove_special_chars(col : str) -> str:
    """Handle column names (removing special characters)

    This Function is used to remove special character from string value

    Parameters
    ----------
    col : str

    Returns
    -------
    string : str
        cleaned col name (removing special characters)
    """
    col = col.strip()  # Remove leading and trailing whitespaces
    col = col.strip('_')  # Remove leading and trailing underscores
    return ''.join(e for e in col if e.isalnum() or e in ['_', '(', ')']) # Keep alphanumeric characters and underscores

def col_name_cleaning(dataset : pd.DataFrame) -> pd.DataFrame:
    """Main col name cleaning function

    Uses remove_special_chars(col : str) -> str
    as well it removes extra whitespaces and convert it to lowercase

    Parameters
    ----------
    dataset : pandas.Dataframe

    Returns
    -------
    dataset : pandas.Dataframe
        cleaned col name (overall)
    """
    dataset.columns = dataset.columns.to_series().apply(lambda x: re.sub(r'\s+', ' ', x.strip()))
    dataset.columns = dataset.columns.str.replace(' ', '_').str.lower()
    dataset.columns = [remove_special_chars(col) for col in dataset.columns]
    dataset.columns = dataset.columns.to_series().apply(lambda x: re.sub(r'_+', '_', x))
    dataset.columns = dataset.columns.str.strip('_')
    return dataset

def dataset_cleaning_rest(dataset : pd.DataFrame) -> pd.DataFrame:
    """Rest data (Data-Points) basic cleaning

    Uses remove_special_chars(col : str) -> str
    as well it removes extra whitespaces and cspecial chracter form Data Values

    Parameters
    ----------
    dataset : pandas.Dataframe

    Returns
    -------
    dataset : pandas.Dataframe
        cleaned Data (overall)
    """
    # Remove leading and trailing white spaces from all string columns
    dataset = dataset.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    # Remove special characters from string values
    dataset[dataset.select_dtypes(include='object').columns] = dataset[dataset.select_dtypes(include='object').columns].apply(lambda x: x.apply(lambda y: re.sub(r'[^a-zA-Z0-9. :/\\s-]', '', str(y)).strip().strip('/').strip('\\')))
    return dataset


"""Actual Column Names Exist or not ?"""
def count_of_non_numeric_and_numeric_columns(columns : list) -> tuple:
    """Actual Column Names Exist or not ? (I)

    Count numeric and non-numeric instance in colname list

    Parameters
    ----------
    columns : list

    Returns
    -------
    column_count_non_numeric,column_count_numeric,float_value_flag : tuple(int,int,bool)
        float_flag_value means either any col name is a float number is there exist
    """
    column_count_numeric = 0
    column_count_non_numeric = 0
    float_value_flag = False
    for column in columns.to_list():
        try:
           if (bool(pd.to_numeric(str(column)))):
               column_count_numeric += 1
               if str((pd.to_numeric(str(column))).dtype) == "float64" or str((pd.to_numeric(str(column))).dtype) == "float32":
                   column_count_non_numeric = 0
                   column_count_numeric = 0
                   float_value_flag = True
                   return column_count_non_numeric,column_count_numeric,float_value_flag
        except Exception as e:
            column_count_non_numeric += 1
    return column_count_non_numeric,column_count_numeric,float_value_flag

def check_for_actual_col_names(dataset : pd.DataFrame) -> bool:
    """Actual Column Names Exist or not ? (II)

    Applied Logic to check whether col names are actual or not

    Parameters
    ----------
    dataset : pandas.DataFrame

    Returns
    -------
    bool : bool
        True or False will be returned based on logic outcome
    """
    column_count_numeric = 0
    column_count_non_numeric = 0
    float_value_flag = False
    zeroth_float_value_flag = False

    column_count_non_numeric,column_count_numeric,float_value_flag = count_of_non_numeric_and_numeric_columns(dataset.columns)
    if float_value_flag == True:
        return False

    # matching with first datapoint without nan consideration
    column_count_non_numeric,column_count_numeric,float_value_flag = count_of_non_numeric_and_numeric_columns(dataset.iloc[0].dropna().index)
    zeroth_data_point_column_count_non_numeric,zeroth_data_point_column_count_numeric,zeroth_float_value_flag = count_of_non_numeric_and_numeric_columns(dataset.iloc[0].dropna())

    if(column_count_non_numeric == zeroth_data_point_column_count_non_numeric and column_count_numeric == zeroth_data_point_column_count_numeric):
        return False
    else:
        return True

"""Check for TS Col Name Fetching"""
def check_for_timestamp_column_name_and_dtypes_handling(dataset : pd.DataFrame) -> pd.DataFrame:
    """check_for_timestamp_column_name_and_dtypes_handling

    As there is a chance that cols won't have proper dtypes ; various other scenarios are considered 
    To achieve : expose all dtypes of every cols before label encoding stage 
    merging of 'date' & 'time' -> to individual col to one col

    Parameters
    ----------
    dataset : pandas.DataFrame

    Returns
    -------
    dataset : pandas.DataFrame
        proper defined dtypes for each column
    """
   
    dataset_object_col_copy = pd.DataFrame()

    # implement detection 1  --- Not to apply List Comprehension
    # check whether which object column belongs to numeric & which to string ('digital state' handling)
    string_values_indexes = []
    for column in dataset.select_dtypes(include='object').columns:
        temp_df = pd.to_numeric(dataset[column], errors='coerce').notnull()
        string_values_indexes = temp_df[temp_df == False].index
        if len(string_values_indexes)> 0 and len(string_values_indexes) < 0.4 * len(dataset[column]):
            for index in string_values_indexes:
                dataset.at[index, column] = np.nan
    last_ts_column_check_run = True

    # column type conversion

    for column in dataset.select_dtypes(include='object').columns:
        # below one line -- bad use/ taking advantage of loop
        dataset_object_col_copy[column] = dataset[column]
        if(pd.to_numeric(dataset[column].dropna(), errors='coerce').notnull().all()):
            dataset[column] = pd.to_numeric(dataset[column],errors='coerce')


    # iterate through each column and try to convert object columns to datetime
    for column in dataset.select_dtypes(include='object').columns:
        try:
            #if(pd.to_datetime(dataset[col].dropna()).notnull().all()):
            dataset[column] = pd.to_datetime(dataset[column])
            config.ts_col_name.append(column)
            last_ts_column_check_run = False
            #print(ts_col_name)

        except Exception as e:
            #print(str(e))
            pass

    # last check for other format (for safety)
    if(last_ts_column_check_run):
        for column in dataset.select_dtypes(include='object').columns:
            try:
                dataset[column] = dataset[column].apply(convert_timestamp)
                dataset[column] = pd.to_datetime(dataset[column])
                config.ts_col_name.append(column)

            except Exception as e:
                #print(str(e))
                pass

    if (len(config.ts_col_name) == 2):
        config.ts_col_name.append('timestamp_merged_manual')
        try:
            dataset['timestamp_merged_manual'] = dataset_object_col_copy[config.ts_col_name[0]] + " " + dataset_object_col_copy[config.ts_col_name[1]]
            dataset['timestamp_merged_manual'] = pd.to_datetime(dataset['timestamp_merged_manual'])
        except:
            print("\nunable to convert datasettype to datetime64[ns] standard format of column  name (manual merged by enviornment):\ntimestamp_merged_manual")
            try:
                dataset = dataset.drop(columns=['timestamp_merged_manual'])
                config.ts_col_name.remove('timestamp_merged_manual')
            except:
                pass

    # for column in dataset.select_dtypes(include='object').columns:
    #     try:
    #         dataset[column] = dataset[column].apply(handle_value_formatted_as_digits_alphabet)
    #         if(pd.to_numeric(dataset[column].dropna(), errors='coerce').notnull().all()):
    #             dataset[column] = pd.to_numeric(dataset[column],errors='coerce')
    #     except Exception as e:
    #         pass
    return dataset

"""NaN Format Check"""
def nan_format_check(dataset : pd.DataFrame) -> pd.DataFrame:
    """Nan Format check

    Generalize np.NaN -> wherever missing value present

    Parameters
    ----------
    dataset : pandas.DataFrame

    Returns
    -------
    dataset : pandas.DataFrame
        
    """
    dataset.loc[:, dataset.select_dtypes(include='object').columns] = dataset.loc[:, dataset.select_dtypes(include='object').columns].applymap(lambda x: np.NaN if re.match(r'^(nan|Nan|NAN|NaN|nAn|NaT|NAT)', str(x)) else x)
    return dataset

"""Const Column Detection & Removal"""
def constant_column_remove(dataset : pd.DataFrame) -> pd.DataFrame:
    """Const Column Detection & Removal

    Constant valued cols will be removed at this stage

    Parameters
    ----------
    dataset : pandas.DataFrame

    Returns
    -------
    dataset : pandas.DataFrame
        
    """
    for column in dataset.columns.to_list():
        if(dataset[column].nunique() < 2):
            dataset = dataset.drop(columns=[column])
    return dataset


"""Negative Value Handling"""
def negative_value_exposing(dataset : pd.DataFrame) -> bool:
    """Negative Value Handling : expose

    Expose Negative values present or not ?

    Parameters
    ----------
    dataset : pandas.DataFrame

    Returns
    -------
    bool : bool

        Negative values are present or not with updation of values(-ve valued Col names) in JSON 
        
    """
    columns_with_negatives_values= []
    if(len(config.ts_col_name) > 0):
        columns_to_be_consider = [col for col in dataset.columns if col != config.ts_col_name[-1]]
    else:
        columns_to_be_consider = dataset.columns
    for col in columns_to_be_consider:
        if dataset[col].lt(0).any():  # Check if any value in the column is negative
            columns_with_negatives_values.append(col) 

    try:
        columns_with_negatives_values.remove(config.target_column)
    except:
        pass
    
    if(len(columns_with_negatives_values) > 0):
        config.preprocess_rule_api["data_preprocessing"]["negative value detection bool"] = bool(True)
        config.preprocess_rule_api["data_preprocessing"]["negative value columns names"] = columns_with_negatives_values
        config.json.dump(config.preprocess_rule_api, open('rules_modular_approach.json', 'w'))
        return True
    else:
        config.preprocess_rule_api["data_preprocessing"]["negative value columns names"] = []
        config.preprocess_rule_api["data_preprocessing"]["negative value detection bool"] = bool(False)
        json.dump(config.preprocess_rule_api, open('rules_modular_approach.json', 'w'))

    return False
   
def negative_value_handling(dataset : pd.DataFrame, method : str) -> pd.DataFrame:
    """Negative Value Handling : handling

    Parameters
    ----------
    dataset : pandas.DataFrame
    method  : str

    Returns
    -------
    dataset : pandas.DataFrame
        Handled negative value with specified method
        
    """
    if(config.preprocess_rule_api["data_preprocessing"]['negative value detection bool'] == True and
            len(config.preprocess_rule_api["data_preprocessing"]["negative value columns to be handled selected by user"]) > 0  ):
            
            if (method =="ABS"):
                for col in config.preprocess_rule_api["data_preprocessing"]["negative value columns to be handled selected by user"]:
                    dataset[col] =  dataset[col].apply(abs)
            elif method == "IGNORE":
                for col in config.preprocess_rule_api["data_preprocessing"]["negative value columns to be handled selected by user"]:
                     dataset = dataset[~dataset[col].lt(0)]
            else:
                pass
    
    return dataset


"""Outlier Handling"""
def outlier_expose(dataset : pd.DataFrame) -> bool:
    """Outlier Handling : expose

    Expose Outliers are present or not ?

    Parameters
    ----------
    dataset : pandas.DataFrame

    Returns
    -------
    bool : bool
        outliers are present or not with updation of values(True/False) in JSON 
        
    """
    config.preprocess_rule_api["data_preprocessing"]["outlier detection bool"] = bool(False)
    outliers_col_count = 0 
    if(len(config.ts_col_name) > 0):
        columns_to_be_consider = [col for col in dataset.columns if col != config.ts_col_name[-1]]
    else:
        columns_to_be_consider = dataset.columns
    for col in columns_to_be_consider:
        q1 = dataset[col].quantile(0.25)
        q3 = dataset[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
   
        outliers = dataset[(dataset[col] < lower_bound) | (dataset[col] > upper_bound)]
        if not outliers.empty:
            outliers_col_count += 1
           
    if(outliers_col_count > 0):
        config.preprocess_rule_api["data_preprocessing"]["outlier detection bool"] = bool(True)
        json.dump(config.preprocess_rule_api, open('rules_modular_approach.json', 'w'))
        return True
    else:
        config.preprocess_rule_api["data_preprocessing"]["outlier detection bool"] = bool(False)
        json.dump(config.preprocess_rule_api, open('rules_modular_approach.json', 'w'))
    return False
    
def outlier_handling(dataset : pd.DataFrame) -> pd.DataFrame:
    """Outlier Handling : handling

    Parameters
    ----------
    dataset : pandas.DataFrame
 
    Returns
    -------
    dataset : pandas.DataFrame
        Handled outliers using Winsorise[0.05,0.05]
        
    """
    if(config.preprocess_rule_api["data_preprocessing"]["outlier detection bool"]== True and
        config.preprocess_rule_api["data_preprocessing"]['outlier handling bool selected by user']== bool(True)):
        # Winsorize the dataset, replacing the 5% lowest and 5% highest values
        if(len(config.ts_col_name) > 0):
            columns_to_be_consider = [col for col in dataset.columns if col != config.ts_col_name[-1]]
        else:
            columns_to_be_consider = dataset.columns
        for col in columns_to_be_consider:
            dataset[col] = winsorize(dataset[col], limits=[0.05, 0.05])
    return dataset


"""Replacing  Missing Values --Imported from Radhika's Working File"""
def replace_missing_values(dataset : pd.DataFrame) -> pd.DataFrame:
    """Replace Missing Values

    Operations: 
    LabelEncoding ofr categorical cols
    missing values
        categorical : mode
        numeric : median
    
    Parameters
    ----------
    dataset : pandas.DataFrame
 
    Returns
    -------
    dataset : pandas.DataFrame
        Label Encoded data with Handled missing values 
    """
    # Replace missing values in categorical columns with the most common class
    for col in dataset.select_dtypes(include=['object']).columns:
       dataset[col].fillna(dataset[col].mode()[0], inplace=True)

    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Encode features
    for feature in dataset.select_dtypes(include=['object']).columns:
       dataset[feature] = label_encoder.fit_transform(dataset[feature])

    # Replace missing values in numerical columns with the median
    imputer = SimpleImputer(strategy='median')
    numerical_cols = dataset.select_dtypes(include=['float64', 'int64']).columns
    dataset[numerical_cols] = imputer.fit_transform(dataset[numerical_cols])
    return dataset


"""Duplicate Values Handling"""
def duplicates_finder(dataset : pd.DataFrame) -> list:
    """Duplicate Finder

    Parameters
    ----------
    dataset : pandas.DataFrame
 
    Returns
    -------
    duplicates : list
        list of duplicate datapoints
        
    """
    # Find duplicate timestamps
    #global ts_col_name
    duplicates = dataset[dataset.duplicated(subset=[config.ts_col_name[-1]], keep=False)]
    return duplicates

def duplicate_value_based_on_timestamps_handling(dataset : pd.DataFrame, method : str) -> pd.DataFrame:
    #global ts_col_name
    """Duplicate Handling : handling

    Parameters
    ----------
    dataset : pandas.DataFrame
    method  : str
 
    Returns
    -------
    dataset : pandas.DataFrame
        Handled duplicate points with specified method
        
    """
    duplicates = duplicates_finder(dataset=dataset)
    if(len(config.ts_col_name)>0 and len(duplicates)>1):
        
        if(method == "LATEST"):
            #print("We have found Duplicates & Handled considering Latest Values")
            # Iterate over duplicate timestamps
            for ts in duplicates[config.ts_col_name[-1]].unique():
                duplicate_rows = dataset[dataset[config.ts_col_name[-1]] == ts]
                # Check if values in duplicate rows are the same
                if duplicate_rows.iloc[:-1].equals(duplicate_rows.iloc[1:]):
                    # If values are the same, keep only one row
                    dataset = dataset.drop(duplicate_rows.index[:-1])
                else:
                    # If values are different, keep the last row
                    if not duplicate_rows.iloc[:-1].equals(duplicate_rows.iloc[1:]):
                        dataset = dataset.drop(duplicate_rows.index[:-1])

        if(method == "EARLIEST"):
            #print("We have found Duplicates & Handled considering Earliest Values")
            # Iterate over duplicate timestamps
            for ts in duplicates[config.ts_col_name[-1]].unique():
                duplicate_rows = dataset[dataset[config.ts_col_name[-1]] == ts]
                # Check if values in duplicate rows are the same
                if duplicate_rows.iloc[:-1].equals(duplicate_rows.iloc[1:]):
                    # If values are the same, keep only one row
                    dataset = dataset.drop(duplicate_rows.index[:-1])
                else:
                    # If values are different, keep the last row
                    if not duplicate_rows.iloc[:-1].equals(duplicate_rows.iloc[1:]):
                        dataset = dataset.drop(duplicate_rows.index[1:])

    # set Index as timestamp --> TS col not considered as Feature in Pycaret
    if(len(config.ts_col_name)>0):
        dataset = dataset.set_index(config.ts_col_name[-1])
       
    return dataset


def duplicate_values_expose(dataset : pd.DataFrame) -> bool:

    #global preprocess_rule_api,ts_col_name
    """Duplicates Handling : expose

    Parameters
    ----------
    dataset : pandas.DataFrame
 
    Returns
    -------
    bool : bool
        duplicates are present or not with updation of values(True/False) in JSON 
        
    """

    # Non Timeserires Data 
    if(len(config.ts_col_name)==0):
        config.preprocess_rule_api["data_preprocessing"]["duplicate value detection bool"] = bool(False)
        json.dump(config.preprocess_rule_api,open(("rules_modular_approach.json"),'w'),indent=4)
        dataset = dataset.drop_duplicates()
        print("\nNon Timeseries Data -> Dropped Duplicates directly if exists !!!!!\n")   
        return False
    
    # Find duplicate timestamps
    duplicates = duplicates_finder(dataset)
    

    #No Duplicacy Found
    if(len(duplicates) == 0):
        print("No Duplicates Found")
        return False
    
    # duplicate dataset Handling --------Works on Timeseries only as for now 
    if(len(duplicates)>0):
        config.preprocess_rule_api["data_preprocessing"]["duplicate value detection bool"] = bool(True)
        json.dump(config.preprocess_rule_api,open(("rules_modular_approach.json"),'w'),indent=4)
        return True
     
"""New Upadates on Duplicate Handling"""
def batches_creator(dataset : pd.DataFrame,no_of_batches : int) -> list:
    """Batches Creaator

    Parameters
    ----------
    dataset : pandas.DataFrame
    no_of_batches : int
 
    Returns
    -------
    batches : List
        Consists all batches of pandas DataFrame
        
    """
    dataset_size = len(dataset)
    # Integer division to get the batch size
    batch_size = dataset_size // no_of_batches  
    # Split the dataset into 10 batches
    batches = []
    for i in range(no_of_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        if i == no_of_batches - 1 :
            # Handle the last batch, which may have a different size
            end = dataset_size
        batches.append(dataset.iloc[start:end])
    return batches

def main_runnable_duplicate_handling_on_timestamp(dataset : pd.DataFrame,no_of_batches : int,method : str) -> pd.DataFrame:
   """main_runnable_duplicate_handling_on_timestamp

    Parameters
    ----------
    dataset : pandas.DataFrame
    no_of_batches : int
    method : str
 
    Returns
    -------
    dataset : pandas.DataFrame
        Main Governing function to handle duplicates
       
        
    """
   duplicate_values_expose_bool = duplicate_values_expose(dataset)
   if duplicate_values_expose_bool == True and dataset.shape[0] > 300000:
      print(dataset.shape)
      print("Process 0 Started")
      while(duplicate_values_expose_bool != False):
         config.reduced_batches = []
         current_shape = dataset.shape[0]
         batches = batches_creator(dataset=dataset,no_of_batches=no_of_batches)
         dataset  = remove_duplicates_multithreading(batches,no_of_batches,method=method)
         print(dataset.shape)
         next_shape = dataset.shape[0]
         duplicate_values_expose_bool = duplicate_values_expose(dataset=dataset)
         if(dataset.shape[0] < 300000):
             duplicate_values_expose_bool = False
             dataset = duplicate_value_based_on_timestamps_handling(dataset= dataset, method=method)
             print("Normal Process Opted as Datapoints < 300000")

         else:
            # Ask Radhika for THis Condition (Priority : High)
            if(current_shape!=next_shape):
                print("+1 Process Added")
            else: 
                ("Come across same shape -> Critical Situation")
                dataset = duplicate_value_based_on_timestamps_handling(dataset= dataset, method=method)
   else:
        dataset = duplicate_value_based_on_timestamps_handling(dataset= dataset, method=method)
   # set Index as timestamp --> TS col not considered as Feature in Pycaret
   if(len(config.ts_col_name)>0):
      dataset = dataset.set_index(config.ts_col_name[-1])
   return dataset

def process_batch_for_duplicate_value_handling_on_timestamp(batch_no : int,batches : list,method : str) -> None:
    """process_batch_for_duplicate_value_handling_on_timestamp

    Parameters
    ----------
    batch_no : int
    batches : list
    method : str
 
    Returns
    -------
    None
        Applies Duplicate Handling with particular method for incoming batch data
        
    """
    batch_data = batches[batch_no]
    if(batch_data[config.ts_col_name[-1]].duplicated().sum() > 0):
        duplicate_free_batch = duplicate_value_based_on_timestamps_handling(dataset=batch_data, method=method)
    else:
        duplicate_free_batch = batch_data
    config.reduced_batches.append(duplicate_free_batch)
   

def remove_duplicates_multithreading(batches : list,no_of_batches : int,method : str) -> pd.DataFrame:
    """remove_duplicates_multithreading

    Parameters
    ----------
    batches : list
    no_of_batches : int
    method : str
 
    Returns
    -------
    reduced_batches : pandas.DataFrame
         Calls different different Function to run batches on threads to faster the process of duplicate handling
         returns a joined Dataframe which available in reduce_batches list which is duplicates free
        
    """
    threads = []

    for i in range(no_of_batches):
        t = threading.Thread(target=process_batch_for_duplicate_value_handling_on_timestamp, args=(i,batches,method))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    config.reduced_batches = pd.concat(config.reduced_batches,ignore_index=False).sort_values(by=config.ts_col_name[-1])
    return config.reduced_batches







'''
"""Duplicate Data Handling --Imported From Aman Working File"""
def handle_duplicate_timestamps(dataset):

    # Check for duplicacy in column names
    if dataset.columns.duplicated().any():
        raise ValueError("Duplicate column names found in the dataset.")

    if(len(config.ts_col_name)>0):
        # Find duplicate timestamps
        duplicates = dataset[dataset.duplicated(subset=[config.ts_col_name[-1]], keep=False)]
        if(len(duplicates)>1):
            print("We have found Duplicates & Handled considering Latest Values")
        # Iterate over duplicate timestamps
        for ts in duplicates[config.ts_col_name[-1]].unique():
            duplicate_rows = dataset[dataset[config.ts_col_name[-1]] == ts]
            # Check if values in duplicate rows are the same
            if duplicate_rows.iloc[:-1].equals(duplicate_rows.iloc[1:]):
                # If values are the same, keep only one row
                dataset = dataset.drop(duplicate_rows.index[:-1])
            else:
                # If values are different, keep the last row
                if not duplicate_rows.iloc[:-1].equals(duplicate_rows.iloc[1:]):
                    dataset = dataset.drop(duplicate_rows.index[:-1])

    elif(len(config.ts_col_name)==0):
        dataset = dataset.drop_duplicates()
    else:
        pass
    return dataset
'''

'''
def replace_and_duplicate_handling(dataset):

    
    # Timeseries dataset is there or not ?
    if(len(config.ts_col_name)>0):
        dataset = dataset.sort_values(by=config.ts_col_name[-1])

    # missing values handling
    dataset = replace_missing_values(dataset)

    # duplicate dataset Handling --------Works on Timeseries only as for now
    dataset = handle_duplicate_timestamps(dataset)

    # set Index as timestamp --> TS col not considered as Feature in Pycaret
    if(len(config.ts_col_name)>0):
        dataset = dataset.set_index(config.ts_col_name[-1])

    # Winsorize the dataset, replacing the 5% lowest and 5% highest values
    for col in dataset.columns:
        dataset[col] = winsorize(dataset[col], limits=[0.05, 0.05])
    return dataset
'''