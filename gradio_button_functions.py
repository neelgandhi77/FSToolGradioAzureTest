# # Python Libraries 
# import pandas as pd
# import json

# # Modules Import
# import config
# import data_cleaning_preprocessing as dcp
# import target_column_fetch as tc
# import pycaret_process_with_SHAP as pycaret

# # UI Libraries
# import gradio as gr
# from gradio.themes.base import Base

# # Visualization
# import matplotlib.pyplot as plt

# # statsmodel
# from statsmodels.tsa.stattools import adfuller
# import statsmodels.api as sm

# # Monitor CPU And Cores
# import psutil
# import time

# def monitor_cpu_usage(count,interval=1):
#    # print(f"Check time : {count}")
#     # Get the CPU usage percentage
#     cpu_usage = psutil.cpu_percent(interval=interval)
#     # Get the number of logical CPUs
#     logical_cpus = psutil.cpu_count()
#     # Get the number of physical cores
#     physical_cores = psutil.cpu_count(logical=False)

#     #print(f"CPU Usage: {cpu_usage}%")
#     print(f"Logical CPUs: {logical_cpus}, Physical Cores: {physical_cores}")
#     time.sleep(interval)


# """Dataset Stationary & Non-Stationary Check"""
# def adfuller_test(dataset : pd.DataFrame) -> tuple:
#     """Dataset Stationary & Non-Stationary Check and Add lags to a timeseries on requirement(I)

#         ADF Test will be carried out ; outcome-> Sationary or Non Stationary data is present for TS DATA? 

#         Parameters
#         ----------
#         dataset : pandas.DataFrame

#         Returns
#         -------
#         result,bool : tuple(list,bool)

#             list : ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
#             False: It is non stationary
#             True: It is stationary
#         """
   
#     result= adfuller(dataset[config.target_column])
#     labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
#     print("ADF Test")
#     for value,label in zip(result,labels):
#         print(label+' : '+str(value) )
    
#     #Ho: It is non stationary
#     #H1: It is stationary
#     if result[1] <= 0.05:
#         print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
#         return [],True
#     else:
#         print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
#         return result,False

# def no_of_lags_in_timeseries_handling(dataset : pd.DataFrame) -> None:

#     """Dataset Stationary & Non-Stationary Check and Add lags to a timeseries on requirement (II)

#     Will automatically detect lags &  add a lags to all input : new lag_feature will be added
#     its using helper function adfuller_test(dataset : pd.DataFrame) -> tuple -> which actually 
#     takes ADF Test and finds Lag is required or not for Target Col.

#     Parameters
#     ----------
#     dataset : pandas.DataFrame

#     Returns
#     -------
#     dataset : pandas.DataFrame
#         dataset with lags added for target columns.
#     """
#     if(len(config.ts_col_name) > 0):
#         try:
#             result,ADF_bool = adfuller_test(dataset=dataset)
#             if(ADF_bool == False):
#                 fig = sm.graphics.tsa.plot_pacf(dataset[config.target_column],lags=60)
#                 # apply lag
#                 #dataset['lag_feature'] = dataset[target_column].shift(result[2])
#                 print("Note : Lag needed : {lag_value}; but not applied in this process".format(lag_value=result[2]))
#             else:
#                 print("No Need to add lags as data is stationary")
                
            
#             # Remove datapoints -> initial lag value count 
#             #dataset = dataset[result[2]:]
#             #return dataset
#         except:
#             if(len(config.ts_col_name) > 0):
#                 print("Classification Problem")


# # Cleaning After Reading CSV
# def read_file(file_path : str) -> pd.DataFrame:
#     """read_file

#     Operations: 
#     col name handling logic, col name cleaning logic, 'id' cols to be dropped
#     all Data Values Cleaning Task, Actual Col name exist or not check ?
#     NaN format generalization , Const valued Col removal
    
#     Parameters
#     ----------
#     file_path : str
 
#     Returns
#     -------
#     dataset : pandas.DataFrame
#         cleaned and preproceed data (left to handle missing value)
#     """

#     #Attempt to read Input CSV File
    
#     config.ts_col_name = []

#     # Read File
#     try:
#         dataset = pd.read_csv(file_path)
        
#     except Exception as e:
#         print("Unable to read File")
#         print(str(e))
#         return 0
    
#     # Not considering Col name contains 'id'
#     dataset = dataset.loc[:, ~dataset.columns.str.contains('id', case=False)]

#     # handling '' or ' ' in col names
#     temp_col_counter = 1
   
#     dataset_col_name_copy = dataset.columns.to_list().copy()
#     for col_index in range(len(dataset_col_name_copy)):
#         if dataset_col_name_copy[col_index] == '' or dataset_col_name_copy[col_index] == ' ':
#             temp_name = f"Temp_Name_{temp_col_counter}"
#             dataset_col_name_copy[col_index] = temp_name
#             temp_col_counter += 1
    
#     if(temp_col_counter > 1):
#         dataset.columns = dataset_col_name_copy

#     header_bool = dcp.check_for_actual_col_names(dataset)
    
#     if (header_bool == False):
#         dataset.loc[-1] = dataset_col_name_copy
#         dataset.index = dataset.index + 1
#         dataset = dataset.sort_index()
#         dataset.columns = [str(f"Column_Name_{i}") for i in range(1, len(dataset.columns)+1)]
#         dataset_col_name_copy = dataset.columns
    
#     dataset = dcp.dataset_cleaning_rest(dataset)
   
#     dataset = dcp.check_for_timestamp_column_name_and_dtypes_handling(dataset)
    
#     dataset = dcp.nan_format_check(dataset)
#     dataset = dcp.constant_column_remove(dataset)
    
#     print("\nActual Column Names Exist ? ")
#     if(header_bool == False):
#         print("No")
#     else:
#         print("Yes")  
        
#     print("\nTimeseries Dataset ? ")
#     if(len(config.ts_col_name) == 0):
#         print("No")
#     else:
#         print("Yes")
#         print("\nTS Col Name")
#         print(config.ts_col_name)

#     return dataset  


# # Functions
# """SE Image"""
# def static_image_path() -> str:
#     try:
#         return "se.png"
#     except Exception as e:
#         print(str(e))
#         #gr.Warning("Unable to Fetch SE Logo")

# """Called when Recomended Features Image won't exist"""
# def disable_shap_plot() -> str:
#     try:
#         return "images\sampleplotimage.png"
#     except Exception as e:
#         print(str(e))
#         gr.Warning("Unable to Fetch SE Logo")

# """Back to Default Json"""      
# def reset_json_api() -> None:
    
#     """Reset JSON API

#     Default setting
    
#     Parameters
#     ----------
#     None
 
#     Returns
#     -------
#     None"""
   
#     config.preprocess_rule_api["data_preprocessing"]["negative value detection bool"] = bool(False)
#     config.preprocess_rule_api["data_preprocessing"]["negative value columns names"] = ""
#     config.preprocess_rule_api["data_preprocessing"]['negative value columns to be handled selected by user'] = ""
#     config.preprocess_rule_api["data_preprocessing"]["negative value handling method selected by user"] = "KEEP AS IT IS"
#     config.preprocess_rule_api["data_preprocessing"]["outlier detection bool"] = bool(False)
#     config.preprocess_rule_api["data_preprocessing"]["outlier handling bool selected by user"] = bool(False)
#     config.preprocess_rule_api["data_preprocessing"]["duplicate value detection bool"] = bool(False)
#     config.preprocess_rule_api["data_preprocessing"]["duplicate value handling method selected by user"] = "LATEST"
   
#     # Dump Values
#     json.dump(config.preprocess_rule_api,open("rules_modular_approach.json",'w'),indent=4)
     
# """Upload Button Tasks"""
# def load_dataset(file : str) -> tuple:
#     # Resetting
#     reset_json_api()
#     try: 
#         if file.name.endswith('.csv'):  
#             progress =gr.Progress()
#             progress(0, desc="Working...")
#             # Apply your custom data cleaning functions here
#             dataset = read_file(file)
#             progress(0.5)
             
#             try:
#                 # missing values handling
#                 dataset = dcp.replace_missing_values(dataset)
#                 # duplicates Exposed
#                 duplicate_values_expose_bool = dcp.duplicate_values_expose(dataset)
#                 # Prompt User - Duplicates
#                 if(duplicate_values_expose_bool == True):
#                     gr.Info("We have Found Duplicates in Timestmap column; kindly Select Method to Handle")
#                 # Temp Conversion
#                 if(len(config.ts_col_name) > 0):
#                     dataset[config.ts_col_name[-1]] = dataset[config.ts_col_name[-1]].astype('str')
#                     # index manually added -> before Pycaret process -> dropped
#                     #dataset['current_index_col'] = dataset[ts_col_name[-1]].astype('str')
#                 else:
#                     #dataset['current_index_col'] = dataset.index.astype('str')
#                     pass     
#             except Exception as e:
#                 return gr.Warning("Unexpected Error While Detecting Missing & Duplicate Data : "  + str(e))
           
        
#             # Move the index column to the first position
#             #column_order = ['current_index_col'] + [col for col in dataset.columns if col != 'current_index_col']
#             #dataset = dataset.reset_index()[column_order]

#             # User Selection Checkbox Value
#             config.column_names_dropdown =[]
#             for col in dataset.columns:
#                 if(len(config.ts_col_name) > 0):
#                     if col != config.ts_col_name[-1]:
#                         config.column_names_dropdown.append(col)
#                     else:
#                         pass
#                 else:
#                     config.column_names_dropdown.append(col)

#             # Reset Global Value
#             config.target_column = ""
#             progress(1)
#             #return gr.Dataframe(dataset,visible=True),gr.Dropdown(column_names_dropdown,label="Select Target",visible=True),gr.Button("Trigger",variant='primary',visible=True),disable_shap_plot()
#             return gr.Dataframe(dataset.head(5),visible=True),gr.Dataframe(dataset,visible=False),gr.Dropdown(config.column_names_dropdown,label="Select Target",
#                                 visible=True),gr.Button(visible=True),gr.Plot(visible=False),gr.Textbox(
#                                 value="No Features to Display",visible=True),gr.Button(visible=False),gr.Dropdown(['ABS','IGNORE','KEEP AS IT IS'],value="<<--SELECT-->> (log : Negative Value Handle Method Name)",
#                                                                                                                   visible=False),gr.CheckboxGroup(
#                                 [],visible=False),gr.Dropdown(visible=False),gr.Dropdown(['EARLIEST','LATEST'],
#                                 visible=duplicate_values_expose_bool)
#         else:
#             # If the file is not a CSV, raise an alert message
#             gr.Warning('Please upload a CSV file.')
#     except Exception as e:
#         print(str(e))
#         gr.Warning("Upload Dataset / Something went wrong : " + str(e))


# """Apply Button First"""

# def select_target_column_pycaret_process_normal(dataset : pd.DataFrame,received_target : str,duplicate_handle_method : str) -> tuple:
#     progress =gr.Progress()
#     progress(0, desc="Working...")
#     try:
#         # Apply your custom functions using the target column
        
#         # Updating Target Col name Value       
#         config.target_column = received_target

#         # set Index as timestamp --> TS col not considered as Feature in Pycaret
#         if(len(config.ts_col_name)>0):
#             # change the type of timestamp col
#             dataset[config.ts_col_name[-1]] = dataset[config.ts_col_name[-1]].astype('datetime64[ns]')
#             dataset = dataset.set_index(config.ts_col_name[-1])
          
#         try:
#             #dataset = dataset.drop(columns = ['current_index_col'])
    
#             # Duplicate Values Handling with particular Method
#             if(duplicate_handle_method in ["EARLIEST","LATEST"] \
#                 and config.preprocess_rule_api["data_preprocessing"]["duplicate value detection bool"] == bool(True)):
#                 config.preprocess_rule_api["data_preprocessing"]["duplicate value handling method selected by user"] = duplicate_handle_method
   
#                 json.dump(config.preprocess_rule_api,open("rules_modular_approach.json",'w'),indent=4)
                
#                 if(duplicate_handle_method == "EARLIEST"):
#                     print("We have found Duplicates & Handled considering Earliest Values")
#                 elif(duplicate_handle_method == "LATEST"):
#                     print("We have found Duplicates & Handled considering Latest Values")
#                 else:
#                     pass
              
#                 # No of Thread/Batch = 10 as Default
#                 #monitor_cpu_usage(1)
#                 dataset = dcp.main_runnable_duplicate_handling_on_timestamp(dataset=dataset,no_of_batches=10,method=duplicate_handle_method)
#                 #monitor_cpu_usage(2)
#                 #dataset = dcp.duplicate_value_based_on_timestamps_handling(dataset=dataset,method=duplicate_handle_method)
        
#             else:
#                 #set Index as timestamp --> TS col not considered as Feature in Pycaret
#                 if(len(config.ts_col_name)>0):
#                     # change the type of timestamp col
#                     dataset[config.ts_col_name[-1]] = dataset[config.ts_col_name[-1]].astype('datetime64[ns]')
#                     dataset = dataset.set_index(config.ts_col_name[-1])
                
#         except:
#             pass
        
#         progress(0.7,desc="Please Wait... it takes time to process")
#         # Negative Values & Outlier Handling
#         negative_value_exposing_bool = dcp.negative_value_exposing(dataset)
#         outlier_expose_bool = dcp.outlier_expose(dataset)
#         progress(1)
#         if (negative_value_exposing_bool == False and outlier_expose_bool == False):
            
      
#             # ADF Test
#             no_of_lags_in_timeseries_handling(dataset)

#             dataset,config.best_model,X_test,y_test = pycaret.pycaret_dataset_feed_and_Initial_Col_name_mapping(dataset)
#             image_path = pycaret.Shap_plot_image(dataset,X_test,y_test)
#             progress(1)
#             return gr.Dataframe(dataset,visible=False),image_path,gr.Textbox(visible=False),gr.Button("Start Feature Selection",variant='primary',
#                             visible=False),gr.Dropdown(['ABS','IGNORE','KEEP AS IT IS'],value="KEEP AS IT IS",visible=False),gr.CheckboxGroup([],
#                             label="Negative Value Handling (Select Columns)",visible=False),gr.Dropdown(visible = False)

#         elif(negative_value_exposing_bool == True and outlier_expose_bool == False):
#             #gr.Info("We have Detected Negative Values for some features")
#             return gr.Dataframe(dataset,visible=False),gr.Plot(visible=False),gr.Textbox(
#                             visible=True),gr.Button("Start Feature Selection",variant='primary',
#                             visible=True),gr.Dropdown(['ABS','IGNORE','KEEP AS IT IS'],value="KEEP AS IT IS",
#                                                       visible=True),gr.CheckboxGroup(config.preprocess_rule_api["data_preprocessing"]['negative value columns names'],
#                             label="Negative Value Handling (Select Columns)",visible=True),gr.Dropdown(visible=False)


#         elif(negative_value_exposing_bool == False and outlier_expose_bool== True):
            
#             #gr.Info("We have Detected Outliers")
#             return gr.Dataframe(dataset,visible=False),gr.Plot(visible=False),gr.Textbox(
#                             visible=True),gr.Button("Start Feature Selection",variant='primary',
#                             visible=True),gr.Dropdown(['ABS','IGNORE','KEEP AS IT IS'],value="KEEP AS IT IS",visible=False),gr.CheckboxGroup(
#                             config.preprocess_rule_api["data_preprocessing"]['negative value columns names'],label="Negative Value Handling (Select Columns)",
#                             visible=False) ,gr.Dropdown(['Yes','No'] ,label="Outlier Handling",visible=True)


#         elif(negative_value_exposing_bool == True and outlier_expose_bool == True):
#             #gr.Info("We have Detected Negative Values & Outliers in your data")
#             return gr.Dataframe(dataset,visible=False),gr.Plot(visible=False),gr.Textbox(
#                             visible=True),gr.Button("Start Feature Selection",variant='primary',
#                             visible=True),gr.Dropdown(['ABS','IGNORE','KEEP AS IT IS'],value="KEEP AS IT IS",visible=True),gr.CheckboxGroup(config.preprocess_rule_api["data_preprocessing"]['negative value columns names'],
#                             label="Negative Value Handling (Select Columns)",visible=True),gr.Dropdown(['Yes','No'] ,label="Outlier Handling",visible=True)
#         else:
#             pass

#     except Exception as e:
#         if (received_target=="" or received_target==None or received_target!="Select Target"):
#             gr.Info("Please Select the Target : Error Info " + str(e))
#         else:
#             gr.Warning("Unexpected Error : " + str(e))



# """Apply Button Two --Extra"""


# def select_target_column_pycaret_process_extra(dataset : pd.DataFrame,negative_value_handle_method_by_user : str,
#                                                negative_value_handle_method_by_user_checkbox_group : list,
#                                                outlier_handle_by_user : str) -> tuple:
#     progress =gr.Progress()
#     progress(0, desc="Working...")
#     try:
#         # Apply your custom functions using the target column
#         negative_value_handle_method_by_user_checkbox_group = [
#             item for item in negative_value_handle_method_by_user_checkbox_group if item in dataset.columns]
#         config.preprocess_rule_api["data_preprocessing"]["negative value handling method selected by user"] = negative_value_handle_method_by_user
                
#         if(negative_value_handle_method_by_user=="KEEP AS IT IS"):
#             config.preprocess_rule_api["data_preprocessing"]["negative value columns to be handled selected by user"] = []
#         else :
       
#             config.preprocess_rule_api["data_preprocessing"]["negative value columns to be handled selected by user"] = negative_value_handle_method_by_user_checkbox_group
            
#             if(len(negative_value_handle_method_by_user_checkbox_group)>0):
                
#                 dataset = dcp.negative_value_handling(dataset,method = negative_value_handle_method_by_user)
            
#         if(outlier_handle_by_user == "Yes"):
#             config.preprocess_rule_api["data_preprocessing"]["outlier handling bool selected by user"] = bool(True)
#             dataset = dcp.outlier_handling(dataset)
#         else: 
#             config.preprocess_rule_api["data_preprocessing"]["outlier handling bool selected by user"] = bool(False)
           
            
#         json.dump(config.preprocess_rule_api, open('rules_modular_approach.json', 'w'),indent=4)
#         progress(0.5)
#          # ADF Test
#         no_of_lags_in_timeseries_handling(dataset)
#         progress(0.7)
#         dataset,config.best_model,X_test,y_test = pycaret.pycaret_dataset_feed_and_Initial_Col_name_mapping(dataset)
#         #monitor_cpu_usage(3)
#         image_path = pycaret.Shap_plot_image(dataset,X_test,y_test)
#         progress(1)
#         return image_path,gr.Textbox(visible=False)
        

#     except Exception as e:
#         gr.Warning("Unexpected Error (extra)  : " + str(e))



# # Latest Value in JSON File
# def current_json_api_values_fetch() -> tuple:
#     return  config.preprocess_rule_api \
#             ,config.preprocess_rule_api["data_preprocessing"]["negative value detection bool"],config.preprocess_rule_api["data_preprocessing"]["negative value columns names"],config.preprocess_rule_api["data_preprocessing"]['negative value columns to be handled selected by user'], config.preprocess_rule_api["data_preprocessing"]["negative value handling method selected by user"]\
#             ,config.preprocess_rule_api["data_preprocessing"]["outlier detection bool"], config.preprocess_rule_api["data_preprocessing"]["outlier handling bool selected by user"] \
#             ,config.preprocess_rule_api["data_preprocessing"]["duplicate value detection bool"],config.preprocess_rule_api["data_preprocessing"]["duplicate value handling method selected by user"]



