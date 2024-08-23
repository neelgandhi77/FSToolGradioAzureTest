# Python Libraries 
import pandas as pd
import json
import os
from fastapi import FastAPI

# UI Libraries
import gradio as gr
#from gradio.themes.base import Base

# Pycaret Framework
from pycaret.classification import ClassificationExperiment
from pycaret.regression import RegressionExperiment
#from explainerdashboard import ClassifierExplainer, RegressionExplainer

# SHaply Explanations
import shap

# # Modules Import
# import config
# import data_upload # Not in-use at this stage
# import data_cleaning_preprocessing as dcp
# import target_column_fetch as tc
# import pycaret_process_with_SHAP as pycaret
# import gradio_button_functions as gbf


# # Visualization
# import matplotlib.pyplot as plt


app = FastAPI()


"""
def pycaret_process(file_path):
    dataset = pd.read_csv(file_path)
    target_column = "SalePrice"
    r = RegressionExperiment()
    filtered_features = dataset.loc[:, dataset.columns != target_column]
    filtered_dataset = pd.DataFrame(filtered_features, columns=filtered_features.columns)
    clf = r.setup(filtered_dataset, target=dataset[target_column],preprocess=False,transformation= False, session_id=2, experiment_name ='Regression',index=False)
    best_model = clf.compare_models(fold=5,exclude = ['lightgbm','qda','dummy','ridge','catboost','br','et','lasso','huber','llar'])
    return gr.Textbox(label="Best Model",value=best_model,placeholder="NOT SET")

with gr.Blocks(analytics_enabled=False,title="SE FSTool") as FSTool:
    gr.Markdown("<span style='color: black;'>Please upload a CSV file</span>",sanitize_html=True)
       
    with gr.Row():
        with gr.Column():  
            file_input = gr.File(label="Upload Dataset",file_types=['.csv'],file_count="single")  
            apply_button = gr.Button("Start Process",variant='primary',visible=True)
            best_model = gr.Textbox(label="Best Model",placeholder="Best Model")
            apply_button.click(pycaret_process,inputs=[file_input],outputs=[best_model])

app = gr.mount_gradio_app(app, FSTool, path="/")

# # Main Method
# #def main():
        
#     # """Folder Creation"""
#     # folder_path = r"C:\Users\SESA749667\Desktop\Feature Selection Tool\Shared\Aman_Branch_Modular_Code\images"
#     # # Check if the folder exists
#     # if not os.path.exists(folder_path):
#     # # Create the folder if it doesn't exist
#     #     os.makedirs(folder_path)


#     # folder_path_json = r"C:\Users\SESA749667\Desktop\Feature Selection Tool\Shared\Aman_Branch_Modular_Code\rules_modular_approach.json"
#     # # Check if the JSON file exists
#     # if not os.path.exists(folder_path_json):
#     #     # Create the JSON file if it doesn't exist
#     #     print("Creating New JSON as its not exist")
#     #     with open(folder_path_json, "w") as file:
#     #         # You can write any initial data to the JSON filC:\Users\SESA749667\Desktop\Feature Selection Tool\Shared\Modular Approach\app.pye here
#     #         json.dump({
#     #             "data_preprocessing": {
#     #                 "negative value detection bool": False,
#     #                 "negative value columns names": "",
#     #                 "negative value columns to be handled selected by user": "",
#     #                 "negative value handling method selected by user": "KEEP AS IT IS",
                    
#     #                 "outlier handling bool selected by user": False,
#     #                 "outlier detection bool": False,
                    
#     #                 "duplicate value detection bool": False,
#     #                 "duplicate value handling method selected by user": "LATEST"
#     #             }
#     #         }, file,indent=4)


# # JSON API
# with open('rules_modular_approach.json') as f:
#     config.preprocess_rule_api = json.load(f)

# # Resetting
# # Resetting
# gbf.reset_json_api()

# # Define the custom theme 
# custom_theme = gr.themes.Default(primary_hue= gr.themes.colors.green)
# # Define custom CSS and theme
# custom_css = """
# footer {visibility: hidden;}
# .gradio-container .image {
#         width: 10%;
#         height: 10%;
#         object-fit: contain;}

# """
# # Define the cache duration in seconds
# #cache_duration = 86400 * 30  # 30 days
# #with gr.Blocks(delete_cache=(120,86400*30),css="footer {visibility: hidden}",theme=custom_theme) as FSTool:
            
            
# with gr.Blocks(analytics_enabled=False,css=custom_css,theme=custom_theme,title="SE FSTool") as FSTool: 
#     gr.Image(type='filepath',value=gbf.static_image_path,width=11,show_download_button=False,show_label=False)
                
#     with gr.Tabs():
#         with gr.TabItem("Configuration"):
#             json_values = gr.Textbox(json.load(open("rules_modular_approach.json",'r')),label="Default",visible=False)
#             #gr.DataFrame(value= pd.read_csv(r"C:\Users\SESA749667\Desktop\Feature Selection Tool\Additional\aevocado_dirty.csv"),label="Cleaned & Preprocessed Dataset",interactive=False,visible=True)
#             refresh_config_button = gr.Button("Refresh",variant='primary')        
#             with gr.Row():
                    
#                     with gr.Column():
#                         negative_value_detection_bool = gr.Textbox(label="Negative value detection bool (by ENVIRONMENT)")
#                         negative_value_column_names = gr.Textbox(label="Negative value columns names (by ENVIRONMENT)")
#                         negative_value_columns_to_be_handled_selected_by_user = gr.Textbox(label="Negative value columns to be handled selected by user")
#                         negative_value_handling_method_selected_by_user = gr.Textbox(label="Negative value handling method selected by user")

#                     with gr.Column():
#                         outlier_detection_bool =  gr.Textbox(label="Outlier detection bool (by ENVIRONMENT)")
#                         outlier_handling_bool_selected_by_user =  gr.Textbox(label="Outlier handling bool selected by user")

#                     with gr.Column():
#                         duplicate_value_detection_bool =  gr.Textbox(label="Duplicate value detection bool (by ENVIRONMENT)")
#                         duplicate_value_handling_method_selected_by_user =  gr.Textbox(label="Duplicate value handling method selected by user")
                        

#             #save_config_button = gr.Button("SAVE",variant='primary')       
#             refresh_config_button.click(gbf.current_json_api_values_fetch,inputs=[],outputs=[json_values,
#                                 negative_value_detection_bool,negative_value_column_names,negative_value_columns_to_be_handled_selected_by_user,negative_value_handling_method_selected_by_user
#                                 ,outlier_detection_bool,outlier_handling_bool_selected_by_user
#                                 ,duplicate_value_detection_bool,duplicate_value_handling_method_selected_by_user])
            
#         with gr.TabItem("Feature Selection"):
#                 #gr.Markdown("<span style='color: red;'>*Note : </span>The current_index : not necessary to be in complete sequence order ; as the duplicates got Removed/Smoothening taken place.",sanitize_html=True)
#                 gr.Markdown("<span style='color: black;'>Please upload a CSV file</span>",sanitize_html=True)
                
#                 with gr.Row():
#                     with gr.Column():
                        
#                         file_input = gr.File(label="Upload Dataset",file_types=['.csv'],file_count="single")  
#                         load_button = gr.Button("Upload Dataset",variant='primary')
#                         cleaned_preprocessed_data_for_show = gr.DataFrame(label="Cleaned & Preprocessed Dataset",interactive=False,visible=True)
#                         cleaned_preprocessed_data_for_stages = gr.DataFrame(label="Cleaned & Preprocessed Dataset",interactive=False,visible=False)
                        
#                         duplicate_handle_method = gr.Dropdown(value= "<<--SELECT-->> (log : Duplicate Handle Method)",label="Select Duplicate Values Handling Method"
#                                                             ,visible=False)
#                         target_column = gr.Dropdown(value= "<<--SELECT-->> (log : Target Column)" ,label="Select Target Column",visible=False)
#                         apply_button = gr.Button("Start Process",variant='primary',visible=False)
                        
#                         negative_value_handle_method_by_user_checkbox_group = gr.CheckboxGroup(
#                             label="Negative Value Handling (Select Columns))",visible=False)
#                         negative_value_handle_method_by_user = gr.Dropdown(value="<<--SELECT-->> (log : Negative Value Handle Method Name)", label="Negative Value Handling Method Name",visible=False)
                    
#                         outlier_handle_by_user = gr.Dropdown(value= "<<--SELECT-->> (log : Outlier Handle)"
#                                                             ,label="Outlier Handling",visible=False)
#                         apply_button_extra = gr.Button("Start Feature Selection",variant='primary',visible=False)
#                         no_plot_text= gr.Textbox(visible=False,show_label=False)
#                         #FS_image_output = gr.Image(type='filepath',label="Recommended Features",visible=False,show_download_button=False,interactive=False)    
#                         FS_image_output = gr.Plot(visible=False)
                    
#                         load_button.click(gbf.load_dataset, inputs=[file_input], outputs=[cleaned_preprocessed_data_for_show,
#                                                                                     cleaned_preprocessed_data_for_stages,
#                                                                                     target_column,
#                                                                                     apply_button,FS_image_output,no_plot_text,
#                                                                                     apply_button_extra,
#                                                                                     negative_value_handle_method_by_user,
#                                                                                     negative_value_handle_method_by_user_checkbox_group,
#                                                                                     outlier_handle_by_user,
#                                                                                     duplicate_handle_method])
                                                                                
                        
                    
#                         apply_button.click(gbf.select_target_column_pycaret_process_normal,queue=True, inputs=[cleaned_preprocessed_data_for_stages
#                                                                         ,target_column,duplicate_handle_method], 
#                                         outputs=[cleaned_preprocessed_data_for_stages,FS_image_output,no_plot_text,apply_button_extra,negative_value_handle_method_by_user,negative_value_handle_method_by_user_checkbox_group,
#                                                     outlier_handle_by_user])       
#                         apply_button_extra.click(gbf.select_target_column_pycaret_process_extra,inputs=[cleaned_preprocessed_data_for_stages,negative_value_handle_method_by_user,
#                                                                                             negative_value_handle_method_by_user_checkbox_group,outlier_handle_by_user],
#                                         outputs=[FS_image_output,no_plot_text])
# #FSTool.launch(server_port=8080,server_name='127.0.0.1')
# #FSTool.launch(share=SHARE,auth=[('admin','pass'),('se','se')],auth_message="Please Enter Credentials")
# #FSTool.queue(default_concurrency_limit=5)
# #FSTool.launch()




# """main Method Call"""
# if __name__ == '__main__':
#     main()





