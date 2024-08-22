import pandas as pd

# Pycaret Framework
from pycaret.classification import ClassificationExperiment
from pycaret.regression import RegressionExperiment
from explainerdashboard import ClassifierExplainer, RegressionExplainer

# SHaply Explanations
import shap

# Visualization
import matplotlib.pyplot as plt
import plotly.graph_objects as go
# Module Import
import config

#Gradio Import
import gradio as gr


def filter_out_target_column(dataset : pd.DataFrame) -> pd.DataFrame:
    """filter/split Dataset

    filter X part from Dataset and Y part separately
    
    Parameters
    ----------
    dataset : pandas.DataFrame
 
    Returns
    -------
    filtered_features_df : pandas.DataFrame
        return X Part
   
    """
    # Check if the target column exists in the DataFrame
    if config.target_column in dataset.columns.to_list():
        # Filter out the target column specified by the user input
        filtered_features = dataset.loc[:, dataset.columns != config.target_column]
        filtered_features_df = pd.DataFrame(filtered_features, columns=filtered_features.columns)
        return filtered_features_df
    else:
        print("Target column specified by the user does not exist in the DataFrame.")
        return pd.DataFrame()


def identify_and_setup_task(dataset : pd.DataFrame,filtered_dataset : pd.DataFrame,target_column : str) -> object:
    """Pycaret Task

    identify the problem type (Calssification/Regression) & prepare Pycaret setup object
    
    Parameters
    ----------
    dataset : pandas.DataFrame
    filtered_dataset : pd.DataFrame 
    target_column : str
 
    Returns
    -------
    clf,X_test,y_test : tuple(object,pandas.DataFrame,pandas.DataFrame)
            
  
    """
    unique_values = dataset[target_column].unique()

    r = RegressionExperiment()
    c = ClassificationExperiment()
    numeric_cols = filtered_dataset.select_dtypes(include=['float64','int64','int32','float32']).columns.to_list()
    # To determine if the target column should be treated as classification or regression
    if len(unique_values) <= 10:  # threshold for classification
        config.problem_type = 'classification'
        print("Target column is for classification")
        clf = c.setup(filtered_dataset, target=dataset[target_column],numeric_features=numeric_cols, preprocess=False,transformation= False,fix_imbalance=True,session_id=1,index=False, experiment_name = 'Classification', remove_multicollinearity=True, multicollinearity_threshold=0.80)
        #clf = c.setup(filtered_dataset, target=dataset[target_column], preprocess=False,transformation= False,fix_imbalance=True,index=False, experiment_name = 'Classification', remove_multicollinearity=True, multicollinearity_threshold=0.80,normalize = True, normalize_method ='zscore')
        X_test = c.get_config('X_test')
        y_test = c.get_config('y_test')
        print("Classification setup completed.")
    else:
        config.problem_type = 'regression'
        print("Target column is for regression")
        clf = r.setup(filtered_dataset, target=dataset[target_column],numeric_features=numeric_cols,preprocess=False,transformation= False, session_id=2, experiment_name ='Regression',index=False)
        #clf = r.setup(filtered_dataset, target=dataset[target_column], preprocess=False,transformation= False, experiment_name ='Regression',index=False,normalize = True, normalize_method ='zscore')
        print("Regression setup completed.")
        X_test = r.get_config('X_test')
        y_test = r.get_config('y_test')
    print(X_test.shape)
    print(y_test.shape)
    return clf,X_test,y_test


"""Pycaret_data_feed Function"""
def pycaret_dataset_feed_and_Initial_Col_name_mapping(dataset : pd.DataFrame) -> tuple:
    """Pycaret Task -> compare models (Fold=5) -> best model

    Find best model 
    
    Parameters
    ----------
    dataset : pandas.DataFrame
 
    Returns
    -------
    dataset,best_model,X_test,y_test : tuple(pandas.DataFrame,str,pandas.DataFrame,pandas.DataFrame)
  
  
    """
    # X & Y part division
    filtered_dataset = filter_out_target_column(dataset)
    # Pycaret setup
    lib,X_test,y_test = identify_and_setup_task(dataset,filtered_dataset,config.target_column)
    # Restriction Kfold= 5
    best_model = lib.compare_models(fold=5,exclude = ['lightgbm','qda','dummy','ridge','catboost','br','et','lasso','huber','llar'])
    return dataset,best_model,X_test,y_test

def Shap_plot_image(dataset : pd.DataFrame,X_test : pd.DataFrame,y_test: pd.DataFrame) -> str:
    try:
        
        #X, y = filtered_dataset, dataset[target_column]
        shap.initjs()
        if(config.problem_type == 'classification'):
            # Load the SHAP values for the features
            explainer = shap.Explainer(config.best_model.predict, X_test)
            #explainer = ClassifierExplainer(best_model.predict, X)
            shap_values = explainer(X_test, max_evals=1200)
            feature_importance = pd.DataFrame(shap_values.values, columns=X_test.columns).abs().mean().sort_values(ascending=False).head(10)
            feature_importance = pd.DataFrame({
                'Feature Name': feature_importance.index,
                'Feature Importance': feature_importance.values
            })
        
        else:
            #shap_values = shap.Explainer(best_model)(Xtrain)
            explainer = RegressionExplainer(config.best_model, X_test, y_test)    
            #InlineExplainer(explainer).tab.importances()
            feature_importance = config.best_model.feature_importances_
            feature_importance = list(zip(X_test.columns, feature_importance))
            feature_importance = pd.DataFrame(feature_importance, columns=['Feature Name', 'Feature Importance']).sort_values(by='Feature Importance', ascending=False).head(20)
            #feature_importance = feature_importance.set_index('Feature Name')
     
        print(feature_importance)
        # plt.tight_layout()
        # # Save plot as an image file
        # feature_importance.plot(kind="barh")
        # image_path = "images\shap_plot.png"
        # plt.title(f"Top influencing features - {target_column}")
        # plt.subplots_adjust(left=0.3, right=0.9, bottom=0.2, top=0.8)
        # plt.savefig(image_path,dpi=300)
        # plt.show()
        #feature_importance = feature_importance.reset_index()
        try:
            fig = go.Figure(go.Bar( \
                                x=feature_importance['Feature Name'], \
                                y=feature_importance['Feature Importance'], \
                                #text=feature_importance['Importance'].round(4), \
                                textposition='auto', \
                                #colorscale='inferno', \
                                #orientation='h',
                                marker=dict(color='forestgreen'), \
                                opacity=0.8
    
                        ))
            fig.update_layout(
                title=f"Top influencing features - {config.target_column}", \
                xaxis_title='Feature Name', \
                yaxis_title='Feature Importance', \
                bargap=0.2, \
                
            )
            #gr.Plot(fig.show(),visible=True)
        except Exception as e:
            print("Exception Block Executed : Shap_Plot_image")
            print(str(e))
            
        #return image_path
        #return gr.BarPlot(feature_importance,title =f"Top influencing features - {target_column}" , x="Importance", y='Feature Name',visible=True)
        return gr.Plot(value = fig,label="Recommended Features",visible=True)
    
    except Exception as e:
        gr.Warning("Check SHAP Plot Function : " + str(e))
"""

def Shap_plot_image(dataset : pd.DataFrame,X_test : pd.DataFrame,y_test: pd.DataFrame) -> str:
    try:
        
        #X, y = config.filtered_dataset, dataset[config.target_column]
        shap.initjs()
        if(config.problem_type == 'classification'):
            # Load the SHAP values for the features
            explainer = shap.Explainer(config.best_model.predict, X_test)
            #explainer = ClassifierExplainer(best_model.predict, X)
            shap_values = explainer(X_test, max_evals=1200)
            feature_importance = pd.DataFrame(shap_values.values, columns=X_test.columns).abs().mean().sort_values(ascending=True).tail(5)
        
        else:
            #shap_values = shap.Explainer(best_model)(Xtrain)
            explainer = RegressionExplainer(config.best_model, X_test, y_test)    
            #InlineExplainer(explainer).tab.importances()
            feature_importance = config.best_model.feature_importances_
            feature_importance = list(zip(X_test.columns, feature_importance))
            feature_importance = pd.DataFrame(feature_importance, columns=['Feature Name', 'Importance']).sort_values(by='Importance', ascending=True).tail(5)
            feature_importance = feature_importance.set_index('Feature Name')
        
        print(feature_importance)

        # Create a bar plot to visualize feature importance using SHAP values
        #shap.plots.bar(shap_values)
        plt.tight_layout()
    
        # Save plot as an image file
        
        #feature_importance = pd.DataFrame(shap_values.values, columns=X.columns).abs().mean().sort_values(ascending=True).head(5)
        feature_importance.plot(kind="barh")
        image_path = "images\shap_plot.png"
        plt.subplots_adjust(left=0.3, right=0.9, bottom=0.2, top=0.8)
        #plt.figure(figsize=plt.gcf().get_size_inches())
        #plt.figure(figsize=(5, 5))         
        plt.savefig(image_path,dpi=300)
        plt.show()
        return image_path

    except Exception as e:
        print(str(e))
"""