import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import yaml
import os
import mlflow
from urllib.parse import urlparse


os.environ['MLFLOW_TRACKING_URI']="https://dagshub.com/abhishek-paul-d/mlpipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME']="abhishek-paul-d"
os.environ['MLFLOW_TRACKING_PASSWORD']="8268eeef9b8163ae34f9154fc93810bc6df074cf" 

##Load the parameters from params.yaml
params=yaml.safe_load(open("params.yaml"))["train"]

def evaluate(data_path,model_path):
    data=pd.read_csv(data_path)
    X=data.drop(columns=["Outcome"]) 
    y=data["Outcome"]
    
    mlflow.set_tracking_uri("https://dagshub.com/abhishek-paul-d/mlpipeline.mlflow")

    ##load the model from the disk 
    model=pickle.load(open(model_path,'rb'))

    predictions=model.predict(X)
    accuracy=accuracy_score(y,predictions)

    ##Log the metrics to MLFLOW

    mlflow.log_metric("accuracy",accuracy)
    print("Model_accuracy:{accuracy}")

if __name__=="__main__": 
    evaluate(params["data"],params["model"])

