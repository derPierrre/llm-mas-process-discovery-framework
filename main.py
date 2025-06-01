from user_functions import *
import mlflow

mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('final_test')



experiment('monolithic', 'shop', 'deepseek')