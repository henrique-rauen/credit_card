#! /usr/bin/python3

#Created by Henrique Rauen (rickgithub@hsj.email)
import pickle
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split

class Model():
    default_data_location = "data/BankChurners.csv"
    def __init__(self,model_name):
        print(f"Trying to open '{model_name}' model files...")
        try:
            with open(f"models/{model_name}.pkl", "rb") as file:
                save=pickle.load(file)
                self._model=save[0]
                self._transformer=save[1]
        except:
            print("Model files not found, trying to train new model")
            try:
                Model.train_model(Model.default_data_location, model_name)
                self.__init__(model_name)
            except:
                print("Unable to train model, object will be empty")

    @staticmethod
    def train_model(data
                    ,model_name
                    ,target=['Attrition_Flag']
                    ,model_choice=BalancedRandomForestClassifier(
                                    sampling_strategy='all'
                                    ,replacement=True)
                    ,columns=[
                              'Total_Relationship_Count'
                              ,'Total_Amt_Chng_Q4_Q1'
                              ,'Total_Trans_Amt'
                              ,'Total_Trans_Ct'
                              ,'Total_Ct_Chng_Q4_Q1'
                              ,'Avg_Utilization_Ratio']):
        """ 'data' = Csv file to read or df,
            'model_name' = name to save the model
            'target' = df column name of the target series
            'model_choice' = Callable sklearn style model
            'columns' = df column names for the features
            Default values assumes the specifics of the Bank Churners project.
            For a given 'model_choice', 'data' and 'columns'/'target' trains a
            new model' """
        df = pd.read_csv(data)
        X = df[columns].to_numpy()
        y = df[target].to_numpy()
        (X_train, X_test, y_train, y_test) = train_test_split(X,y,test_size= 0.2)
        transformer = MaxAbsScaler().fit(X_train)
        X_train = transformer.transform(X_train)
        X_test = transformer.transform(X_test)
        model=model_choice
        model.fit(X_train,y_train)
        with open(f"models/{model_name}.pkl", "wb") as file:
            pickle.dump([model, transformer],file)

    def apply_model(self, df):
        X=df.to_numpy()
        X=self._transformer.transform(X)
        pred=self._model.predict()
        df["Predictions"] = pred
        return df