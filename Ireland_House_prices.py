import matplotlib
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
import numpy as np
import re
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from fastapi import FastAPI
from enum import Enum
import uvicorn
app = FastAPI()


class IrelandHousePrice:
    # ---------------------  String datatype check ----------------------#
    def is_string_dtype(self,inputcolumn):
            return  is_string_dtype(inputcolumn)

    # ---------------------  Numeric datatype check ----------------------#
    def is_numeric_dtype(self,inputcolumn):
            return  is_numeric_dtype(inputcolumn)

    # ---------------------  Numeric value check ----------------------#
    def CheckNumericColValues(self,a):
            output = re.sub(r'\d+', '', a)
            return len(output)

    # ---------------------  Removing outliers----------------------#
    def remove_pps_outliers(self,df):
        df_out =pd.DataFrame()
        print('called def  remove_pps_outliers ............')
        for key,subdf in  df.groupby('Area'):
            #print(key,'.....',subdf)
            med = np.mean(subdf.price_per_m2)
            std = np.std(subdf.price_per_m2)
            #print(med, '..med...')
            #print(std, '...std..')
            reduced_df=subdf[(subdf.price_per_m2 > (med-std)) & (subdf.price_per_m2 <=(med+std))]
            df_out=pd.concat([df_out,reduced_df],ignore_index=True)
        return df_out

    # ---------------------  Scatter plotting----------------------#

    def plot_scatter(self,df,Area):
        bed3 = df[(df.Area == Area) & (df.no_of_Bedrooms == 2)]
        bed4 = df[(df.Area == Area) & (df.no_of_Bedrooms == 4)]
        matplotlib.rcParams['figure.figsize'] = (15,10)
        plt.scatter(bed3.Floor_Area_m2,bed3.price_per_m2, color='blue',  label='3 BED', s=50)
        plt.scatter(bed4.Floor_Area_m2,bed4.price_per_m2, color='green', label='4 BED', s=50)
        plt.xlabel("Total square feet Area")
        plt.ylabel("Price square feet Area")
        plt.title(Area)
        plt.legend()
        #plt.show()

    # ---------------------  choose the best model score----------------------#
    def best_ml_models_score(self,x_train,x_test,y_train,y_test):
        models = [
            # ,('Linear Regression', LinearRegression())
            ('Decision Tree', DecisionTreeClassifier())
           ,('Random Forest', RandomForestClassifier())
            ,('SVM', SVC())
            ,('KNN', KNeighborsClassifier())
            ,('Naive Bayes', GaussianNB())
            #,('xgboost', XGBClassifier())
                ]
        results = []
        for name, model in models:
            model.fit(x_train, y_train)  # Train the model
            y_pred = model.predict(x_test)  # Make predictions
            accuracy = accuracy_score(y_test, y_pred)  # Evaluate the model
            results.append((name, accuracy))
        #  Print the results
        print("Model Performance (Accuracy):")
        for name, accuracy in results:
            print(f"{name}: {accuracy:.4f}")
        # Optionally, select and train the best model
        best_model_name, best_accuracy = results[0]
        best_model = models[0][1]
        best_model.fit(x_train, y_train)
        print(f"\nBest model: {best_model_name} with accuracy: {best_accuracy:.4f}")

    # --------------------- Model Linear regression -----------------------#
    def Tuning_Linear_Regression(self, x_train, x_test, y_train, y_test):
            param_space = {'copy_X': [True, False],
                           'fit_intercept': [True, False],
                           'n_jobs': [1, 5, 10, 15, None],
                           'positive': [True, False]}

            random_search = RandomizedSearchCV(LinearRegression(), param_space, n_iter=40, cv=5)
            random_search.fit(x_train, y_train)
            # Parameter which gives the best results
            print(f"Best Hyperparameters: {random_search.best_params_}")
            # Accuracy of the model after using best parameters
            print(f"Best Score: {random_search.best_score_}")

    # ---------------------  Predict price using Model Linear regression -----------------------#
    def predict_price(self,x,Area,no_of_Bedrooms,no_of_Bathrooms,Floor_Area_m2):
            area_index = np.where(x.columns ==Area)[0][0]
            x=np.zeros(len(x.columns))
            x[0] = no_of_Bedrooms
            x[1] = no_of_Bathrooms
            x[2] = Floor_Area_m2
            if area_index >= 0:
                x[area_index] = 1
            return lnr_clf.predict([x])[0]

if __name__ == '__main__':
        #uvicorn.run(app, host="127.0.0.1", port=5049)
        obj = IrelandHousePrice()
        # Reading data from Ireland_housing_data.csv
        df = pd.read_csv('D:\\My Documents\\AI\\python\\AI_ML_PYTHON\\source_data\\Ireland_housing_data.csv')
        df = df.rename_axis(None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 10000)
        #print (df.groupby('Area')['Area'].agg('count'))
        # --------------------------------------------------------------------------------------------------------------------------------------#
        #
        #----------------------------------------###########  STAGE 1 : DATA CLEANING  ########### ---------------------------------------------#
        #
        # --------------------------------------------------------------------------------------------------------------------------------------#
        print('---------- # Dropping columns which not required for prediction......---------------')
        print (df.shape)
        print(df.describe())
        df_req_columns=df.drop(['Title','Latitude','Longitude','Listing Views','Features','Date of Construction'],axis='columns')
        print('---------- # Dropping Unwanted rows......---------------')
        df_req_columns = df_req_columns.dropna()
        df_req_columns = df_req_columns.query('Price not in ["N/A","Price on Application","AMV: Price on Application","£200000(232959)"]')
        df_req_columns = df_req_columns.rename(columns={'BER Rating': 'BER_Rating'})
        df_req_columns = df_req_columns.rename(columns={'Floor Area (m2)': 'Floor_Area_m2'})
        df_req_columns = df_req_columns.rename(columns={'Property Type': 'Property_Type'})
        df_req_columns = df_req_columns.rename(columns={'Number of Bathrooms': 'no_of_Bathrooms'})
        df_req_columns = df_req_columns.rename(columns={'Number of Bedrooms': 'no_of_Bedrooms'})
        df_req_columns = df_req_columns.query('BER_Rating not in ["BER_PENDING","SI_666"]')

        #Column - Number of Bedrooms"

        df_req_columns['no_of_Bedrooms'] = (
                                     df_req_columns['no_of_Bedrooms']
                                     .str.strip(",")
                                     .astype(str)
                                     .astype(int)
                                     )
        # Column - PriceCorrected"
        # ---------------------  Exception handling  --------------------- #
        try:
         df_req_columns['PriceCorrected'] = (
                                    df_req_columns["Price"]
                                    .str.replace('AMV:', '')
                                    .str.replace('From', '')
                                    .astype(int)
                                    )
        except ValueError as e:
            print(e)
            print(df_req_columns["Price"],"The value you entered was not a number, please Validate this")

        # Column - Number of Bedrooms"
        print('-------------------------')
        print(df_req_columns.isnull().sum())
        print('-------------------------')
        print(df_req_columns.head)
        print('-------------------------')
        print (df_req_columns.shape)
        print('-------------------------')
        # checking null rows and dropping which is not required
        print('-------unique onPriceCorrected------------------')
        print(df_req_columns['PriceCorrected'].unique())
        print('-------unique on no_of_Bedrooms------------------')
        print(df_req_columns['no_of_Bedrooms'].unique())
        print('-------unique on no_of_Bathrooms------------------')
        print(df_req_columns['no_of_Bathrooms'].unique())
        print('-------unique on Property_Type------------------')
        print(df_req_columns['Property_Type'].unique())
        print('-------unique on BER_Rating------------------')
        print(df_req_columns['BER_Rating'].unique())
        print('-------unique on Floor_Area_(m2)------------------')
        print(df_req_columns['Floor_Area_m2'].unique())
        NumericCheck = obj.is_numeric_dtype(df_req_columns['Floor_Area_m2'])
        print(NumericCheck)
        print(df_req_columns.dtypes)

        #checking numeric values exist
        print('-------------***********************************************------------')
        for (columnName) in df_req_columns:
         if columnName == 'PriceCorrected' or columnName == 'no_of_Bathrooms' or columnName == 'no_of_Bedrooms':
          NumericCheck = obj.is_numeric_dtype(df_req_columns[columnName])
          print('Iteration for  :  ', columnName, ', NumericCheck = ', NumericCheck)
          if NumericCheck is True:
           for index, row in df_req_columns.iterrows():
             if obj.CheckNumericColValues(str(row[columnName])) != 0 :
                        raise ValueError(columnName,' - There is a  string values in the columns')
        print('-------------***********************************************------------')
        print(df_req_columns.head)
        print('-------------------------')
        df_final = df_req_columns.drop( ['Price','BER_Rating','Property_Type'], axis='columns')
        print(df_final.head)
        df_final['price_per_m2']=df_final['PriceCorrected']/df_final['Floor_Area_m2']
        print(df_final.head)
        print('-------------------------')
        # --------------------------------------------------------------------------------------------------------------------------------------#
        #
        #----------------------------------------########### STAGE 2 : preparing column for ONE HOT ENCODING  ########### ----------------------#
        #
        # --------------------------------------------------------------------------------------------------------------------------------------#
        print('-------********reducing the area which count is less than 10  to make less columns in one hot coding ***********---------')
        Area_count= df_final.groupby('Area')['Area'].agg('count')
        Area_count_less_than_10 =Area_count[ Area_count <=10]
        print(Area_count_less_than_10)
        print(len(df_final.Area.unique()))  #1440
        print(len(Area_count_less_than_10)) #1252
        df_final.Area=df_final.Area.apply(lambda x: 'other' if x in Area_count_less_than_10 else x)
        print('-------------------------')
        df_final = df_final.rename_axis(None)
        print(df_final.head(100))
        print(len(df_final.Area.unique()))  #189
        # --------------------------------------------------------------------------------------------------------------------------------------#
        #
        #----------------------------------------########### STAGE 3 :  Remove outliers  ########### -------------------------------------------#
        #
        # --------------------------------------------------------------------------------------------------------------------------------------#

        df_outliers=obj.remove_pps_outliers(df_final)
        print(df_outliers.shape)
        # ---------------------------------------------------------------------------------------------------------------------------------------#
        #
        #-----------------------########### STAGE 4 : Plotting the data points - scatter plot  ########### ---------------------------------------#
        #
        # ----------------------------------------------------------------------------------------------------------------------------------------#

        obj.plot_scatter(df_outliers,"ballsbridge-dublin")
        df_model_pre = df_outliers.drop(['price_per_m2','County'], axis='columns')
        # --------------------------------------------------------------------------------------------------------------------------------------#
        #
        #----------------------------------------###########  STAGE  5 : Building Model ########### ----------------------------------------#
        #
        # --------------------------------------------------------------------------------------------------------------------------------------#
        #print(df_model_pre.head(10))
        dummies = pd.get_dummies(df_model_pre.Area,dtype=int)
        df_model1 = pd.concat([df_model_pre,dummies.drop('other',axis='columns')],axis='columns')
        df_model  = df_model1.drop('Area', axis='columns')
        print(df_model)
        # --------------------------------------------------------------------------------------------------------------------------------------#
        #
        #----------------------------------------###########  STAGE  6 : Train test split  ########### ----------------------------------------#
        #
        # --------------------------------------------------------------------------------------------------------------------------------------#
        x=df_model.drop('PriceCorrected', axis='columns')
        y=df_model.PriceCorrected
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
        print(x)
        print(y)
        #--------------------------------------------------------------------------------------------------------------------------------------#
        #
        ##################################################  STAGE 7 : best_ml_models_score  ########### ----------------------------------------#
        #
        #--------------------------------------------------------------------------------------------------------------------------------------#
        #y = y.values.reshape(-1, 1)
        print(obj.best_ml_models_score(x_train, x_test, y_train, y_test))
        # --------------------------------------------------------------------------------------------------------------------------------------#
        #
        ##################################################  STAGE 8 : Use Linear Regression ########### ----------------------------------------#
        #
        # --------------------------------------------------------------------------------------------------------------------------------------#
        #cv=ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
        #print(cross_val_score(LinearRegression(),x,y,cv=cv))
        #print(cross_val_score(LogisticRegression(), x, y, cv=cv))
        #print(cross_val_score(RandomForestClassifier(), x, y, cv=cv))
        print('----------------------------------------------')
        #print(obj.Tuning_Linear_Regression(x_train, x_test, y_train, y_test))
        print('----------------------------------------------')
        print('Selected linear aggression....')
        lnr_clf = linear_model.LinearRegression()
        lnr_clf.fit(x, y)
        print(lnr_clf.score(x_test, y_test))
        print('prediction.............')
        print(obj.predict_price(x,'carrigaline-cork',4,3,217))

        #@app.get("/predict")
        #async def predict(x, Area, no_of_Bedrooms, no_of_Bathrooms, Floor_Area_m2):
        #return {"result": obj.predict_price(x, 'carrigaline-cork', 4, 3, 217)}



