import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from datasist.structdata import detect_outliers
from category_encoders.binary import BinaryEncoder
 
## other
#from imblearn.over_sampling import SMOTE

## sklearn -- preprocessing
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn_features.transformers import DataFrameSelector
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, cross_validate, GridSearchCV
from sklearn.impute import SimpleImputer

from sklearn.feature_selection import SelectKBest, mutual_info_regression

## sklearn -- models
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.tree import DecisionTreeRegressor

## skelarn -- metrics
from sklearn.metrics import f1_score, accuracy_score, r2_score, mean_squared_error

## ensemble models
from sklearn.ensemble import VotingRegressor,  AdaBoostRegressor, GradientBoostingRegressor

from xgboost import XGBRegressor

import re


# read dataset
df = pd.read_csv("houses.csv", na_values=['Unknown', '', 'na', 'nan', 'NA', 'NAN', '-' ])

# Drop index column and description column 
df.drop(['Ad List', 'description'], axis=1, inplace=True)



# trim and replace space inside columns names with _
df.columns = df.columns.str.lower().str.strip().str.strip().str.replace(' ', '_')
df.rename(columns={'#_of_floors':'number_of_floors'}, inplace=True)


# drop duplicates
df.drop_duplicates(inplace= True)
df.reset_index(inplace= True, drop= True)


# convert property_size to numerical
def fix_property_size(value:str):
    return value.replace('sq.ft.', '').strip()
df['property_size'] = df['property_size'].apply(fix_property_size).astype(float)


# drop category column
df.drop(['category'], axis=1, inplace=True)



# convert price to float
def fix_price(value:str):
    if isinstance(value, float):
        return value
    else:
        modified_value = value.replace(' ', '').replace('RM', '')
        return float(modified_value) 

df['price'] = df['price'].apply(fix_price) 


# convert discrete fields to Int64
df['bedroom'] = df['bedroom'].astype('Int64')
df['bathroom'] = df['bathroom'].astype('Int64')
df['completion_year'] = df['completion_year'].astype('Int64')
df['number_of_floors'] = df['number_of_floors'].astype('Int64')
df['total_units'] = df['total_units'].astype('Int64')
df['parking_lot'] = df['parking_lot'].astype('Int64')


# create column for each nearby service
df['nearby_highway'] = df['highway'].isna() == False
df['nearby_hospital'] = df['hospital'].isna() == False
df['nearby_railway_station'] = df['nearby_railway_station'].isna() == False
df['nearby_mall'] = df['nearby_mall'].isna() == False
df['nearby_mall2'] = df['mall'].isna() == False
df['nearby_railway_station'] = df['railway_station'].isna() == False
df['nearby_school'] = df['nearby_school'].isna() == False
df['nearby_bus_stop'] = df['bus_stop'].isna() == False
df['nearby_park'] = df['park'].isna() == False
df['nearby_school'] = df['school'].isna() == False

df.drop(columns=['highway', 'hospital', 'mall', 'railway_station', 'bus_stop', 'park', 'school'], axis=1, inplace=True)



# split facilities into separate columns
def split_facilities_into_separate_columns(df):
    for index in df.index:
        facilities = df.loc[index, 'facilities']
        
        if type(facilities) == str:
    
            parts = facilities.split(',')

            # loop on each facility 
            for part in parts:
                part = part.strip()
            
                col_name = 'facility_' + part
                if col_name not in df.columns:
                    # initialize new column with value False
                    df.loc[:, col_name] = False
                    
                # indicate that the current row has this facility
                df.loc[index, col_name] = True
    return df

df = split_facilities_into_separate_columns(df)
df.drop(['facility_10'], axis=1, inplace=True)
df.drop(['facilities'], axis=1, inplace=True)


# split ren_number into two columns
def split_agent_number(value):
    if isinstance(value, float):
        return pd.Series([np.nan, float(value)])
    else:
        parts = value.split(' ')
        return pd.Series([parts[0], float(parts[1])]) 
 
df[['ren_type', 'ren_number']] =  df['ren_number'].apply(split_agent_number) 
df['ren_number'] = df['ren_number'].astype('Int64')


# extract city, state from address
states = ['Johor', 'Kedah', 'Kelantan', 'Malacca', 'Negeri Sembilan', 'Pahang', 'Penang', 
          'Perak', 'Perlis', 'Sabah', 'Sarawak', 'Selangor', 'Terengganu', 'Kuala Lumpur', 'Labuan', 'Putrajaya']

cities = ['Kajang', 'Seberang Perai', 'Subang Jaya', 'Klang', 'Johor Bahru', 'Shah Alam', 'George Town', 'Petaling Jaya',
           'Selayang', 'Ipoh', 'Seremban', 'Iskandar Puteri', 'Kuantan', 'Sungai Petani', 'Ampang Jaya', 'Kota Kinabalu',
            'Melaka City', 'Sandakan', 'Alor Setar', 'Tawau', 'Batu Pahat', 'Kota Bharu', 'Kuala Terengganu', 'Kuching',
             'Sepang', 'Kulim', 'Muar', 'Pasir Gudang', 'Kuala Langat', 'Kulai', 'Kangar',
           'Kuala Selangor', 'Padawan', 'Miri', 'Manjung', 'Hulu Selangor', 'Taiping', 'Bintulu', 'Kubang Pasu', 'Kluang',
             'Pasir Mas', 'Lahad Datu', 'Alor Gajah', 'Kemaman', 'Hang Tuah Jaya', 'Tumpat', 'Pontian', 'Teluk Intan', 'Sibu', 
             'Temerloh', 'Semporna', 'Kerian', 'Tangkak', 'Penampang', 'Kota Samarahan', 'Ketereh', 'Dungun', 'Bachok',
               'Besut', 'Segamat', 'Keningau', 'Tanah Merah', 'Papar', 'Ampang', 'Setapak', 'Bayan Baru', 'Puchong', 'Wangsa Maju', 
               'Simpang Ampat', 'Cheras', 'Semenyih', 'Iskandar Puteri', 'Bangi', 'Ayer Keroh', 'Setia Alam', 'Sentul', 'Cyberjaya',
               'Seri Kembangan', 'Gelugor', 'Skudai', 'Ayer Itam', ' Tanjung Bungah', 'Rawang', 'Gelang Patah', 'Nusajaya', 'Damansara Perdana']
 

def extract_address_fields(value):
    if isinstance(value, str):   

        ## extract postal code, it is a number consists of 5 digits
        #postal_code = np.nan
        #postal_codes = re.findall(r'\d{5}', value)        
        #if len(postal_codes) > 0:
        #    postal_code = postal_codes[0]

        ## extract street names and numbers, they exist on the format 
        #postal_codes = re.findall(r'[^,]+\d+/\d+|[^,]+\d+', value)
        #if len(postal_codes) > 0:
        #    print(postal_codes)

        ## extract city, state they are the last two parts in the string
        address_parts = [x.strip() for x in value.split(',')]
        #print(list(set(cities) & set(address_parts)))
        
        #result = address_parts[-2:]
        result = list()

        found_states = list(set(states) & set(address_parts))
        if len(found_states) > 0:
            result = np.append(result, found_states[0])

        found_cities = list(set(cities) & set(address_parts))
        if len(found_cities) > 0:
            result = np.append(result, found_cities[0])
 
        #result = np.append(result, postal_code)
        return pd.Series(result)
    else: 
        return pd.Series([np.nan, np.nan])
   
df[['state', 'city']] = df['address'].apply(extract_address_fields)
df.drop(['address'], axis=1, inplace=True)


# detect outliers of property_size
cols = np.array(['property_size'])
 
for col in cols: 
    outliers_indices = detect_outliers(df, 0, [col]) 
     
    col_median = df[col].median() 
    df.loc[outliers_indices, col] = col_median


# take log of the price 
df['price'] = np.log(df['price'])



# split the dataset
## Features and target
X = df.drop(columns=['price'], axis=1)
y = df['price']

## to full train and test
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=50)


#preprocessing
## Slice cols
int_cols = X_train.select_dtypes(include=['Int64']).columns.tolist()
float_cols = X_train.select_dtypes(include=['float64']).columns.tolist()
bool_cols = X_train.select_dtypes(include=['bool']).columns.tolist()

#small_categ_cols = ['tenure_type', 'property_type', 'floor_range', 'land_title', 'firm_type', 'ren_type', 'state']
small_categ_cols = ['tenure_type', 'property_type', 'floor_range', 'land_title', 'ren_type', 'city', 'state']

categ_cols = X_train.select_dtypes(include=['object']).columns.tolist()
other_categ_cols = list(set(categ_cols) - set(small_categ_cols)) 


print(X_train.columns) 
print('*'*10)
#pd.set_option('display.max_columns', None)
#print(X_train.iloc[0:2, :])


## Int
int_pipline = Pipeline(steps=[
                ('selector', DataFrameSelector(int_cols)),
                ('imputer', SimpleImputer(strategy='most_frequent')),
               # ('scaler', MinMaxScaler())
            ])

## Float
float_pipline = Pipeline(steps=[
                ('selector', DataFrameSelector(float_cols)),
                ('imputer', SimpleImputer(strategy='median')),
               # ('scaler', MinMaxScaler())
            ])

## Bool
bool_pipline = Pipeline(steps=[
                ('selector', DataFrameSelector(bool_cols)),
               # ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder())
                #,
                #('scaler', MinMaxScaler())
            ])

## Categorical
small_categ_pipline = Pipeline(steps=[
                ('selector', DataFrameSelector(small_categ_cols)),
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder( sparse_output=False, handle_unknown='ignore')),
                #('scaler', MinMaxScaler())
            ])

other_categ_pipline = Pipeline(steps=[
                ('selector', DataFrameSelector(other_categ_cols)),
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value = -1)),
                #('encoder', BinaryEncoder(drop_invariant=True))
                #,                
                #('scaler', MinMaxScaler())
            ])

## Combine all
all_pipeline = FeatureUnion(transformer_list=[
                        ('int', int_pipline),
                        ('float', float_pipline),
                        ('bool', bool_pipline),
                        ('small_categ', small_categ_pipline),
                        ('other_categ', other_categ_pipline)
                    ])

all_pipeline = Pipeline(steps=[('pipeline', all_pipeline), 
                               ('FeatureSelection', SelectKBest(mutual_info_regression, k= int(0.9 * X_train.shape[1]))),
                               ('Scaler', RobustScaler())])
 

## apply
_ = all_pipeline.fit(X_train, y_train) 

def process_new(X_new):

    df_new = pd.DataFrame([X_new], columns=X_train.columns)

    X_processed = all_pipeline.transform(df_new)

    return X_processed

 