# -*- coding: utf-8 -*-
"""
Created on Mon May 18 11:10:52 2020

@author: ADIL KHAN
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt




df1 = pd.read_csv("Bengaluru_House_Data.csv")


df1.head()

df1.shape


df1.columns

a=df1['location'].unique()


df1['area_type'].value_counts()

df2 = df1.drop(['area_type','society','balcony','availability'],axis=1)
df2.shape


#Data Cleaning: Handle NA values

df2.isnull().sum()

df3 = df2.dropna()

df3.isnull().sum()

#Feature Engineering

#Add new feature(integer) for bhk (Bedrooms Hall Kitchen)

df3['size'].unique()

df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
df3.bhk.unique()        

df3['bhk'].unique() 

df3[df3.bhk>20]  # greather than 20 bhk delete karna hai 

df3.total_sqft.unique()


#Explore total_sqft feature- to convert into float

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

df3[~df3['total_sqft'].apply(is_float)]

'''
Above shows that total_sqft can be a range (e.g. 2100-2850). 
For such case we can just take average of min and max value in the range. 
There are other cases such as 34.46Sq. Meter which one can convert to 
square ft using unit conversion. 
I am going to just drop such corner cases to keep things simple
'''


def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None

convert_sqft_to_num('2000.0')

convert_sqft_to_num('2000 - 2400')


df4 = df3.copy()
df4.total_sqft = df4.total_sqft.apply(convert_sqft_to_num)
df4 = df4[df4.total_sqft.notnull()]
df4.head(2)

#For below row, it shows total_sqft as 2475 which is an average of the range 2100-2850


df4.iloc[30]

(2100+2850)/2




#Feature Engineering

#Add new feature called price per square feet

df5 = df4.copy()
df5['price_per_sqft'] = df5['price']*100000/df5['total_sqft']
df5.head()


df5_stats = df5['price_per_sqft'].describe()
df5_stats



#Examine locations which is a categorical variable. 
#We need to apply dimensionality reduction technique here to reduce number of locations

len(df5.location.unique())

df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5['location'].value_counts(ascending=False)
location_stats

location_stats.values.sum()

len(location_stats[location_stats<=10])    

len(location_stats[location_stats>10])


len(location_stats)



'''
Dimensionality Reduction
Any location having less than 10 data points should be tagged as "other" location. 
This way number of categories can be reduced by huge amount. 
Later on when we do one hot encoding, 
it will help us with having fewer dummy columns
'''

location_stats_less_than_10 = location_stats[location_stats<=10]
location_stats_less_than_10

len(df5.location.unique())

df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
len(df5.location.unique())

df5.head(10)

'''
Outlier Removal Using Business Logic
As a data scientist when you have a conversation with your business manager 
(who has expertise in real estate), he will tell you that normally square ft 
per bedroom is 300 (i.e. 2 bhk apartment is minimum 600 sqft. If you have for
example 400 sqft apartment with 2 bhk than that seems suspicious and can be 
removed as an outlier. We will remove such outliers by keeping our minimum 
thresold per bhk to be 300 sqft
'''
#1000/4= remove 250

df5[df5.total_sqft/df5.bhk<300].head()

'''
Check above data points. We have 6 bhk apartment with 1020 sqft. 
Another one is 8 bhk and 
total sqft is 600. These are clear data errors that can be removed safely
'''
df5.shape

df6 = df5[~(df5.total_sqft/df5.bhk<300)]
df6.shape

df6.price_per_sqft.describe()

#Here we find that min price per sqft is 267 rs/sqft whereas max is 12000000,
#this shows a wide variation in property prices. We should remove outliers per 
#location using mean and one standard deviation




def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df7 = remove_pps_outliers(df6)
df7.shape



#Let's check if for a given location how does the 2 BHK and 
#3 BHK property prices look like


def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    plt.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=100)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=100)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(df7,"Rajaji Nagar")

plot_scatter_chart(df7,"Hebbal")

'''
We should also remove properties where for same location, the price of 
(for example) 3 bedroom apartment is less than 2 bedroom apartment 
(with same square ft area). What we will do is for a given location, 
we will build a dictionary of stats per bhk

Now we can remove those 2 BHK apartments whose price_per_sqft is less than 
mean price_per_sqft of 1 BHK apartment
'''
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):  
            stats = bhk_stats.get(bhk-1) 
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df8 = remove_bhk_outliers(df7)

df8.shape

plot_scatter_chart(df8,"Rajaji Nagar")

plot_scatter_chart(df8,"Hebbal")

'''
Based on above charts we can see that data points highlighted in red below 
are outliers and they are being removed due to remove_bhk_outliers function
'''

plt.rcParams["figure.figsize"] = (20,10)
plt.hist(df8.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")

#Outlier Removal Using Bathrooms Feature

df8.bath.unique()

plt.hist(df8.bath,rwidth=0.8)
plt.xlabel("Number of bathrooms")
plt.ylabel("Count")

df8[df8.bath>10]

#It is unusual to have 2 more bathrooms than number of bedrooms in a home

df8[df8.bath>df8.bhk+2]  

'''
Again the business manager has a conversation with you (i.e. a data scientist)
that if you have 4 bedroom home and even if you have bathroom in all 4 rooms 
plus one guest bathroom, you will have total bath = total bed + 1 max. 
Anything above that is an outlier or a data error and can be removed
'''

df9 = df8[df8.bath<df8.bhk+2]     2+2=4  
df9.shape

df9.head(2)

df10 = df9.drop(['size','price_per_sqft'],axis=1)
df10.head(3)

#Use One Hot Encoding For Location

dummies = pd.get_dummies(df10.location)
dummies.head(3)

dummies=dummies.drop(['other'],axis=1)

df11 = pd.concat([df10,dummies],axis=1)

df11.head()


df12 = df11.drop('location',axis='columns')
df12.head(2)

#Build a Model Now.

X = df12.drop(['price'],axis='columns')
X.head(3)

y = df12['price']
y.head(3)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=20)

from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)

lr_clf.score(X_train,y_train)
lr_clf.score(X_test,y_test)








def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]



predict_price('1st Phase JP Nagar',1000, 2, 2)

predict_price('1st Phase JP Nagar',1000, 3, 3)


predict_price('Indira Nagar',100000000, 250, 250)

predict_price('Indira Nagar',1000, 3, 3)
