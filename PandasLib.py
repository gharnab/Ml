# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 17:34:37 2020

@author: shamaun
"""

import pandas as pd

#creating pandas dataframe

d = {'name':['Aman','Ananya',
             'Anshuman'],
     'roll':[1,2,3],
     'marks':[89,87,86]}

df = pd.DataFrame(d)

#Setting index

df.set_index('roll',inplace=True)

#reading external data

#data = pd.read_csv(r"E:\auto-mpg.csv")



#data cleaning using pandas

data = pd.read_csv(r"E:\auto-mpg.csv",header=None)

"""setting column names"""


data.columns = ['mpg',
                'cylinders',
                'displacement',
                'horsepower',
                'weight',
                'acceleration',
                'model_year',
                'origin',
                'car name'
                ]


#counting "?" in horsepower column

print(sum(data.horsepower=='?'))

#replacing "?"

data['horsepower'].replace('?',150.0,inplace=True)


desc = data.describe(include='all')

data.horsepower = data['horsepower'].astype(float)

data.to_csv("auto-mpg-clean.csv")


#  get see the EDA( univariant or Multivariant) 
pip install ydata-profiling

from pandas_profiling import ProfileReport
prof=ProfileReport(data)
prof.to_file(output_file="auto-mpg.csv")

# The above code will generate an HTML page that has Exploratory Data Analysis with the plotted Graph for better understanding of Data set 












