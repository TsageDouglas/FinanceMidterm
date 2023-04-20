#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This notebook is for coding Sankey Diagrams for the Applied Finance 2023 Midterm Project. 
# Author: Tsage Douglas 
# Date: 6th April 2023 
# Data provided by: Banque de France


# In[2]:


import requests
import zipfile
import io
import os
import pandas as pd
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio


# In[3]:


get_ipython().system('pip install -U kaleido')


# In[4]:


#Automate data download 

# Set the URL of the file to download
#url = "https://webstat.banque-france.fr/ws_wsfr/downloadFile.do?id=51050"

# Send a GET request to the URL
#response = requests.get(url)

# Create a ZipFile object from the response content
#zip_file = zipfile.ZipFile(io.BytesIO(response.content))

# Extract the contents of the zip file to a temporary folder
#zip_file.extractall("temp")

# Check that the CSV file was successfully extracted
#if not os.path.exists("temp/Webstat_Export_20220318.csv"):
#    raise FileNotFoundError("CSV file not found in temp folder")

# Get the path of the extracted csv file
#csv_path = os.path.join("temp", "Webstat_Export_20220318.csv")

# Load the csv file into a Pandas DataFrame
#df = pd.read_csv(csv_path, sep=";", header=None, encoding="ISO-8859-1")

# Set the first row as the column names
#df.columns = df.iloc[0]

# Remove the first row (which is now the column names)
#df = df.iloc[1:]

# Save the DataFrame as a CSV file in the Jupyter notebook folder
#df.to_csv("Finance_Midterm_Project/Webstat_Export_20220318.csv", index=False)
#
# Clean up: delete the temporary folder and its contents
#os.remove(csv_path)
#os.rmdir("temp")


# In[5]:


os.chdir("/Users/tsagedouglas/Desktop/")
df = pd.read_stata('CFT_TRIMESTRIEL.dta')
print(df.head())


# In[6]:


# convert columns to float data type
df = df.astype({
    '_31dec1996': 'float',
    '_31mar1996': 'float',
    '_30jun1996': 'float',
    '_30sep1996': 'float',
    '_31dec1996': 'float',
    '_31mar1997': 'float',
    '_30jun1997': 'float',
    '_30sep1997': 'float',
    '_31dec1997': 'float',
    '_31mar1998': 'float',
    '_30jun1998': 'float',
    '_30sep1998': 'float',
    '_31dec1998': 'float',
    '_31mar1999': 'float',
    '_30jun1999': 'float',
    '_30sep1999': 'float',
    '_31dec1999': 'float',
    '_31mar2000': 'float',
    '_30jun2000': 'float',
    '_30sep2000': 'float',
    '_31dec2000': 'float',
    '_31mar2001': 'float',
    '_30jun2001': 'float',
    '_30sep2001': 'float',
    '_31dec2001': 'float',
    '_31mar2002': 'float',
    '_30jun2002': 'float',
    '_30sep2002': 'float',
    '_31dec2002': 'float',
    '_31mar2003': 'float',
    '_30jun2003': 'float',
    '_30sep2003': 'float',
    '_31dec2003': 'float',
    '_31mar2004': 'float',
    '_30jun2004': 'float',
    '_30sep2004': 'float',
    '_31dec2004': 'float',
    '_31mar2005': 'float',
    '_30jun2005': 'float',
    '_30sep2005': 'float',
    '_31dec2005': 'float',
    '_31mar2006': 'float',
    '_30jun2006': 'float',
    '_30sep2006': 'float',
    '_31dec2006': 'float',
    '_31mar2007': 'float',
    '_30jun2007': 'float',
    '_30sep2007': 'float',
    '_31dec2007': 'float',
    '_31mar2008': 'float',
    '_30jun2008': 'float',
    '_30sep2008': 'float',
    '_31dec2008': 'float',
    '_31mar2009': 'float',
    '_30jun2009': 'float',
    '_30sep2009': 'float',
    '_31dec2009': 'float',
    '_31mar2010': 'float',
    '_30jun2010': 'float',
    '_30sep2010': 'float',
    '_31dec2010': 'float',
    '_31mar2011': 'float',
    '_30jun2011': 'float',
    '_30sep2011': 'float',
    '_31dec2011': 'float',
    '_31mar2012': 'float',
    '_30jun2012': 'float',
    '_30sep2012': 'float',
    '_31dec2012': 'float',
    '_31mar2013': 'float',
    '_30jun2013': 'float',
    '_30sep2013': 'float',
    '_31dec2013': 'float',
    '_31mar2014': 'float',
    '_30jun2014': 'float',
    '_30sep2014': 'float',
    '_31dec2014': 'float',
    '_31mar2015': 'float',
    '_30jun2015': 'float',
    '_30sep2015': 'float',
    '_31dec2015': 'float',
    '_31mar2016': 'float',
    '_30jun2016': 'float',
    '_30sep2016': 'float',
    '_31dec2016': 'float',
    '_31mar2017': 'float',
    '_30jun2017': 'float',
    '_30sep2017': 'float',
    '_31dec2017': 'float',
    '_31mar2018': 'float',
    '_30jun2018': 'float',
    '_30sep2018': 'float',
    '_31dec2018': 'float',
    '_31mar2019': 'float',
    '_30jun2019': 'float',
    '_30sep2019': 'float',
    '_31dec2019': 'float',
    '_31mar2020': 'float',
    '_30jun2020': 'float',
    '_30sep2020': 'float',
    '_31dec2020': 'float',
    '_31mar2021': 'float',
    '_30jun2021': 'float',
    '_30sep2021': 'float',
    '_31dec2021': 'float',
    '_31mar2022': 'float',
    '_30jun2022': 'float',
    '_30sep2022': 'float'
})

# replace commas with dots
df = df.replace(',', '.', regex=True)


# In[7]:


# rename zonegéographiquederéférencerefer to geographicterminus
df.rename(columns={"zonegéographiquederéférencerefer": "geographicterminus"}, inplace=True)
df['geographicterminus'].rename("geographicterminus")

# replace geographicterminus values
df.loc[df['geographicterminus'] == 'FR', 'geographicterminus'] = 'france'

# rename zonedecontrepartiecounterpartare to geographicorigin
df.rename(columns={"zonedecontrepartiecounterpartare": "geographicorigin"}, inplace=True)
df['geographicorigin'].rename("geographicorigin")

# replace geographicorigin values
df.loc[df['geographicorigin'] == 'W1', 'geographicorigin'] = 'foreigneconomy'
df.loc[df['geographicorigin'] == 'W0', 'geographicorigin'] = 'totaleconomies'
df.loc[df['geographicorigin'] == 'W2', 'geographicorigin'] = 'domesticeconomy'


# rename columns
df = df.rename(columns={
    'secteurderéférencereferencesecto': 'domesticsectors',
    'secteurdecontrepartiecounterpart': 'foreignsectors',
    'paritéaccountentry': 'accountparity',
    'naturesto': 'flows',
    'instrumentinstrumentasset': 'instrument',
    'maturitématurity': 'maturity',
    'devisecurrency': 'currency'
})

# replace values
df['domesticsectors'].replace({
    'S1': 'totaldomesticeconomy',
    'S11': 'nonfinancialcorps',
    'S12': 'financialcorps',
    'S121': 'centralbank',
    'S122': 'deposittaking',
    'S123': 'moneymarket',
    'S124': 'nonmoneymarket',
    'S125': 'other',
    'S126': 'financialaux',
    'S127': 'captivefinancial',
    'S128': 'insurance',
    'S12K': 'monetaryinstitutions',
    'S12O': 'otherinstitutions',
    'S13': 'generalgovt',
    'S1311': 'centralgovt',
    'S13111': 'stategovt',
    'S13112': 'othercentralgovt',
    'S1313': 'localgvt',
    'S1314': 'socialsecurity',
    'S14': 'households',
    'S14A': 'employers',
    'S14B': 'employees',
    'S15': 'nonprofits',
    'S1M': 'NPISH'
}, inplace=True)

df['foreignsectors'].replace({
    'S1': 'totalforeigneconomy',
    'S11': 'nonfinancialcorps',
    'S12': 'financialcorps',
    'S122': 'deposittaking',
    'S124': 'nonMMFinvestment',
    'S128': 'insurance',
    'S12K': 'monetaryfinancial',
    'S12O': 'other',
    'S13': 'generalgovt',
    'S1M': 'NPISH'
}, inplace=True)

df['accountparity'].replace({
    'A': 'assets',
    'L': 'liabilities',
    'N': 'net'
}, inplace=True)

df['flows'].replace({
    'B102': 'changeassetvolume',
    'B103': 'networthreval',
    'B9F' : 'netlendborrow',  
    'BF90' : 'financialnet', 
    'F' : 'flows', 
    'KA' : 'other', 
    'LE' : 'stocks', 
    'K7' : 'revaluations' 
}, inplace=True)

# Rename columns
df = df.rename(columns={"instrumentinstrumentasset": "instrument", "maturitématurity": "maturity", "devisecurrency": "currency"})

# Recode instrument
df.loc[df["instrument"] == "F1", "instrument"] = "monetarygoldsdr"
df.loc[df["instrument"] == "F11", "instrument"] = "monetarygold"
df.loc[df["instrument"] == "F12", "instrument"] = "SDRs"
df.loc[df["instrument"] == "F2", "instrument"] = "currencydeposits"
df.loc[df["instrument"] == "F21", "instrument"] = "currency"

# Recode maturity
df.loc[df["maturity"] == "L", "maturity"] = "longterm"
df.loc[df["maturity"] == "S", "maturity"] = "shortterm"
df.loc[df["maturity"] == "T", "maturity"] = "allmaturities"
df.loc[df["maturity"] == "_Z", "maturity"] = "notapplicable"

# Recode currency
df.loc[df["currency"] == "_T", "currency"] = "all"
df.loc[df["currency"] == "EUR", "currency"] = "euro"
df.loc[df["currency"] == "X3", "currency"] = "noneuro"


# In[8]:


#Summing the data by decades 

df['total_90s'] = df.iloc[:, 0:16].sum(axis=1)
df['total_00s'] = df.iloc[:, 16:48].sum(axis=1)
df['total_10s'] = df.iloc[:, 48:80].sum(axis=1)
df['total_20s'] = df.iloc[:, 80:].sum(axis=1)


# In[9]:


print(df.head())


# In[10]:


# Sankey Diagram 0 will depict domestic total wealth over the decades, divided by sector

# 'domesticsectors', 'total_90s', 'total_00s', 'total_10s' , 'total_20s'

# Domestic Sectors : S14 Household, S11 Nonfinancial corporations, S12 financial corporations, S121 central bank, S128 Insurance, S1313 local government, S1311 central government, S13111 state government, S1 total


# In[11]:


# create a pivot table with 'code_sdmx', 'total_90s', 'total_00s', 'total_10s', 'total_20s', and 'domesticsectors' as columns
pivot_df = df.pivot_table(index='code_sdmx', columns='domesticsectors', values=['total_90s', 'total_00s', 'total_10s', 'total_20s'])

# flatten the multi-level column index
pivot_df.columns = [f'{col[0]}_{col[1]}' for col in pivot_df.columns]

# reset the index to create a new dataframe 'S0'
S0 = pivot_df.reset_index()

# display the new dataframe
print(S0)


# In[12]:


# create a list of keywords to filter by
keywords = ['nonfinancialcorps', 'financialcorps', 'centralbank', 'insurance', 'centralgovt', 'stategovt', 'localgvt', 'socialsecurity', 'households']

# select only the columns that contain one of the keywords
S0_filtered = S0[[col for col in S0.columns if any(keyword in col for keyword in keywords)]]

# display the new filtered dataframe
print(S0_filtered)



# In[13]:


# display a list of each column name in the dataframe
print(list(S0_filtered.columns))


# In[14]:


#Sankey 0.1 : 90s to 00s : domestic total wealth over the decades, divided by sector

import pandas as pd
import plotly.graph_objects as go

# Define the labels for each node
labels = ['total_00s_centralbank', 'total_00s_centralgovt', 'total_00s_financialcorps', 'total_00s_households', 'total_00s_insurance', 'total_00s_localgvt', 'total_00s_nonfinancialcorps', 'total_00s_othercentralgovt', 'total_00s_socialsecurity', 'total_00s_stategovt', 'total_90s_centralbank', 'total_90s_centralgovt', 'total_90s_financialcorps', 'total_90s_households', 'total_90s_insurance', 'total_90s_localgvt', 'total_90s_nonfinancialcorps', 'total_90s_othercentralgovt', 'total_90s_socialsecurity', 'total_90s_stategovt']

# Define the nodes
nodes = []
for label in labels:
    nodes.append(dict(label=label, pad=20))

# Define the source, target, and value for each link between nodes
source = [11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
target = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


value = [S0_filtered[label].sum() for label in labels]

# Create the Sankey diagram
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=20,
        thickness=20,
        label=labels
    ),
    link=dict(
        source=source,
        target=target,
        value=value
    )
)])

fig.update_layout(title="Domestic total wealth divided by sector: 90s to 00s")

fig.show()


# In[15]:


#Sankey 0.2  : 00s to 10s 

import pandas as pd

# Define the labels for each node
labels = ['total_00s_centralbank', 'total_00s_centralgovt', 'total_00s_financialcorps', 'total_00s_households', 'total_00s_insurance', 'total_00s_localgvt', 'total_00s_nonfinancialcorps', 'total_00s_othercentralgovt', 'total_00s_socialsecurity', 'total_00s_stategovt', 'total_10s_centralbank', 'total_10s_centralgovt', 'total_10s_financialcorps', 'total_10s_households', 'total_10s_insurance', 'total_10s_localgvt', 'total_10s_nonfinancialcorps', 'total_10s_othercentralgovt', 'total_10s_socialsecurity', 'total_10s_stategovt']

# Define the nodes
nodes = []
for label in labels:
    nodes.append(dict(label=label, pad=20))

# Define the source, target, and value for each link between nodes
source = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
target = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

value = [S0_filtered[label].sum() for label in labels]

# Create the Sankey diagram
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=20,
        thickness=20,
        label=labels
    ),
    link=dict(
        source=source,
        target=target,
        value=value
    )
)])
fig.update_layout(title="Domestic total wealth divided by sector: 00s to 10s")

fig.show()


# In[16]:


#Sankey 0.3 : 10s to 20s
import pandas as pd

# Define the labels for each node
labels = ['total_10s_centralbank', 'total_10s_centralgovt', 'total_10s_financialcorps', 'total_10s_households', 'total_10s_insurance', 'total_10s_localgvt', 'total_10s_nonfinancialcorps', 'total_10s_othercentralgovt', 'total_10s_socialsecurity', 'total_10s_stategovt', 'total_20s_centralbank', 'total_20s_centralgovt', 'total_20s_financialcorps', 'total_20s_households', 'total_20s_insurance', 'total_20s_localgvt', 'total_20s_nonfinancialcorps', 'total_20s_othercentralgovt', 'total_20s_socialsecurity', 'total_20s_stategovt']

# Define the nodes
nodes = []
for label in labels:
    nodes.append(dict(label=label, pad=20))

# Define the source, target, and value for each link between nodes
source = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
target = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

value = [S0_filtered[label].sum() for label in labels]

# Create the Sankey diagram
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=20,
        thickness=20,
        label=labels
    ),
    link=dict(
        source=source,
        target=target,
        value=value
    )
)])
fig.update_layout(title="Domestic total wealth divided by sector: 10s to 20s")

fig.show()


# In[17]:


# Sankey Diagram 1 will depict key foreign sectors compared to key domestic sectors. 

# More specifically, we will look at the total wealth in the 2010 decade, defined as 'total_10s' 
# This wealth is then divided by foreign or domestic sector. 
# The flow chart shows how wealth generated by each foreign sectors transfers in to wealth being generated by domestic sectors. 

# Foreign Sectors : S11 Nonfinancial corporations, S128 Insurance, S13 Governemnt, S1M households, S1 total 
# Domestic Sectors : S14 Household, S11 Nonfinancial corporations, S12 financial corporations, S121 central bank, S128 Insurance, S1313 local government, S1311 central government, S13111 state government, S1 total


# In[18]:


# Create Sankey diagram 1.1: Foreign to Domestic Industries
# This graph includes ALL Indusries 
# Using data from the 2010s
# The graph takes awhile to load! 
fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,
      thickness = 20,
      line = dict(color = "black", width = 0.5),
      label = df['foreignsectors'].drop_duplicates().tolist() + df['domesticsectors'].drop_duplicates().tolist(),
      color = "blue"
    ),
    link = dict(
      source = df['foreignsectors'].map(lambda x: df['foreignsectors'].drop_duplicates().tolist().index(x)+len(df['domesticsectors'].drop_duplicates())),
      target = df['domesticsectors'].map(lambda x: df['domesticsectors'].drop_duplicates().tolist().index(x)),
      value = df['total_10s']
  ))])

fig.update_layout(title_text="Foreign to Domestic Sectors", font=dict(size=12))
#pio.write_image(fig, 'Sankey_1_NoEdits.png')
fig.show()


# In[19]:


#Prep for Sankey 1.2

num_foreignsectors = df['foreignsectors'].nunique()
num_domesticsectors = df['domesticsectors'].nunique()

# Sum the numeric variable by each unique value in the string variable
sum_foreignsectors = df.groupby('foreignsectors')['total_10s'].sum()
sum_domesticsectors = df.groupby('domesticsectors')['total_10s'].sum()

print('Unique domestic sectors:', num_domesticsectors)
print('Unique foreign sectors:', num_foreignsectors)

print('\nTotal 10s by domestic sectors:\n', sum_domesticsectors.head())
print('\nTotal 10s by foreign sectors:\n', sum_foreignsectors.head())


# In[20]:


# Sankey diagram 1.2: Key Foreign to Domestic Industries
# This graph DROPS some industries & makes the graph much faster 
# Start by making a copy of the original dataframe
S1 = df.copy()
S1 = S1[['geographicorigin', 'foreignsectors', 'domesticsectors', 'code_sdmx', 'total_10s']]

# Create a new column named layer_1 with missing values
S1['layer_1'] = np.nan

# Drop rows where geographicorigin is equal to totaleconomies 
S1 = S1[~S1['geographicorigin'].isin(['totaleconomies'])]

#S1: 
    # foriegn : nonfinancialcorps , financialcorps, insurance, households, generalgovt
    # domesic: households, centralgvt, generalgovt, stategovt, localgvt, 'nonfinancialcorps', financialcorps', centralbank, insurance

domestic_sectors = ['households', 'centralgvt', 'generalgovt', 'stategovt', 'financialcorps', 'nonfinancialcorps', 'insurance', 'centralbank', 'localgvt']
foreign_sectors = ['nonfinancialcorps', 'financialcorps', 'insurance', 'households', 'generalgovt']

S1 = S1[S1['domesticsectors'].isin(domestic_sectors) & S1['foreignsectors'].isin(foreign_sectors)]


print(S1.head())


# In[21]:


num_foreignsectors = S1['foreignsectors'].nunique()
num_domesticsectors = S1['domesticsectors'].nunique()

# Sum the numeric variable by each unique value in the string variable
sum_foreignsectors = S1.groupby('foreignsectors')['total_10s'].sum()
sum_domesticsectors = S1.groupby('domesticsectors')['total_10s'].sum()

print('Unique domestic sectors:', num_domesticsectors)
print('Unique foreign sectors:', num_foreignsectors)

print('\nTotal 10s by domestic sectors:\n', sum_domesticsectors.head())
print('\nTotal 10s by foreign sectors:\n', sum_foreignsectors.head())


# In[22]:


# Pivot the dataframe to create columns for unique values in each string variable
for sector in S1['foreignsectors'].unique():
    S1['_for' + sector] = S1['total_10s'].where(S1['foreignsectors']==sector, 0)

for sector in S1['domesticsectors'].unique():
    S1['_dom' + sector] = S1['total_10s'].where(S1['domesticsectors']==sector, 0)

# Drop the original columns
S1.drop(columns=['foreignsectors', 'domesticsectors', 'total_10s'], inplace=True)

print(S1.head())




# In[23]:


import pandas as pd

# Define the labels for each node
labels = ['_fornonfinancialcorps', '_forfinancialcorps', '_forinsurance', '_forgeneralgovt', '_domnonfinancialcorps', '_domfinancialcorps', '_domcentralbank', '_dominsurance', '_domgeneralgovt', '_domstategovt', '_domlocalgvt', '_domhouseholds']

# Define the nodes
nodes = []
for label in labels:
    nodes.append(dict(label=label, pad=20))

# Define the source, target, and value for each link between nodes
source = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3]
target = [4, 5, 6, 7, 8, 9, 10, 11, 4, 5, 6, 7, 8, 9, 10, 11, 4, 5, 6, 7, 8, 9, 10, 11, 4, 5, 6, 7, 8, 9, 10, 11]
value = [S1['_fornonfinancialcorps'].sum(), S1['_forfinancialcorps'].sum(), S1['_forinsurance'].sum(), S1['_forgeneralgovt'].sum(), S1['_domnonfinancialcorps'].sum(), S1['_domfinancialcorps'].sum(), S1['_domcentralbank'].sum(), S1['_dominsurance'].sum(), S1['_domgeneralgovt'].sum(), S1['_domstategovt'].sum(), S1['_domlocalgvt'].sum(), S1['_domhouseholds'].sum()]
# Create the Sankey diagram
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=20,
        thickness=20,
        label=labels
    ),
    link=dict(
        source=source,
        target=target,
        value=value
    )
)])
fig.update_layout(title="Key Foreign to Domestic Industries")

fig.show()


# In[24]:


# Sankey Diagram 2:  Total Foreign Wealth to Assets/ Liabilities to Instruments 

# More specifically, we will look at the total wealth in the 2010s decade, defined as 'total_10s' 
# This wealth is summed to foreign origin only.  
# The flow chart shows how wealth generated in foreign sectors transfers is defined as an asset or liability and then if that asset or liabiility belongs to a tradeable asset class. 

# Foreign Sectors :  S1 total foreign sectors 
# Account parity : Asset or Liability 
# Instrument : F11 monetarygold, F12 SDRs, F2 currency deposit


# In[25]:


#Prep for Sankey 2.1 
# Count the number of unique values in each string variable
num_foreignsectors = df['foreignsectors'].nunique()
num_accountparity = df['accountparity'].nunique()
num_instrument = df['instrument'].nunique()

# Sum the numeric variable by each unique value in the string variable
sum_foreignsectors = df.groupby('foreignsectors')['total_10s'].sum()
sum_accountparity = df.groupby('accountparity')['total_10s'].sum()
sum_instrument = df.groupby('instrument')['total_10s'].sum()

print('Unique foreign sectors:', num_foreignsectors)
print('Unique account parity:', num_accountparity)
print('Unique instruments:', num_instrument)

print('\nTotal 10s by foreign sectors:\n', sum_foreignsectors.head())
print('\nTotal 10s by account parity:\n', sum_accountparity.head())
print('\nTotal 10s by instrument:\n', sum_instrument.head()) 


# In[26]:


# Create a new dataframe with selected columns
S2 = df[['code_sdmx', 'domesticsectors', 'foreignsectors', 'accountparity', 'instrument', 'total_10s']].copy()

# drop rows where 'accountparity' column equals "net"
S2.drop(S2[S2['accountparity'] == "net"].index, inplace=True)
print("Number of rows after dropping accountparity == net:", len(S2))

# drop rows where 'instrument' column does not match any of the given values
S2 = S2[(S2['instrument'] == 'monetarygoldsdr') |
        (S2['instrument'] == 'monetarygold') |
        (S2['instrument'] == 'SDRs') |
        (S2['instrument'] == 'currencydeposits') |
        (S2['instrument'] == 'currency')]
print("Number of rows after dropping instrument values:", len(S2))

# Reset the index of the dataframe
S2 = S2.reset_index()
print(S2.head())


# In[27]:


num_foreignsectors = S2['foreignsectors'].nunique()
num_accountparity = S2['accountparity'].nunique()
num_instrument = S2['instrument'].nunique()

# Sum the numeric variable by each unique value in the string variable
sum_foreignsectors = S2.groupby('foreignsectors')['total_10s'].sum()
sum_accountparity = S2.groupby('accountparity')['total_10s'].sum()
sum_instrument = S2.groupby('instrument')['total_10s'].sum()

print('Unique foreign sectors:', num_foreignsectors)
print('Unique account parity:', num_accountparity)
print('Unique instruments:', num_instrument)

print('\nTotal 10s by foreign sectors:\n', sum_foreignsectors.head())
print('\nTotal 10s by account parity:\n', sum_accountparity.head())
print('\nTotal 10s by instrument:\n', sum_instrument.head()) 


# In[28]:


# Pivot the dataframe to create columns for unique values in each string variable
for sector in S2['foreignsectors'].unique():
    S2[sector] = S2['total_10s'].where(S2['foreignsectors']==sector, 0)

    # create new columns for each unique value in 'accountparity'
for value in S2['accountparity'].unique():
    S2[f'AP_{value}'] = S2['total_10s'].where(S2['accountparity']==value)

# create new columns for each unique value in 'instrument'
for value in S2['instrument'].unique():
    S2[f'Inst_{value}'] = S2['total_10s'].where(S2['instrument']==value)

print(S2.head())


# In[29]:


import pandas as pd

# Define the labels for each node
labels = ['totalforeigneconomy', 'AP_assets', 'AP_liabilities', 'Inst_monetarygoldsdr', 'Inst_monetarygold', 'Inst_SDRs', 'Inst_currencydeposits', 'Inst_currency']

# Define the nodes
nodes = []
for label in labels:
    nodes.append(dict(label=label, pad=20))

# Define the source, target, and value for each link between nodes
source = [0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
target = [1, 2, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7]
value = [S2['totalforeigneconomy'].sum(), S2['AP_assets'].sum(), S2['AP_liabilities'].sum(), S2['Inst_monetarygoldsdr'].sum(), S2['Inst_monetarygold'].sum(), S2['Inst_SDRs'].sum(), S2['Inst_currencydeposits'].sum(), S2['Inst_currency'].sum()]

# Create the Sankey diagram
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=20,
        thickness=20,
        label=labels
    ),
    link=dict(
        source=source,
        target=target,
        value=value
    )
)])
fig.update_layout(title="Total Foreign Wealth to Assets/ Liabilities to Instruments")

fig.show()


# In[30]:


# This is Sankey 2.2 in which the foreign sectors are expanded upon. 

# Foreign Sectors : nonfinancialcorps  financialcorps  deposittaking, generalgovt 
# Account parity : Asset or Liability 
# Instruments : F11 monetarygold, F12 SDRs, F2 currency deposit, etc



# In[31]:


import pandas as pd

# Define the labels for each node
labels = ['nonfinancialcorps', 'financialcorps', 'deposittaking', 'generalgovt', 'AP_assets', 'AP_liabilities', 'Inst_monetarygoldsdr', 'Inst_monetarygold', 'Inst_SDRs', 'Inst_currencydeposits', 'Inst_currency']

# Define the nodes
nodes = []
for label in labels:
    nodes.append(dict(label=label, pad=20))

# Define the source, target, and value for each link between nodes
source = [0, 1, 2, 3, 0, 1, 2, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5]
target = [4, 4, 4, 4, 5, 5, 5, 5, 6, 7, 8, 9, 10, 6, 7, 8, 9, 10]
value = [S2['nonfinancialcorps'].sum(), S2['financialcorps'].sum(), S2['deposittaking'].sum(), S2['generalgovt'].sum(), S2['AP_assets'].sum(), S2['AP_liabilities'].sum(), S2['Inst_monetarygoldsdr'].sum(), S2['Inst_monetarygold'].sum(), S2['Inst_SDRs'].sum(), S2['Inst_currencydeposits'].sum(), S2['Inst_currency'].sum()]

# Create the Sankey diagram
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=20,
        thickness=20,
        label=labels
    ),
    link=dict(
        source=source,
        target=target,
        value=value
    )
)])
fig.update_layout(title="Total Foreign Sectors to Assets/ Liabilities to Instruments")

fig.show()


# In[32]:


#Prep for Sankey 2.3
# Here, I will repeat Sankey 2.2, but using domestic economy. 

# Count the number of unique values in each string variable
num_domesticsectors = df['domesticsectors'].nunique()
num_accountparity = df['accountparity'].nunique()
num_instrument = df['instrument'].nunique()

# Sum the numeric variable by each unique value in the string variable
sum_domesticsectors = df.groupby('domesticsectors')['total_10s'].sum()
sum_accountparity = df.groupby('accountparity')['total_10s'].sum()
sum_instrument = df.groupby('instrument')['total_10s'].sum()

print('Unique domestic sectors:', num_domesticsectors)
print('Unique account parity:', num_accountparity)
print('Unique instruments:', num_instrument)

print('\nTotal 10s by domestic sectors:\n', sum_domesticsectors.head())
print('\nTotal 10s by account parity:\n', sum_accountparity.head())
print('\nTotal 10s by instrument:\n', sum_instrument.head()) 


# In[33]:


# Create a new dataframe with selected columns
S2 = df[['code_sdmx', 'domesticsectors', 'foreignsectors', 'accountparity', 'instrument', 'total_10s']].copy()

# drop rows where 'accountparity' column equals "net"
S2.drop(S2[S2['accountparity'] == "net"].index, inplace=True)
print("Number of rows after dropping accountparity == net:", len(S2))

# drop rows where 'instrument' column does not match any of the given values
S2 = S2[(S2['instrument'] == 'monetarygoldsdr') |
        (S2['instrument'] == 'monetarygold') |
        (S2['instrument'] == 'SDRs') |
        (S2['instrument'] == 'currencydeposits') |
        (S2['instrument'] == 'currency')]
print("Number of rows after dropping instrument values:", len(S2))

# Reset the index of the dataframe
S2 = S2.reset_index()
print(S2.head())


# In[34]:


num_domesticsectors = S2['domesticsectors'].nunique()
num_foreignsectors = S2['foreignsectors'].nunique()
num_accountparity = S2['accountparity'].nunique()
num_instrument = S2['instrument'].nunique()

# Sum the numeric variable by each unique value in the string variable
sum_domesticsectors = S2.groupby('domesticsectors')['total_10s'].sum()
sum_foreignsectors = S2.groupby('foreignsectors')['total_10s'].sum()
sum_accountparity = S2.groupby('accountparity')['total_10s'].sum()
sum_instrument = S2.groupby('instrument')['total_10s'].sum()

print('Unique domestic sectors:', num_domesticsectors)
print('Unique foreign sectors:', num_foreignsectors)
print('Unique account parity:', num_accountparity)
print('Unique instruments:', num_instrument)

print('\nTotal 10s by domesic sectors:\n', sum_domesticsectors.head())
print('\nTotal 10s by foreign sectors:\n', sum_foreignsectors.head())
print('\nTotal 10s by account parity:\n', sum_accountparity.head())
print('\nTotal 10s by instrument:\n', sum_instrument.head()) 


# In[35]:


# Pivot the dataframe to create columns for unique values in each string variable
for sector in S2['domesticsectors'].unique():
    S2[sector] = S2['total_10s'].where(S2['domesticsectors']==sector, 0)
    
for sector in S2['foreignsectors'].unique():
    S2[sector] = S2['total_10s'].where(S2['foreignsectors']==sector, 0)

    # create new columns for each unique value in 'accountparity'
for value in S2['accountparity'].unique():
    S2[f'AP_{value}'] = S2['total_10s'].where(S2['accountparity']==value)

# create new columns for each unique value in 'instrument'
for value in S2['instrument'].unique():
    S2[f'Inst_{value}'] = S2['total_10s'].where(S2['instrument']==value)

print(S2.head())


# In[36]:


import plotly.graph_objects as go
import pandas as pd

# Define the labels for each node
labels = ['totaldomesticeconomy', 'AP_assets', 'AP_liabilities', 'Inst_monetarygoldsdr', 'Inst_monetarygold', 'Inst_SDRs', 'Inst_currencydeposits', 'Inst_currency']


# Define the nodes
nodes = []
for label in labels:
    nodes.append(dict(label=label, pad=20))

# Define the source, target, and value for each link between nodes
source = [0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
target = [1, 2, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7]
value = [S2['totaldomesticeconomy'].sum(), S2['AP_assets'].sum(), S2['AP_liabilities'].sum(), S2['Inst_monetarygoldsdr'].sum(), S2['Inst_monetarygold'].sum(), S2['Inst_SDRs'].sum(), S2['Inst_currencydeposits'].sum(), S2['Inst_currency'].sum()]

# Create the Sankey diagram
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=20,
        thickness=20,
        label=labels
    ),
    link=dict(
        source=source,
        target=target,
        value=value
    )
)])
fig.update_layout(title="Total Domestic Economy to Assets/ Liabilities to Instruments")

fig.show()


# In[37]:


import plotly.graph_objects as go
import pandas as pd

# Define the labels for each node
labels = ['totaldomesticeconomy', 'totalforeigneconomy', 'AP_assets', 'AP_liabilities', 'Inst_monetarygoldsdr', 'Inst_monetarygold', 'Inst_SDRs', 'Inst_currencydeposits', 'Inst_currency']

# Define the nodes
nodes = []
for label in labels:
    nodes.append(dict(label=label, pad=20))

# Define the source, target, and value for each link between nodes
source = [0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
target = [2, 3, 2, 3, 4, 5, 6, 7, 8, 9, 4, 5, 6, 7, 8, 9]
value = [S2['totaldomesticeconomy'].sum(), S2['totalforeigneconomy'].sum(), S2['AP_assets'].sum(), S2['AP_liabilities'].sum(), S2['Inst_monetarygoldsdr'].sum(), S2['Inst_monetarygold'].sum(), S2['Inst_SDRs'].sum(), S2['Inst_currencydeposits'].sum(), S2['Inst_currency'].sum()]

# Create the Sankey diagram
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=20,
        thickness=20,
        label=labels
    ),
    link=dict(
        source=source,
        target=target,
        value=value
    )
)])
fig.update_layout(title="Domestic versus Foreign Economy to Assets/ Liabilities to Instruments")

fig.show()


# In[38]:


# Sankey Diagram 3 : Long or shart maturity of the wealth flowing to the type of asset flowing to the type of instrumen being represented 
#These graphs depict the flow of all wealth, divided by the length of maturity, flowing to  to A/L and then to instruments


# In[39]:


# Goal 3: All Wealth's Maturity --> Assets or Liabilities --> Instrument and Asset Allocation
    #rename maturitématurity maturity
    #label variable maturity "maturity"
    #replace maturity = "longterm" if maturity == "L"
    #replace maturity = "shortterm" if maturity == "S"
    #replace maturity = "allmaturities" if maturity == "T"
    #replace maturity = "notapplicable" if maturity == "_Z"

    # rename paritéaccountentry accountparity
    #label variable accountparity "accountparity"
    #replace accountparity = "assets" if accountparity == "A"
    #replace accountparity = "liabiliies" if accountparity == "L"
    #replace accountparity = "net" if accountparity == "N"
    
    #rename instrumentinstrumentasset instrument
    #label variable instrument "instrument"
    #replace instrument = "monetarygoldsdr" if instrument == "F1" 
    #replace instrument = "monetarygold" if instrument == "F11"
    #replace instrument = "SDRs" if instrument == "F12"
    #replace instrument = "currencydeposits" if instrument == "F2"
    #replace instrument = "currency" if instrument == "F21"
    
   #  'maturity''accountparity' 'instrument'

#S3 : Change maturity to long/ short term 
#S3: Repeat with Foreign v Domestic to start 


# In[40]:


#Prep for Sankey 3
# Count the number of unique values in each string variable
num_maturity = df['maturity'].nunique()
num_accountparity = df['accountparity'].nunique()
num_instrument = df['instrument'].nunique()

# Sum the numeric variable by each unique value in the string variable
sum_maturity = df.groupby('maturity')['total_10s'].sum()
sum_accountparity = df.groupby('accountparity')['total_10s'].sum()
sum_instrument = df.groupby('instrument')['total_10s'].sum()

print('Unique maturities:', num_maturity)
print('Unique account parity:', num_accountparity)
print('Unique instruments:', num_instrument)

print('\nTotal 10s by maturity:\n', sum_maturity.head())
print('\nTotal 10s by account parity:\n', sum_accountparity.head())
print('\nTotal 10s by instrument:\n', sum_instrument.head())


# In[41]:


# Create a new dataframe with selected columns
S3 = df[['code_sdmx', 'maturity', 'accountparity', 'instrument', 'total_10s']]

# drop rows where 'instrument' column does not match any of the given values
S3 = S3[(S3['maturity'] == 'longterm') |
        (S3['maturity'] == 'shortterm') |
        (S3['maturity'] == 'notapplicable')]

# drop rows where 'instrument' column does not match any of the given values
#S3 = S3[(S3['instrument'] == 'monetarygoldsdr') |
#        (S3['instrument'] == 'monetarygold') |
#        (S3['instrument'] == 'SDRs') |
#        (S3['instrument'] == 'currencydeposits') |
#        (S3['instrument'] == 'currency')]

# Reset the index of the dataframe
S3 = S3.reset_index()
print(S3.head())


# In[42]:


print(S3.columns)


# In[43]:


num_maturity = S3['maturity'].nunique()
num_accountparity = S3['accountparity'].nunique()
num_instrument = S3['instrument'].nunique()

# Sum the numeric variable by each unique value in the string variable
sum_maturity = S3.groupby('maturity')['total_10s'].sum()
sum_accountparity = S3.groupby('accountparity')['total_10s'].sum()
sum_instrument = S3.groupby('instrument')['total_10s'].sum()

print('Unique maturities:', num_maturity)
print('Unique account parity:', num_accountparity)
print('Unique instruments:', num_instrument)

print('\nTotal 10s by maturity:\n', sum_maturity.head())
print('\nTotal 10s by account parity:\n', sum_accountparity.head())
print('\nTotal 10s by instrument:\n', sum_instrument.head())



# In[44]:


# Pivot the dataframe to create columns for unique values in each string variable
for sector in S3['maturity'].unique():
    S3[sector] = S3['total_10s'].where(S3['maturity']==sector, 0)

    # create new columns for each unique value in 'accountparity'
for value in S3['accountparity'].unique():
    S3[f'AP_{value}'] = S3['total_10s'].where(S3['accountparity']==value)

# create new columns for each unique value in 'instrument'
for value in S3['instrument'].unique():
    S3[f'Inst_{value}'] = S3['total_10s'].where(S3['instrument']==value)

print(S3.head())


# In[45]:


import plotly.graph_objects as go
import pandas as pd

# Define the labels for each node
labels = ['notapplicable', 'longterm', 'shortterm', 'AP_assets', 'AP_liabilities', 'Inst_monetarygoldsdr', 'Inst_monetarygold', 'Inst_SDRs']

# Define the nodes
nodes = []
for label in labels:
    nodes.append(dict(label=label, pad=20))

# Define the source, target, and value for each link between nodes
source = [0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4]
target = [3, 4, 3, 4, 3, 4, 5, 6, 7, 5, 6, 7]
values = [ S3['notapplicable'].sum(), S3['longterm'].sum(), S3['shortterm'].sum(), S3['AP_assets'].sum(), S3['AP_liabilities'].sum(), S3['Inst_monetarygoldsdr'].sum(), S3['Inst_monetarygold'].sum(), S3['Inst_SDRs'].sum()] 

# Create the Sankey diagram
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=20,
        thickness=20,
        label=labels
    ),
    link=dict(
        source=source,
        target=target,
        value=values
    )
)])
fig.update_layout(title="Wealth's Maturity to Assets/ Liabilities to Instruments")

fig.show()


# In[46]:


# Sankey 4 : A comparison between domestic versus foreign wealth which flows into 'naturesto' (or how the wealth is being used?)


# In[47]:


# Goal 4: RoW versus Domestic Wealth --> "Stocks, Transaction or other flows" --> Assets/ Liabilities

    #rename zonedecontrepartiecounterpartare geographicorigin
    #label variable geographicorigin "geographicorigin"
    #replace geographicorigin = "foreigneconomy" if geographicorigin == "W1"
    #replace geographicorigin = "totaleconomies" if geographicorigin == "W0"
    #replace geographicorigin = "domesticeconomy" if geographicorigin == "W2"
    #gen domwealth=sum(total_10s) if geographicorigin == "W2" 
    #gen forerignwealth=sum(total_10s) if geographicorigin == "W2" 

    #rename naturesto flows
    #label variable flows "flows"
    #replace flows = "changeassetvolume" if flows == "B102"
    #replace flows = "networthreval" if flows == "B103"
    #replace flows = "netlendborrow" if flows == "B9F"
    #replace flows = "financialnet" if flows == "BF90"
    #replace flows = "flows" if flows == "F"
  


# In[48]:


# create new columns and assign values based on the 'geographicorigin' column
df['domwealth'] = df['total_10s'][df['geographicorigin'] == 'domesticeconomy'].sum()
df['fornwealth'] = df['total_10s'][df['geographicorigin'] == 'foreigneconomy'].sum()


# In[49]:


# Prep for Sankey 4
# Count the number of unique values in each string variable
num_domwealth = df['domwealth'].nunique()
num_fornwealth = df['fornwealth'].nunique()
num_flows = df['flows'].nunique()
num_accountparity = df['accountparity'].nunique()

# Sum the numeric variable by each unique value in the string variable
sum_domwealth = df.groupby('domwealth')['total_10s'].sum()
sum_fornwealth = df.groupby('fornwealth')['total_10s'].sum()
sum_flows = df.groupby('flows')['total_10s'].sum()
sum_accountparity = df.groupby('accountparity')['total_10s'].sum()

print('Unique domestic wealth:', num_domwealth)
print('Unique foreign wealth:', num_fornwealth)
print('Unique flows:', num_flows)
print('Unique account parity:', num_accountparity)

print('\nTotal 10s by domestic wealth:\n', sum_domwealth.head())
print('\nTotal 10s by foreign wealth:\n', sum_fornwealth.head())
print('\nTotal 10s by flows:\n', sum_flows.head())
print('\nTotal 10s by account parity:\n', sum_accountparity.head())


# In[50]:


# Create a new dataframe with selected columns
S4 = df[['code_sdmx', 'domwealth', 'fornwealth', 'flows', 'accountparity', 'total_10s']]

# Reset the index of the dataframe
S4 = S4.reset_index()
print(S4.head())


# In[51]:


num_domwealth = S4['domwealth'].nunique()
num_fornwealth = S4['fornwealth'].nunique()
num_flows = S4['flows'].nunique()
num_accountparity = S4['accountparity'].nunique()

# Sum the numeric variable by each unique value in the string variable
sum_domwealth = S4.groupby('domwealth')['total_10s'].sum()
sum_fornwealth = S4.groupby('fornwealth')['total_10s'].sum()
sum_flows = S4.groupby('flows')['total_10s'].sum()
sum_accountparity = S4.groupby('accountparity')['total_10s'].sum()

print('Unique domesic wealth:', num_domwealth)
print('Unique foreign wealth:', num_fornwealth)
print('Unique flows:', num_flows)
print('Unique account parity:', num_accountparity)

print('\nTotal 10s by dom wealth:\n', sum_domwealth.head())
print('\nTotal 10s by forn wealth:\n', sum_fornwealth.head())
print('\nTotal 10s by flows:\n', sum_flows.head())
print('\nTotal 10s by account parity:\n', sum_accountparity.head())


# In[52]:


print(df['domwealth'].dtype)
print(df['fornwealth'].dtype)
print(df['flows'].dtype)
print(df['accountparity'].dtype)


# In[53]:


print(S4.columns)


# In[54]:


# Pivot the dataframe to create columns for unique values in each string variable

# create new columns for each unique value in 'flows'
for sector in S4['flows'].unique():
    S4[sector] = S4['total_10s'].where(S4['flows']==sector, 0)
       
# create new columns for each unique value in 'accountparity'
for sector in S4['accountparity'].unique():
    S4[sector] = S4['total_10s'].where(S4['accountparity']==sector, 0)

print(S4.head())


# In[55]:


import plotly.graph_objects as go
import pandas as pd

# Define the labels for each node
labels = ['domwealth', 'fornwealth', 'revaluations', 'other', 'stocks', 'changeassetvolume', 'networthreval', 'netlendborrow', 'financialnet', 'assets', 'liabilities', 'net']

# Define the nodes
nodes = []
for label in labels:
    nodes.append(dict(label=label, pad=20))

# Define the source, target, and value for each link between nodes
source = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8]
target = [2, 3, 4, 5, 6, 7, 8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 9, 10, 11, 9, 10, 11, 9, 10, 11, 9, 10, 11, 9, 10, 11, 9, 10, 11]
values = [S4['domwealth'].sum(), S4['fornwealth'].sum(), S4['revaluations'].sum(), S4['other'].sum(), S4['stocks'].sum(), S4['changeassetvolume'].sum(), S4['networthreval'].sum(), S4['netlendborrow'].sum(), S4['financialnet'].sum(), S4['assets'].sum(), S4['liabilities'].sum(), S4['net'].sum()]
# replace 'net' with the actual numeric value for the 'net' link:
values[11] = values[9] - values[10]

fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=20,
        thickness=20,
        label=labels
    ),
    link=dict(
        source=source,
        target=target,
        value=values
    )
)])
fig.update_layout(title="Foreign versus Domestic Wealth to Flows to Assets/ Liabilities")

fig.show()


# In[56]:


# Sankey 5: Domestic wealth is flowing into assets or liabilities and then into maturity status.
fig.update_layout(title="Domestic Wealth to Assets/ Liabilities to Maturity Status")


# In[57]:


# Goal 5: Domestic Wealth --> Assets or Liabilities --> Maturity 

    # 'domwealth' 'accountparity' 'maturity'
    
    # #rename zonedecontrepartiecounterpartare geographicorigin
    #label variable geographicorigin "geographicorigin"
    #replace geographicorigin = "foreigneconomy" if geographicorigin == "W1"
    #replace geographicorigin = "totaleconomies" if geographicorigin == "W0"
    #replace geographicorigin = "domesticeconomy" if geographicorigin == "W2"
    #gen domwealth=sum(total_10s) if geographicorigin == "W2" 
    #gen forerignwealth=sum(total_10s) if geographicorigin == "W2" 
    
     # rename paritéaccountentry accountparity
    #label variable accountparity "accountparity"
    #replace accountparity = "assets" if accountparity == "A"
    #replace accountparity = "liabiliies" if accountparity == "L"
    #replace accountparity = "net" if accountparity == "N"
    
    #rename maturitématurity maturity
    #label variable maturity "maturity"
    #replace maturity = "longterm" if maturity == "L"
    #replace maturity = "shortterm" if maturity == "S"
    #replace maturity = "allmaturities" if maturity == "T"
    #replace maturity = "notapplicable" if maturity == "_Z"


# In[58]:


# Prep for Sankey 5. 1
# Count the number of unique values in each string variable
num_domwealth = df['domwealth'].nunique()
num_accountparity = df['accountparity'].nunique()
num_maturity = df['maturity'].nunique()

# Sum the numeric variable by each unique value in the string variable
sum_domwealth = df.groupby('domwealth')['total_10s'].sum()
sum_accountparity = df.groupby('accountparity')['total_10s'].sum()
sum_maturity = df.groupby('maturity')['total_10s'].sum()

print('Unique domestic wealth:', num_domwealth)
print('Unique account parity:', num_accountparity)
print('Unique maturity:', num_maturity)

print('\nTotal 10s by domestic wealth:\n', sum_domwealth.head())
print('\nTotal 10s by account parity:\n', sum_accountparity.head())
print('\nTotal 10s by maturity:\n', sum_maturity.head())


# In[59]:


# Create a new dataframe with selected columns
S5 = df[['code_sdmx', 'domwealth', 'accountparity', 'maturity', 'total_10s']]

# Reset the index of the dataframe
S5 = S5.reset_index()
print(S5.head())


# In[60]:


print(S5.columns)


# In[61]:


num_domwealth = S5['domwealth'].nunique()
num_accountparity = S5['accountparity'].nunique()
num_maturity = S5['maturity'].nunique()

# Sum the numeric variable by each unique value in the string variable
sum_domwealth = S5.groupby('domwealth')['total_10s'].sum()
sum_accountparity = S5.groupby('accountparity')['total_10s'].sum()
sum_maturity = S5.groupby('maturity')['total_10s'].sum()


print('Unique domestic wealth:', num_domwealth)
print('Unique account parity:', num_accountparity)
print('Unique maturity:', num_maturity)

print('\nTotal 10s by domestic wealth:\n', sum_domwealth.head())
print('\nTotal 10s by account parity:\n', sum_accountparity.head())
print('\nTotal 10s by maturity:\n', sum_maturity.head())


# In[62]:


print(S5['domwealth'].dtype)
print(df['accountparity'].dtype)
print(df['maturity'].dtype)


# In[63]:


# Pivot the dataframe to create columns for unique values in each string variable

# create new columns for each unique value in 'flows'
for sector in S5['accountparity'].unique():
    S5[sector] = S5['total_10s'].where(S5['accountparity']==sector, 0)
       
# create new columns for each unique value in 'accountparity'
for sector in S5['maturity'].unique():
    S5[sector] = S5['total_10s'].where(S5['maturity']==sector, 0)

print(S5.head())


# In[64]:


import plotly.graph_objects as go
import pandas as pd

# Define the labels for each node
labels = ['domwealth', 'assets', 'liabilities', 'net', 'notapplicable', 'allmaturities', 'longterm', 'shortterm']

# Define the nodes
nodes = []
for label in labels:
    nodes.append(dict(label=label, pad=20))

# Define the source, target, and value for each link between nodes
source = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
target = [1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7]
values = [S5['domwealth'].sum(), S5['assets'].sum(), S5['liabilities'].sum(), S5['net'].sum(), S5['notapplicable'].sum(), S5['allmaturities'].sum(), S5['longterm'].sum(), S5['shortterm'].sum()] 

# Create Sankey diagram 
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=20,
        thickness=20,
        label=labels
    ),
    link=dict(
        source=source,
        target=target,
        value=values
    )
)])
fig.update_layout(title="Domestic Wealth to Assets/ Liabilities to Maturity Status")

fig.show()


# In[65]:


# Prep for Sankey 5.2
# Count the number of unique values in each string variable
num_fornwealth = df['fornwealth'].nunique()
num_accountparity = df['accountparity'].nunique()
num_maturity = df['maturity'].nunique()

# Sum the numeric variable by each unique value in the string variable
sum_fornwealth = df.groupby('fornwealth')['total_10s'].sum()
sum_accountparity = df.groupby('accountparity')['total_10s'].sum()
sum_maturity = df.groupby('maturity')['total_10s'].sum()

print('Unique foreign wealth:', num_fornwealth)
print('Unique account parity:', num_accountparity)
print('Unique maturity:', num_maturity)

print('\nTotal 10s by foreign wealth:\n', sum_fornwealth.head())
print('\nTotal 10s by account parity:\n', sum_accountparity.head())
print('\nTotal 10s by maturity:\n', sum_maturity.head())


# In[66]:


S5 = df[['code_sdmx', 'fornwealth', 'accountparity', 'maturity', 'total_10s']]

# Reset the index of the dataframe
S5 = S5.reset_index()
print(S5.head())


# In[67]:


print(S5.columns)


# In[68]:


num_fornwealth = S5['fornwealth'].nunique()
num_accountparity = S5['accountparity'].nunique()
num_maturity = S5['maturity'].nunique()

# Sum the numeric variable by each unique value in the string variable
sum_fornwealth = S5.groupby('fornwealth')['total_10s'].sum()
sum_accountparity = S5.groupby('accountparity')['total_10s'].sum()
sum_maturity = S5.groupby('maturity')['total_10s'].sum()


print('Unique foreign wealth:', num_fornwealth)
print('Unique account parity:', num_accountparity)
print('Unique maturity:', num_maturity)

print('\nTotal 10s by foreign wealth:\n', sum_fornwealth.head())
print('\nTotal 10s by account parity:\n', sum_accountparity.head())
print('\nTotal 10s by maturity:\n', sum_maturity.head())


# In[69]:


#All the graphs: 

#Finally: Make a word document with each graph and a brief explanation 

#Package into a .zip and send it 
    


# In[ ]:




