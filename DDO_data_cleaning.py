#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 16:26:55 2025
@author: dragon


Williamson County Regional Animal Shelter, in Georgetown, TX, runs a one day 
foster program called Doggy Day Out (DDO). This program allows people over 18 to 
choose a dog to take on an outing for the day. The shelter will sign the person 
up as a foster for that dog and only for that day. Then they take the dog on 
the outing. They can: Go to a park. Get a pup cup. Take a nap on the couch. 
And most importantly, HAVE FUN!!!¶

We will take data from Williamson County Regional Animal Shelter to assess the 
postive impact of this program. Are Doggy Day Out dogs apoted sooner? Does it 
help dogs who have been in shelter longer get adopted?


This code processes and analyzes animal shelter data from multiple CSV files 
related to dog intakes, outcomes, and Doggy Day Out. 
The goal is to clean, combine, and analyze the data to understand patterns 
like multiple shelter stays, length of stay, and frequency of DDO. 
The output can be then used for machine learning to see if DDO can predict 
the target or outcome for the dog



CSV Data Input Files:
        • Intakes - Combined_raw__AnimalIntakeExtended.csv
            – Represents the status of animals as they arrive at the animal shelter. All animals receive
            a unique Animal ID during intake.
            – 10,022 rows of data with 9,219 unique animal ids from 4/22-6/25
            – 690 or 7.48% of animal ids have repeat entries
            – Features of interest without missing data: Animal ID, Primary Breed, Gender, Primary
            Colour, Size, Age Group, Condition [EX: Normal, Appears Healthy, Thin, Injured],
            Intake Date/Time
        
        • Outcomes - Combined_raw__AnimalOutcomeExtended.csv
            – Represents the status of the animals as they leave the animal shelter. Were they adopted
            or reunited with the owner?
            – 9,851 rows of data with 9,078 unique animal ids from 4/22-6/25
            – 667 or 7.35% of animal ids have repeat entries
            – 653 animal ids have had multiple shelter stays that are seen in both Intakes and Outcomes
            – Features of interest without missing data: Animal ID, Primary Breed, Gender, Primary
            Colour, Age Group, Outcome Age As Month, Outcome YYYYMMDD
            – Target: Operation Type [Service Out, Return to Owner/Guardian, Transfer Out, Adop-
            tion, Euthanasia, Died, DOA, Clinic Out, Admin Missing]
        
        
        • Dog Day Out (DDO) - Combined_raw__DDOnumbers.csv
            – Represents data on dogs that were fostered for a day through the Doggy Day Out
            program.
            – 1,501 rows of data with 419 unique animal ids from 3/23-6/25
            – 245 or 58.47% of the Doggy Day Out Animal Ids are dogs that have gotten to go on
            multiple days out.
            – 72 dogs in the DDO have had multiple shelter stays.
            – Useful columns without missing data: Animal ID, Foster Start Date
            – Puppies, dogs under 12 months, can not participate in DDO

• Additional features created from working with the data
    – Count of DDO events
    – Count of repeated shelter stays
    – Time in the shelter in days

• Final Dataframes: One row represents an animal id’s stay in the shelter
    – 3,755 unique animal ids are puppies
    – 9,667 rows if keep one row per animal id per stay
    – 8,914 rows if keep one row per animal id with cumulative sum of shelter stay time over
    all shelter stays
    
    Output Data files:
        one row per animal id per stay:
                Output__Animal_ID_per_stay_n_ddo_cnt_df.csv
                Output__Animal_ID_per_stay_n_ddo_cnt_NO_PUPPIES_df.csv
            
        one row per animal id with cumulative sum of shelter stay time over:
                Output___per_Animal_ID_n_ddo_cnt_df.csv
                Output___per_Animal_ID_n_ddo_cnt_NO_PUPPIES_df.csv
                

"""

#package imports we need for this project
import pandas as pd # Pandas to work with our data
# import plotly.express as px # Plotly for visuliztion of our data
import matplotlib.pyplot as plt
from datetime import datetime # to clean up dates
# import warnings # for ingnoring warnings
# import re # for string parsing
# import os

# warnings.filterwarnings('ignore') # warnings will not be printed out

# import plotly.io as pio



# # -----------------------------------------------------------------------------
# # Merge data new data if have it ***
# # -----------------------------------------------------------------------------

# # My OG data I recived in 2024
# # ------------------------------
# outcomes_df = pd.read_csv('Data/w_o_4_24__AnimalOutcomeExtended (12).csv', encoding='latin-1')  # dog outcomes data
# intakes_df = pd.read_csv('Data/w_o_4_24__AnimalIntakeExtended (29).csv', encoding='latin-1')  # dog intake data
# ddo_df = pd.read_csv('Data/w_o_4_24__DDOnumbers.csv', encoding='latin-1')  # Doggy Day Out (DDO) data



# # ------------------------------
# # filter out incomplete month of 4/2024 in OG data 
#             # issues with data time saving correctly. tured out I needed to save excel files in CSV
# # ------------------------------


# # # outcomes_df
# # # ------------------------------

# # # print(outcomes_df.shape) # (6369, 29)

# # outcomes_df['Outcome YYYYMMDD'] = outcomes_df['Outcome YYYYMMDD']\
# #                                     .apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d')) 

# # outcomes_df = outcomes_df[~((outcomes_df['Outcome YYYYMMDD'].dt.month.astype('int64') == 4) &
# #                           (outcomes_df['Outcome YYYYMMDD'].dt.year.astype('int64') == 2024))]

# # # print(outcomes_df.shape) # (6253, 29)


# # # intakes_df
# # # ------------------------------
# # # print(intakes_df.shape) # (6558, 39)
 
# # intakes_df['Intake Date/Time'] = pd.to_datetime(intakes_df['Intake Date/Time']) # change to datetime

# # intakes_df = intakes_df[~((intakes_df['Intake Date/Time'].dt.month.astype('int64') == 4) &
# #                           (intakes_df['Intake Date/Time'].dt.year.astype('int64') == 2024))]

# # # print(intakes_df.shape) # (6415, 39)



# # # ddo_df
# # # ------------------------------
# # print(ddo_df.shape) # (561, 37)
 
# # ddo_df[['Foster Start Date', 'Foster End Date']] = \
# #         ddo_df[['Foster Start Date', 'Foster End Date']].apply(pd.to_datetime)

# # ddo_df = ddo_df[~((ddo_df['Foster End Date'].dt.month.astype('int64') == 4) &
# #                           (ddo_df['Foster End Date'].dt.year.astype('int64') == 2024))]

# # print(ddo_df.shape) # (528, 37)



# # New data!!
# # ------------------------------read_excel
# outcomes_df2 =pd.read_csv('Data/2025_AnimalOutcomeExtended (8).csv') # dog outcomes data
# intakes_df2 = pd.read_csv('Data/2025_AnimalIntakeExtended (26).csv')  # dog intake data
# ddo_df2 = pd.read_csv('Data/2025_FosterAnimalExtended (5).csv')  # Doggy Day Out (DDO) data



# # issues with data time saving correctly. tured out I needed to save excel files in CSV
# # ------------------------------
# # # outcomes_df
# # # ------------------------------
# # outcomes_df2['Outcome YYYYMMDD'] = outcomes_df2['Outcome YYYYMMDD']\
# #                                     .apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d')) 


# # # intakes_df
# # # ------------------------------
# # intakes_df2['Intake Date/Time'] = pd.to_datetime(intakes_df2['Intake Date/Time']) # change to datetime



# # # ddo_df
# # # ------------------------------
# # ddo_df2[['Foster Start Date', 'Foster End Date']] = \
# #         ddo_df2[['Foster Start Date', 'Foster End Date']].apply(pd.to_datetime)




# # # What are the diffrent cols?
# # [col for col in outcomes_df2.columns if col not in outcomes_df.columns]
# # [col for col in intakes_df2.columns if col not in intakes_df.columns]
# # [col for col in ddo_df.columns if col not in ddo_df2.columns]


# # Columns in both
# # ------------------------------
# outcomes_cols = list(set(outcomes_df.columns).intersection(set(outcomes_df2.columns)))
# intakes_cols = list(set(intakes_df.columns).intersection(set(intakes_df2.columns)))
# ddo_cols = list(set(ddo_df.columns).intersection(set(ddo_df2.columns)))


# # concat dfs
# # ------------------------------
# outcomes_df_combined = pd.concat([outcomes_df[outcomes_cols], outcomes_df2[outcomes_cols]], ignore_index=True)
# intakes_df_combined = pd.concat([intakes_df[intakes_cols], intakes_df2[intakes_cols]], ignore_index=True)
# ddo_df_combined = pd.concat([ddo_df[ddo_cols], ddo_df2[ddo_cols]], ignore_index=True)


# # Save files
# # ------------------------------
# outcomes_df_combined.to_csv('Data/Combined_raw__AnimalOutcomeExtended.csv', index=False)
# intakes_df_combined.to_csv('Data/Combined_raw__AnimalIntakeExtended.csv', index=False)
# ddo_df_combined.to_csv('Data/Combined_raw__DDOnumbers.csv', index=False)



# -----------------------------------------------------------------------------
# Read in our data ***
# -----------------------------------------------------------------------------
# Intakes represent the status of animals as they arrive at the Animal Center. 
        # All animals receive a unique Animal ID during intake.
# Outcomes represent the status of animals as they leave the Animal Center. 
        # Were they adopted? or Reunited with the owner?
# Dog Day Out (DDO) represent data about the dogs that were fostered for a day via Doggy Day Out
outcomes_df = pd.read_csv('Data/Combined_raw__AnimalOutcomeExtended.csv', encoding='latin-1')  # dog outcomes data
intakes_df = pd.read_csv('Data/Combined_raw__AnimalIntakeExtended.csv', encoding='latin-1')  # dog intake data
ddo_df = pd.read_csv('Data/Combined_raw__DDOnumbers.csv', encoding='latin-1')  # Doggy Day Out (DDO) data


print('\n\n')
print('Shape of animal outcomes dataframe:', outcomes_df.shape,'\n')
# use .info() to get info about DataFrame like index dtype and columns, non-null values and memory usage.
print(outcomes_df.info())

# Shape of animal outcomes dataframe: (9851, 29) 

# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 9851 entries, 0 to 9850
# Data columns (total 29 columns):
#  #   Column                Non-Null Count  Dtype  
# ---  ------                --------------  -----  
#  0   Animal ID             9851 non-null   object 
#  1   Jurisdiction Out      8408 non-null   object 
#  2   Outcome YYYYMMDD      9851 non-null   object 
#  3   Age                   9536 non-null   object 
#  4   ARN                   3367 non-null   object 
#  5   Pre Altered           9851 non-null   object 
#  6   Gender                9851 non-null   object 
#  7   Danger                9851 non-null   object 
#  8   Secondary Colour      6382 non-null   object 
#  9   Date Of Birth         9537 non-null   object 
#  10  Outcome Date Time     9851 non-null   object 
#  11  Operation Sub Type    9851 non-null   object 
#  12  Altered               9851 non-null   object 
#  13  Operation Type        9851 non-null   object 
#  14  Species               9851 non-null   object 
#  15  Pet ID Type           5972 non-null   object 
#  16  Primary Colour        9851 non-null   object 
#  17  Asilomar Status       2144 non-null   object 
#  18  Outcome Age As Month  9537 non-null   float64
#  19  Pet ID                5972 non-null   object 
#  20  Primary Breed         9851 non-null   object 
#  21  Danger Reason         5 non-null      object 
#  22  Outcome Reason        1840 non-null   object 
#  23  Microchip Number      9507 non-null   object 
#  24  Secondary Breed       9847 non-null   object 
#  25  Animal Name           9833 non-null   object 
#  26  Age Group             9850 non-null   object 
#  27  Spayed Neutered       9851 non-null   object 
#  28  Site Name             9851 non-null   object 
# dtypes: float64(1), object(28)
# memory usage: 2.2+ MB


print('\n\n')
print('Shape of animal intake dataframe:', intakes_df.shape,'\n')
# use .info() to get info about DataFrame like index dtype and columns, non-null values and memory usage.
print(intakes_df.info())

# Shape of animal intake dataframe: (10022, 39) 

# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 10022 entries, 0 to 10021
# Data columns (total 39 columns):
#  #   Column                 Non-Null Count  Dtype  
# ---  ------                 --------------  -----  
#  0   Animal ID              10022 non-null  object 
#  1   Source                 1773 non-null   object 
#  2   Location Found         8236 non-null   object 
#  3   Intake Reason          1773 non-null   object 
#  4   Age                    9694 non-null   object 
#  5   Pre Altered            10022 non-null  object 
#  6   Jurisdiction           10016 non-null  object 
#  7   Condition              10021 non-null  object 
#  8   Location               10020 non-null  object 
#  9   Gender                 10022 non-null  object 
#  10  Danger                 10022 non-null  object 
#  11  Secondary Colour       6470 non-null   object 
#  12  Second Colour Pattern  2 non-null      object 
#  13  Sub Location           10020 non-null  object 
#  14  Operation Sub Type     10022 non-null  object 
#  15  Altered                10022 non-null  object 
#  16  Operation Type         10022 non-null  object 
#  17  Agency Name            4118 non-null   object 
#  18  Species                10022 non-null  object 
#  19  Colour Pattern         630 non-null    object 
#  20  Injury Type            0 non-null      float64
#  21  Pet ID Type            0 non-null      float64
#  22  Unit                   1223 non-null   object 
#  23  Primary Colour         10022 non-null  object 
#  24  Intake Date/Time       10022 non-null  object 
#  25  Third Colour           371 non-null    object 
#  26  Cause                  0 non-null      float64
#  27  Asilomar Status        2084 non-null   object 
#  28  Size                   10022 non-null  object 
#  29  DOA                    10022 non-null  bool   
#  30  Pet ID                 0 non-null      float64
#  31  Primary Breed          10022 non-null  object 
#  32  Danger Reason          5 non-null      object 
#  33  Length Owned           9930 non-null   float64
#  34  Secondary Breed        10018 non-null  object 
#  35  Animal Name            10003 non-null  object 
#  36  Age Group              10021 non-null  object 
#  37  Spayed Neutered        10022 non-null  object 
#  38  Site Name              10022 non-null  object 
# dtypes: bool(1), float64(5), object(33)
# memory usage: 2.9+ MB



print('\n\n')
print('Shape of Doggy Day Out dataframe:', ddo_df.shape,'\n')
# use .info() to get info about DataFrame like index dtype and columns, non-null values and memory usage.
ddo_df.info()


# Shape of Doggy Day Out dataframe: (1501, 35) 

# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 1501 entries, 0 to 1500
# Data columns (total 35 columns):
#  #   Column                                Non-Null Count  Dtype  
# ---  ------                                --------------  -----  
#  0   Animal #                              1501 non-null   object 
#  1   Current Altered                       1501 non-null   object 
#  2   Current Age                           1500 non-null   object 
#  3   Animal Species                        1501 non-null   object 
#  4   Outcome Subtype                       1150 non-null   object 
#  5   Foster End Date                       1501 non-null   object 
#  6   Intake Condition                      0 non-null      float64
#  7   Outcome Date                          1155 non-null   object 
#  8   Gender                                1501 non-null   object 
#  9   Foster End Reason                     349 non-null    object 
#  10  Intake Date                           1501 non-null   object 
#  11  Outcome Jurisdiction                  1032 non-null   object 
#  12  Animal Type                           1501 non-null   object 
#  13  Outcome Site                          1155 non-null   object 
#  14  Foster Start Status                   1501 non-null   object 
#  15  # of Foster Visits                    1501 non-null   int64  
#  16  Foster End Status                     1501 non-null   object 
#  17  Outcome Asilomar Status               272 non-null    object 
#  18  Intake Site                           1501 non-null   object 
#  19  Primary / Secondary / Tertiary Color  1501 non-null   object 
#  20  Intake Subtype                        1501 non-null   object 
#  21  Intake Asilomar Status                390 non-null    object 
#  22  Intake Type                           1501 non-null   object 
#  23  Primary / Secondary Breed             1501 non-null   object 
#  24  Foster Start Site                     1501 non-null   object 
#  25  Foster Record #                       1501 non-null   object 
#  26  Pre-Altered                           1501 non-null   object 
#  27  Outcome Type                          1155 non-null   object 
#  28  Outcome Condition                     0 non-null      float64
#  29  Foster Start Reason                   1501 non-null   object 
#  30  Intake Jurisdiction                   1501 non-null   object 
#  31  Foster Start Date                     1501 non-null   object 
#  32  Animal Name                           1501 non-null   object 
#  33  Foster End Site                       1501 non-null   object 
#  34  Color Pattern 1 / 2                   117 non-null    object 
# dtypes: float64(2), int64(1), object(32)
# memory usage: 410.6+ KB




# -----------------------------------------------------------------------------
# Cleanup Our Data ****
# -----------------------------------------------------------------------------
# drop na How = ‘all’ meaning If all values in row are NA, drop that row
outcomes_df = outcomes_df.dropna(how='all') 
intakes_df = intakes_df.dropna(how='all') # intakes_df.shape = (6558, 39) but no col had more then 6558 non-null 
ddo_df = ddo_df.dropna(how='all') # ddo_df.shape = (561, 37) but no col had more then 561 non-null


# ---------------------------------------------
# Clean date cols
# ---------------------------------------------
# start with the date fields b/c they all object Dtypes

# Let's clean up the date columns! We want them to be datetime!
# We are also going to create a few columns for month and year in outcomes_df and intakes_df 
# for later data manipulation

# Clean up on outcomes_df: 
outcomes_df['Outcome Date Time'] = pd.to_datetime(outcomes_df['Outcome Date Time']) # change to datetime
outcomes_df['Outcome YYYYMMDD'] = outcomes_df['Outcome YYYYMMDD']\
                                    .apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d')) # change to datetime

# outcomes_df[['Outcome Date Time', 'Outcome YYYYMMDD']] = \
#         outcomes_df[['Outcome Date Time','Outcome YYYYMMDD']].apply(pd.to_datetime) 
outcomes_df['Outcome YR'] = outcomes_df['Outcome YYYYMMDD'].dt.year.astype('int64')
outcomes_df['Outcome Month'] = outcomes_df['Outcome YYYYMMDD'].dt.month.astype('int64')

# print('\n\n','Cleaned Outcomes Date Fields:')
# print('----------------------------')
# print(outcomes_df[['Outcome Date Time','Outcome YYYYMMDD','Outcome YR','Outcome Month']].dtypes)
# print(outcomes_df[['Outcome Date Time','Outcome YYYYMMDD','Outcome YR','Outcome Month']].head(3)) # print out results of our hard work



# Clean up on intakes_df: 
intakes_df[['Intake Date/Time']] = \
        intakes_df[['Intake Date/Time']].apply(pd.to_datetime) 
intakes_df['Intake YR'] = intakes_df['Intake Date/Time'].dt.year.astype('int64')
intakes_df['Intake Month'] = intakes_df['Intake Date/Time'].dt.month.astype('int64')

# print('\n\n','Cleaned Intakes Date Fields:')
# print('----------------------------')
# print(intakes_df[['Intake Date/Time','Intake YR','Intake Month']].dtypes)
# print(intakes_df[['Intake Date/Time','Intake YR','Intake Month']].head(3)) # print out results of our hard work



# Clean up on ddo_df: 
ddo_df[['Foster Start Date','Foster End Date','Intake Date','Outcome Date']] = \
        ddo_df[['Foster Start Date','Foster End Date','Intake Date','Outcome Date']].apply(pd.to_datetime) # change to datetime
ddo_df['Fost St YR'] = ddo_df['Foster Start Date'].dt.year.astype('int64')
ddo_df['Fost St Month'] = ddo_df['Foster Start Date'].dt.month.astype('int64')
    
# print('\n\n','Cleaned Doggy Day Out Date Fields:')
# print('----------------------------')
# print(ddo_df[['Foster Start Date','Foster End Date','Intake Date','Outcome Date','Fost St YR','Fost St Month']].dtypes)
# print(ddo_df[['Foster Start Date','Foster End Date','Intake Date','Outcome Date','Fost St YR','Fost St Month']].head(3)) # print out results of our hard work



# --------------------------------
# plot of DDO over time by month
# --------------------------------
# ddo_df['Year-Month'] = ddo_df['Foster Start Date'].dt.to_period('M')

# # number of rows per month
# monthly_counts = ddo_df.groupby('Year-Month').size()

# # Plot
# plt.figure(figsize=(12,6))
# ax = plt.gca()
# monthly_counts.plot(kind='line', marker='o')
# plt.title('Count of DDOs by Date', fontsize=20)
# plt.xlabel('DDO Date', fontsize=16)
# plt.ylabel('Count', fontsize=16)
# plt.xticks(rotation=45, fontsize=14)
# ax.tick_params(axis='x', which='minor', labelsize=14) 
# plt.yticks(fontsize=14)
# plt.tight_layout()
# plt.show()


# ---------------------------------------------
# Cleanup Our Data: ddo age
# ---------------------------------------------

# print(ddo_df['Current Age'].head())
# 0    10y 3m 20d
# 1    10y 3m 20d
# 2    10y 3m 20d
# 3      9y 7m 5d
# 4      9y 7m 5d
# Name: Current Age, dtype: object


# We can use Regex to capture those values!!! 
age = (
    ddo_df['Current Age']
    .str.extract(r"(?:(?P<y>\d+)y?)?\s*(?:(?P<m>\d+)m?)?\s*(?:(?P<d>\d+)d?)?")
    .astype(float)
    .fillna(0)
)

ddo_df['DDO_Age_As_Month'] = (age["y"] * 12 + age["m"] + age["d"]/30) # Let's convert age to months! 
ddo_df['DDO_Age_As_Month'] = round(ddo_df['DDO_Age_As_Month'], 1) # Round the decimals

# print(ddo_df[['Current Age','DDO_Age_As_Month']].head()) # Check Results
#   Current Age  DDO_Age_As_Month
# 0  10y 3m 20d             123.7
# 1  10y 3m 20d             123.7
# 2  10y 3m 20d             123.7
# 3    9y 7m 5d             115.2
# 4    9y 7m 5d             115.2

# min age of DDO participant?  Puppies, less than 12 month, can't do DDO
min_ddo_age = ddo_df['DDO_Age_As_Month'].min() 
print('\n\n')
print(f"Lowest age doggie day out participant: {min_ddo_age}")
# Lowest age doggie day out participant: 13.0


# -----------------------------------------------------------------------------
# Data Investigation: Dogs that had Multiple Shelter Stays ***
# -----------------------------------------------------------------------------
# Next, let's understand how many duplicate dogs we have in our data. 
# Some animals will come back to the shelter for multiple reasons and will 
# therefore have multiple stays in the shelter.

# look at unique Animal # accross our diffrent data sets

print('\n\n','Dogs that had Multiple Shelter Stays')
print('--------------------------------------------------------')


print('\n\n','Outcomes Animal ID:') 
print('----------------------------')
# print(outcomes_df.columns)

num_of_outcomes_ids_seen_mult = sum(outcomes_df['Animal ID'].value_counts()> 1)
num_unique_outcome_ids = outcomes_df['Animal ID'].nunique()
percent_mult_outcome_ids = (num_of_outcomes_ids_seen_mult / num_unique_outcome_ids) * 100
print('Shape of animal outcomes dataframe:', outcomes_df.shape)
print('Number of Animal IDs seen multiple times in df: ', num_of_outcomes_ids_seen_mult)
print('Number of unique Animal IDs seen in df: ', num_unique_outcome_ids)
print('% of Animal IDs seen multiple times in df: ', "%.2f" % percent_mult_outcome_ids,'%')

#  Outcomes Animal ID:
# ----------------------------
# Number of Animal IDs seen multiple times in df:  414
# Number of unique Animal IDs seen in df:  5883
# % of Animal IDs seen multiple times in df:  7.04 %



print('\n\n','Intakes Animal ID:')
print('----------------------------')
# print(intakes_df.columns)

num_of_intakes_ids_seen_mult = sum(intakes_df['Animal ID'].value_counts()> 1)
num_unique_intakes_ids = intakes_df['Animal ID'].nunique()
percent_mult_intakes_ids = (num_of_intakes_ids_seen_mult / num_unique_intakes_ids) * 100
print('Shape of animal intake dataframe:', intakes_df.shape)
print('Number of Animal IDs seen multiple times in df: ', num_of_intakes_ids_seen_mult)
print('Number of unique Animal IDs seen in df: ', num_unique_intakes_ids)
print('% of Animal IDs seen multiple times in df: ', "%.2f" % percent_mult_intakes_ids,'%')

#  Intakes Animal ID:
# ----------------------------
# Number of Animal IDs seen multiple times in df:  442
# Number of unique Animal IDs seen in df:  6046
# % of Animal IDs seen multiple times in df:  7.31 %


    
print('\n\n','Doggy Day Out Animal ID:')
print('----------------------------')
# print(ddo_df.columns)

num_of_ddo_ids_seen_mult = sum(ddo_df['Animal #'].value_counts()> 1)
num_unique_ddo_ids = ddo_df['Animal #'].nunique()
percent_mult_ddo_ids = (num_of_ddo_ids_seen_mult / num_unique_ddo_ids) * 100
print('Shape of Doggy Day Out dataframe:', ddo_df.shape)
print('Number of Animal IDs seen multiple times in df: ', num_of_ddo_ids_seen_mult)
print('Number of unique Animal IDs seen in df: ', num_unique_ddo_ids)
print('% of Animal IDs seen multiple times in df: ', "%.2f" % percent_mult_ddo_ids,'%')

#  Doggy Day Out Animal ID:
# ----------------------------
# Number of Animal IDs seen multiple times in df:  111
# Number of unique Animal IDs seen in df:  190
# % of Animal IDs seen multiple times in df:  58.42 %


# Find the 'Animal ID' that had multiple stays from our intake data:
intake_mult_stays_ids = intakes_df['Animal ID'].value_counts()[intakes_df['Animal ID'].value_counts()>1].reset_index()['Animal ID']


# Find the 'Animal ID' that had multiple stays from our Outcome data:
outcome_mult_stays_ids = outcomes_df['Animal ID'].value_counts()[outcomes_df['Animal ID'].value_counts()>1].reset_index()['Animal ID']


# get a unquie list of multiple stays 'Animal ID' by creating a common set
mult_stays_ids_set = set(intake_mult_stays_ids).intersection(set(outcome_mult_stays_ids))


# print out results:
print('\n\n')
print('How many unique Dogs have had multiple stays from both the Outcomes and Intake data:', 
      len(mult_stays_ids_set),'\n')

print('How many unique Dogs in the Doggy Day Out data set have had multiple stays:',
      sum(ddo_df['Animal #'].drop_duplicates().isin(mult_stays_ids_set)))

# How many unique Dogs have had multiple stays from both the Outcomes and Intake data: 401 

# How many unique Dogs in the Doggy Day Out data set have had multiple stays: 33


# -----------------------------------------------------------------------------
# Get count and length of stays for the dogs ***
# -----------------------------------------------------------------------------
# We will start by getting the below dataframes:
    # outcome with columns 'Animal ID','Outcome Date Time','Operation Type'
    # Intake with columns 'Animal ID','Intake Date/Time'
    
# We will the concatenate the two dataframe and sort them date and Animal ID.
    #  We will use a cumulative sum and a groupby to number the stays for the dogs 
    #  that have had multiple stays in the shelter.
# -----------------------------------------------------------------------------


# --------------------------------
# Intakes
# --------------------------------
# For the intake data, we will create a column of 'Outcome/Intake_0_1' 
        # filled with 1's. This is because every time a dog is taken in, 
        # that will count as a new stay. Keep following along to see how 
        # the cumulative sum will use this to number stays.

# get intake data ready to concatenate with outcome data
intakes_for_stay_length_df = intakes_df[['Animal ID','Intake Date/Time', 'Condition', 'Size']]# Col I'm intrested in  from intakes_df
intakes_for_stay_length_df = intakes_for_stay_length_df.rename(columns={"Intake Date/Time": "Date/Time"})
intakes_for_stay_length_df['Outcome/Intake'] = 'Intake' # label as Intake data point
intakes_for_stay_length_df['Outcome/Intake_0_1'] = 1 # col need for Cumsum. Every intake will count as a new stay
# print(intakes_for_stay_length_df.head()) # let's see what we got! 

#      Animal ID           Date/Time Outcome/Intake  Outcome/Intake_0_1
# 0  A0009297150 2022-07-26 13:43:00         Intake                   1
# 1  A0011526731 2022-12-26 17:44:00         Intake                   1
# 2  A0014797550 2023-01-11 14:04:00         Intake                   1
# 3  A0015632857 2022-08-10 17:43:00         Intake                   1
# 4  A0015633054 2023-11-16 15:00:00         Intake                   1


# --------------------------------
# Outcomes - pick out come col here ****
# --------------------------------
# For the outcome data, we will create a column of 'Outcome/Intake_0_1' 
# but filled with 0's. We do not want to increase the cumulative sum when an 
# outcome happens. Keep following along to see how the cumulative sum will use this to number stays.

# get outcome data ready to concatenate with intake data
outcome_col_I_want = ['Animal ID','Outcome Date Time' #
                      ,'Primary Breed', 'Gender', 'Primary Colour' 
                      ,'Age Group', 'Outcome Age As Month', 'Operation Type'
                      ] 
outcome_for_stay_length_df = outcomes_df[outcome_col_I_want] 
outcome_for_stay_length_df = outcome_for_stay_length_df.rename(columns={"Outcome Date Time": "Date/Time"})
outcome_for_stay_length_df['Outcome/Intake'] = 'Outcome' # label as Outcome data point
outcome_for_stay_length_df['Outcome/Intake_0_1'] = 0 # col need for Cusum

# print('Results of concatenating intake and Outcomes data: ')
# print(outcome_for_stay_length_df.head()) # let's see what we got! 

#      Animal ID           Date/Time  ... Outcome/Intake  Outcome/Intake_0_1
# 0  A0009297150 2022-07-26 13:48:00  ...        Outcome                   0
# 1  A0011526731 2022-12-30 12:37:00  ...        Outcome                   0
# 2  A0014439605 2023-06-03 15:26:00  ...        Outcome                   0
# 3  A0014797550 2023-01-12 17:11:00  ...        Outcome                   0
# 4  A0015632857 2022-08-10 17:47:00  ...        Outcome                   0



# --------------------------------
# Concatenate the tables together and sort them by 'Animal ID' and 'Date/Time' columns .
# --------------------------------

# concatenate and sort the data by 'Animal ID' and 'Date/Time' columns 
#       The sorting my date is VERY important for our cumulative sum to work correctly
per_stay_df = pd.concat([outcome_for_stay_length_df,intakes_for_stay_length_df])
per_stay_df = per_stay_df.sort_values(by=['Animal ID','Date/Time'], ascending=[True,True])




# We have dogs who have had multiple stays in the shelter and we need to number their stays for our analysis
#       let's look at the data for dogs who had multiple stays at the shelter
#       NOTE: Only filtering here for multiple stays dogs to understand cumulative sum process
# print('Intake and Outcomes Concatenate Data, Sorted by Date, Filtered for Multiple Stay Dogs: ')
# print(per_stay_df[per_stay_df['Animal ID'].isin(mult_stays_ids_set)].head(10))

# Intake and Outcomes Concatenate Data, Sorted by Date, Filtered for Multiple Stay Dogs: 
#         Animal ID           Date/Time Outcome/Intake  Outcome/Intake_0_1
# 1702  A0028569416 2022-11-04 12:36:00         Intake                   1
# 32    A0028569416 2022-11-04 16:01:00        Outcome                   0
# 1676  A0028569416 2023-04-18 13:11:00         Intake                   1
# 33    A0028569416 2023-12-21 15:38:00        Outcome                   0
# 1698  A0033526535 2022-05-17 14:15:00         Intake                   1
# 49    A0033526535 2022-05-18 14:24:00        Outcome                   0
# 1684  A0033526535 2022-10-10 08:55:00         Intake                   1
# 50    A0033526535 2022-10-10 16:31:00        Outcome                   0
# 1696  A0033526535 2022-11-13 13:24:00         Intake                   1
# 51    A0033526535 2022-12-29 12:05:00        Outcome                   0


# --------------------------------
# Shift Condition and Size from intake animel id row to outcome row b/c we only want to keep outcome rows
# --------------------------------
per_stay_df['Condition_shifted'] = per_stay_df.groupby('Animal ID')['Condition'].shift(1)
per_stay_df['Size_shifted'] = per_stay_df.groupby('Animal ID')['Size'].shift(1)



# --------------------------------
# 'Stay_Number' >>> cumulative sum!!!!
# --------------------------------
# We use a group by 'Animal ID' and do a cumulative sum and save in in col 'Stay_Number'
#       cumulative sum is used to display the total sum of data as it grows with time
per_stay_df['Stay_Number'] = per_stay_df\
                                        .groupby(by=['Animal ID'])['Outcome/Intake_0_1'].cumsum()


# We have dogs who have had multiple stays in the shelter and we need to number their stays for our analysis
#       let's look at the data for dogs who had multiple stays at the shelter
#       now we can see how the cumulative sum has numbered the stay for returning dogs
#       NOTE: Only filtering here for multiple stays dogs to understand cumulative sum process
# print('Intake and Outcomes Concatenate Data with Stay Count, Filtered for Multiple Stay Dogs: ')
# print(per_stay_df[per_stay_df['Animal ID'].isin(mult_stays_ids_set)].head(10))

# Intake and Outcomes Concatenate Data with Stay Count, Filtered for Multiple Stay Dogs: 
#         Animal ID           Date/Time  ... Outcome/Intake_0_1  Stay_Number
# 1702  A0028569416 2022-11-04 12:36:00  ...                  1            1
# 32    A0028569416 2022-11-04 16:01:00  ...                  0            1
# 1676  A0028569416 2023-04-18 13:11:00  ...                  1            2
# 33    A0028569416 2023-12-21 15:38:00  ...                  0            2
# 1698  A0033526535 2022-05-17 14:15:00  ...                  1            1
# 49    A0033526535 2022-05-18 14:24:00  ...                  0            1
# 1684  A0033526535 2022-10-10 08:55:00  ...                  1            2
# 50    A0033526535 2022-10-10 16:31:00  ...                  0            2
# 1696  A0033526535 2022-11-13 13:24:00  ...                  1            3
# 51    A0033526535 2022-12-29 12:05:00  ...                  0            3


# --------------------------------
# Calculate stay length!
# --------------------------------
# calculate the time difference between consecutive rows
per_stay_df['stay_time_length'] = per_stay_df\
                                            .groupby(by =['Animal ID','Stay_Number'])['Date/Time'].diff()
                                            
                                            
# Creating a length of stay column in days 

per_stay_df['stay_time_length_days'] = \
                    per_stay_df['stay_time_length'].values.astype(int) # Times in nanoseconds

per_stay_df['stay_time_length_days'] = \
                    per_stay_df['stay_time_length_days'] * 1.15741e-14 # Convert to days
                       

                                            
# display(find_per_stay_df.head(10))

# We have animal who have multiple stays in the shelter and we need to number their stays for our analysis
#       NOTE: Only filtering here for multiple stays dogs to check our process
# print('Intake and Outcomes Concatenat Data with Stay length, Filtered for Multiple Stay Dogs: ')
# print(per_stay_df[per_stay_df['Animal ID'].isin(mult_stays_ids_set)].head(10))

# Intake and Outcomes Concatenat Data with Stay length, Filtered for Multiple Stay Dogs: 
#         Animal ID           Date/Time  ...  stay_time_length  stay_time_length_days
# 1702  A0028569416 2022-11-04 12:36:00  ...               NaT         -106752.230292
# 32    A0028569416 2022-11-04 16:01:00  ...   0 days 03:25:00               0.142361
# 1676  A0028569416 2023-04-18 13:11:00  ...               NaT         -106752.230292
# 33    A0028569416 2023-12-21 15:38:00  ... 247 days 02:27:00             247.102637
# 1698  A0033526535 2022-05-17 14:15:00  ...               NaT         -106752.230292
# 49    A0033526535 2022-05-18 14:24:00  ...   1 days 00:09:00               1.006252
# 1684  A0033526535 2022-10-10 08:55:00  ...               NaT         -106752.230292
# 50    A0033526535 2022-10-10 16:31:00  ...   0 days 07:36:00               0.316667
# 1696  A0033526535 2022-11-13 13:24:00  ...               NaT         -106752.230292
# 51    A0033526535 2022-12-29 12:05:00  ...  45 days 22:41:00              45.945242


# --------------------------------
# Filter to just 'Outcome'rows
# --------------------------------
# Filter per_stay_df to only rows that have Outcomes
        # NOTE !!!rows with  stay length of "NaT" didn't have a matching income row 
        # becuse it was outside the widow the data was pulled for
per_stay_df = per_stay_df[per_stay_df['Outcome/Intake']=='Outcome']

per_stay_df.drop(['Outcome/Intake', 'Outcome/Intake_0_1', 'Condition', 'Size'], axis=1, inplace=True)



# --------------------------------
# Cusum of stay_time_length_days by Animal id
# --------------------------------
per_stay_df['Cusum_stay_time_length_days'] = per_stay_df\
                                        .groupby(by=['Animal ID'])['stay_time_length_days'].cumsum()


# -----------------------------------------------------------------------------
# count of Doggy Day Outs events by a unique 'Animal #' ***
# -----------------------------------------------------------------------------
# Use a groupby to get the a count of Doggy Day Outs by a unique 'Animal #'
count_of_ddo_df = ddo_df.groupby(by =['Animal #'])['Animal Species'].agg('count').reset_index()
count_of_ddo_df = count_of_ddo_df.rename(columns={"Animal #": "Animal ID", "Animal Species":"num_of_DDO"})

# print('Count of Doggy Day Outs by Unique Animal #:')
# print(count_of_ddo_df.head()) # Let's see what we made!

# Count of Doggy Day Outs by Unique Animal #:
#      Animal ID  num_of_DDO
# 0  A0024671856           3
# 1  A0028569416          13
# 2  A0030784143           3
# 3  A0031354985          12
# 4  A0038700097           4





# -----------------------------------------------------------------------------
# FILTER DATA ****
# -----------------------------------------------------------------------------

# drop dogs that we only saw in outcome and not intake file
per_stay_df = per_stay_df.dropna(subset=['stay_time_length'])


# Drop rows with any NaN values
per_stay_df = per_stay_df.dropna()


# # ------------------------------------
# # Remove puppies from the data
# # ------------------------------------
# not_a_puppy_mask = per_stay_df['Outcome Age As Month'] > 12 # puppies: under 12 months of age
# per_stay_df = per_stay_df[not_a_puppy_mask]


# # ------------------------------------
# # To keep ONLY the most recent animel stay line ***
# # ------------------------------------
# # keep only the most recent record for each Animal id
# per_stay_df = per_stay_df.sort_values('Date/Time').groupby('Animal ID').tail(1).reset_index(drop=True)

# per_stay_df.drop(['stay_time_length', 'stay_time_length_days'], axis=1, inplace=True)


# -----------------------------------------------------------------------------
# JOIN THE DATA!!!! ***
# -----------------------------------------------------------------------------

# --------------------------------
# Join per_stay_df and the count_of_ddo_df
# --------------------------------
# Left Join per_stay_df with count_of_ddo_df
per_stay_n_ddo_cnt_df = per_stay_df.merge(count_of_ddo_df, how='left', on='Animal ID')

per_stay_n_ddo_cnt_df['num_of_DDO'] = per_stay_n_ddo_cnt_df['num_of_DDO'].fillna(0)

# print('Intake and Outcomes Concatenated Data joined with count of Doggy Day Outs events: ')
# print(per_stay_n_ddo_cnt_df.head()) # See what we got! 
# print('Note, num_of_DDO will be NaN for a dog that has NOT gone on a doggie day out')

#      Animal ID           Date/Time  ... Cusum_stay_time_length_days num_of_DDO
# 0  A0049989897 2022-04-18 12:32:00  ...                    0.129861        0.0
# 1  A0049998281 2022-04-19 13:03:00  ...                    0.090278        0.0
# 2  A0049997773 2022-04-19 17:49:00  ...                    0.313890        0.0
# 3  A0050001726 2022-04-20 13:12:00  ...                    0.949308        0.0
# 4  A0040885657 2022-04-20 14:53:00  ...                    0.101389        0.0


# get ride of spaces in col names
per_stay_n_ddo_cnt_df = per_stay_n_ddo_cnt_df.rename(columns={'Animal ID': 'Animal_ID', 
                                          'Primary Breed': 'Primary_Breed', 
                                          'Primary Colour': 'Primary_Colour', 
                                          'Operation Type': 'Operation_Type',
                                          'Outcome Age As Month':'Outcome_Age_As_Month',
                                          'Age Group': 'Age_Group'})

# print('\n\n','Final df  per_stay_n_ddo_cnt_df:')
# print('----------------------------')
# print(per_stay_n_ddo_cnt_df.info())

# # Check that 'Animal ID' is_unique 
# print('\n\n')
# print("'Animal ID' is_unique? ", per_stay_n_ddo_cnt_df['Animal_ID'].is_unique)
# print("Count of unique puppy ids", len(per_stay_n_ddo_cnt_df[per_stay_n_ddo_cnt_df['Outcome_Age_As_Month'] <= 12]['Animal_ID'].unique()))
# print('Final df: per_stay_n_ddo_cnt_df.shape', per_stay_n_ddo_cnt_df.shape)



# -----------------------------------------------------------------------------
# SAVE DATA!!!! ***
# -----------------------------------------------------------------------------

per_stay_n_ddo_cnt_df.to_csv('Data/Output__Animal_ID_per_stay_n_ddo_cnt_df.csv', index=False)


print('\n\n','Final df:  one row per animal id per stay:')
print('--------------------------------------------------------')
print('Final df: per_stay_n_ddo_cnt_df.shape', per_stay_n_ddo_cnt_df.shape)


not_a_puppy_mask = per_stay_n_ddo_cnt_df['Outcome_Age_As_Month'] > 12 # puppies: under 12 months of age
per_stay_n_ddo_cnt_df[not_a_puppy_mask].to_csv('Data/Output__Animal_ID_per_stay_n_ddo_cnt_NO_PUPPIES_df.csv', index=False)


print('\n\n','Final df:  one row per animal id per stay NO PUPPIES:')
print('--------------------------------------------------------')
print('Final df: per_stay_n_ddo_cnt_df[not_a_puppy_mask].shape', per_stay_n_ddo_cnt_df[not_a_puppy_mask].shape)



# ------------------------------------
# To keep ONLY the most recent animel stay line ***
# ------------------------------------
# keep only the most recent record for each Animal id
per_stay_n_ddo_cnt_df = per_stay_n_ddo_cnt_df.sort_values('Date/Time').groupby('Animal_ID').tail(1).reset_index(drop=True)

per_stay_n_ddo_cnt_df.drop(['stay_time_length', 'stay_time_length_days'], axis=1, inplace=True)


per_stay_n_ddo_cnt_df.to_csv('Data/Output___per_Animal_ID_n_ddo_cnt_df.csv', index=False)


print('\n\n','Final df:  one row per animal id:')
print('--------------------------------------------------------')
print('Final df: per_stay_n_ddo_cnt_df.shape', per_stay_n_ddo_cnt_df.shape)


not_a_puppy_mask = per_stay_n_ddo_cnt_df['Outcome_Age_As_Month'] > 12 # puppies: under 12 months of age
per_stay_n_ddo_cnt_df[not_a_puppy_mask].to_csv('Data/Output___per_Animal_ID_n_ddo_cnt_NO_PUPPIES_df.csv', index=False)

print('\n\n','Final df:  one row per animal id NO PUPPIES:')
print('--------------------------------------------------------')
print('Final df: per_stay_n_ddo_cnt_df[not_a_puppy_mask].shape', per_stay_n_ddo_cnt_df[not_a_puppy_mask].shape)