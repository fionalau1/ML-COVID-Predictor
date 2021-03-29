import pandas as pd
import numpy as np
import numpy as np
from scipy.ndimage.filters import uniform_filter1d

# DATA CLEANING: compile all datasets into one, save as labeledData.csv
# this file take several minutes to run

# load labels as df
cases = pd.read_csv("cases.csv") 
# load features as df
demographic = pd.read_csv("demographic.csv") 
mobility = pd.read_csv("mobility.csv")
policies = pd.read_csv("policies.csv")

# Delete the counties that are in cases but not mobility
# Also delete the rows that describe states instead of counties (where FIPS number is 1-56)
for index, row in cases.iterrows():
    if row['FIPS'] not in mobility['FIPS'].to_numpy() or int(row['FIPS']<=60):
        cases.drop(index, inplace = True) 
        
# Delete the counties that are mobility but not cases
for index, row in mobility.iterrows():
    if row['FIPS'] not in cases['FIPS'].to_numpy() or int(row['FIPS']<=60):
        mobility.drop(index, inplace = True)

# Delete the counties that are demographic but not mobility/cases
for index, row in demographic.iterrows():
    if row['FIPS'] not in cases['FIPS'].to_numpy():
        demographic.drop(index, inplace = True)

# Arrays to store feature and labels
# Each value in the following arrays correspond to info for a single record (= unique county/day pair)
countyCodes, prevCases, mobil, labels, dates, states = [], [], [], [], [], []
population, density, num_houses, income, poverty, white, black, native, asian, hispanic, no_college, blue = [], [], [], [], [], [], [], [], [], [], [], []
policyDeclaration = [[] for p in range(16)]

# Create dataframe to combine all the labeled data; we want to match the correct features (mobility, demographic, policy etc.) to the correct county and date
# Iterate over every county
for index, row in cases.iterrows(): 
    print(row)
    FIPS = row['FIPS'] # county FIPS
    state = row['State'] # state abbreviation
    dailyCases=row['39':'171'] # county cases for each day between day 39-171
    mobilityRates= mobility.loc[mobility['FIPS'] == FIPS, '39':'171'].to_numpy().flatten() # mobility rate for each day between day 39-171
    demographicFeats = demographic.loc[demographic['FIPS'] == FIPS, 'population':].to_numpy().flatten() # county demographic data

    # compute running average of previous cases as a new feature
    avgCases = list(uniform_filter1d(dailyCases.to_list(), size=7, origin = 3)) # average over 7 days inclusive of the current day
    avgCases.insert(0,0) # add dummy 0 to the beginning of the list and shift the running average left by 1 day 
    avgCases = avgCases[:-1] # avgCases becomes the average over past 7 days not including the current day
    
    # iterate over every day
    for day in range(len(dailyCases)):
        # append labels
        labels.append(dailyCases[day]) # daily labels

        # append constant features that only depend on county
        countyCodes.append(FIPS) # county code
        states.append(state) # state 
        population.append(demographicFeats[0]) # all demographic data
        density.append(demographicFeats[1])
        num_houses.append(demographicFeats[2])
        income.append(demographicFeats[3])
        poverty.append(demographicFeats[4])
        white.append(demographicFeats[5])
        black.append(demographicFeats[6])
        native.append(demographicFeats[7])
        asian.append(demographicFeats[8])
        hispanic.append(demographicFeats[9])
        no_college.append(demographicFeats[10])
        blue.append(demographicFeats[11])
        # append varying features that depend on the day and county
        prevCases.append(avgCases[day]) # add average previous cases
        mobil.append(mobilityRates[day]) #add daily mobility
        dates.append(day + 39) #add day number

        # apend policy information from policy.csv
        policyDates = (policies.loc[policies['abbr'] == state,'Emergency Declaration':]).to_numpy().flatten() 
        # large gathering column and large gathering number may contain multiple values
        gatheringDate_Index = 0 # var to keep track of which large gathering date is being used because there may be multiple entries
        
        i = 0
        banDate = 0
        # for each of the 16 different policies 
        for polDate in range(len(policyDates)):
            # split the policy date value by ';' if such a split exists
            x = str(policyDates[polDate]).split(';')
            # if the column is Large Gathering Ban, and it has more than one value indicating the different days that the policy was re-implemented
            if polDate == 6: 
                d = float(x[0]) # get the first date in large gathering ban
                if len(x)>1: # if large gathering ban has more than one value, we want to use the most recent past date
                    banDate = d
                    # while the policy implementation date is before the currrent day, update d
                    while i<len(x) and float(x[i]) <= (day + 39):
                        d = float(x[i]) # set d to be this large gathering ban date
                        i = i+1 # i will keep track of which large gathering date we are using
                banDate = d # keep track of this date to use the right value for the large gathering ban number column
            # if the column is Large Gathering Ban Number,
            if polDate == 15:
                d = banDate # set the date to be the banDate from the Large Gathering Ban column
            # otherwise, the date is just the only date listed for this policy and county
            elif polDate != 6:
                d = float(x[0])
            # if the policy date is before the current day, add the date value
            if d <= (day + 39) and d!=0:
                #if the column is Large Gathering Number, then use the correct number that corresponds to the i-1 date from Large Gathering Ban column
                if polDate == 15:
                    policyDeclaration[polDate].append(float(x[i-1]))
                #otherwise, just use the only date listed
                else:
                    policyDeclaration[polDate].append(d)
            # else add 0 as the policy has not yet been implemented
            else: 
                policyDeclaration[polDate].append(0)
 
# create dataframe of the complete cleaned data
#each row is a record and each column is a feature/label
            # features: (state), county, day, mobility, and demographics
allData = {"state": states, "county": countyCodes, "day": dates, "mobility": mobil, "population": population, "density": density, 
            "num_households": num_houses, "income": income, "poverty": poverty, "white": white, "black": black, "native": native, "asian": asian, 
            "hispanic": hispanic, "no_college": no_college, "blue": blue, 
            # features: policy info
            "emergency": policyDeclaration[0], "election_post": policyDeclaration[1], "school_closure": policyDeclaration[2], 
            "individual_mask": policyDeclaration[3], "public_mask": policyDeclaration[4], "social_distancing": policyDeclaration[5], 
            "large_gathering_ban": policyDeclaration[6], "gathering_ban": policyDeclaration[7], "gathering_lifted": policyDeclaration[8], 
            "nonessential_bus_closure": policyDeclaration[9], "nonessential_bus_lift": policyDeclaration[10], "stay_home": policyDeclaration[11], 
            "stay_home_lifted": policyDeclaration[12], "travel_quarantine": policyDeclaration[13], "restaurant_closure": policyDeclaration[14], 
            "limited_gathering": policyDeclaration[15], 
            # feature: running average of previous number of cases
            "prevCases": prevCases,
            # label
            "label": labels}
labeledData = pd.DataFrame(allData)
# export the dataframe to csv for easy access to the complete dataset
labeledData.to_csv("labeledData2.csv", index=False)