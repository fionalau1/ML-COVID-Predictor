import pandas as pd
from sklearn.impute import KNNImputer

demographics = pd.read_csv("demographicOld.csv")
print(demographics.head())
cols = demographics.columns.to_list()
print(cols)

#Drop string columns, keep floats for the KNNImputer
float_demo = demographics.drop(['County Name', 'State'], 1)

#Use KNNImputer to fill in missing values w/13 neighbors and euclidean distance
imputer = KNNImputer(n_neighbors=13, weights='uniform', metric='nan_euclidean')
imputer.fit(float_demo)
policies_filled = imputer.transform(float_demo)
#print(policies_filled)

#add columns back on except State and County
col_list=['FIPS', 'stateFIPS', 'population', 'population_density', 'no_households', 'household_income', 'p_poverty', 'p_white', 'p_black', 'p_native', 'p_asian', 'p_hispanic', 'p_belowcollege', 'blue']
policies_filled=pd.DataFrame(policies_filled, columns=col_list)
print(policies_filled.head())
#print(policies_filled.iloc[69:99, 13])

#add State and County cols back
policies_filled.insert(loc=1, column='County Name', value=demographics['County Name'])
policies_filled.insert(loc=2, column='State', value=demographics['State'])
print(policies_filled.head())
policies_filled.to_csv("demographic.csv", index=False) #export file to new csv






