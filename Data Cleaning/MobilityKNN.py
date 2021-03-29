import pandas as pd
from sklearn.impute import KNNImputer

mobility = pd.read_csv("mobilityOld.csv")
cols = mobility.columns.to_list()

#Store State col (string values do not work for this KNN)
mobility_state = mobility['State']
print(mobility_state)
#Store County col
mobility_county = mobility['County']
print(mobility_county)
float_mob = mobility.drop(['State', 'County'], 1)
print(float_mob)

#Use KNNImputer to fill in missing values w/13 neighbors and euclidean distance
imputer = KNNImputer(n_neighbors=13, weights='uniform', metric='nan_euclidean')
imputer.fit(float_mob)
mobility_filled = imputer.transform(float_mob)
#print(policies_filled)

#add columns back on except State and County
col_list=['FIPS', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168', '169', '170', '171']
mobility_filled=pd.DataFrame(mobility_filled, columns=col_list)
print(mobility_filled.head())

#add State and County cols back
mobility_filled.insert(loc=1, column='State', value=mobility_state)
mobility_filled.insert(loc=2, column='County', value=mobility_county)
mobility_filled = mobility_filled.iloc[1:] #Drop first row, corresponds to a missing county
mobility_filled.to_csv("mobility.csv", index=False)
