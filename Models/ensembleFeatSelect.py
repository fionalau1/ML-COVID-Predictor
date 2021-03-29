import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from operator import itemgetter

# 1. Run Linear Regression, get p-values
# Drop state, county, and day: these features were not used when running a linear regression
completeData = pd.read_csv("labeledData1.csv")
completeData = completeData.drop(columns = ['state', 'county', 'day'])
completeData=completeData.to_numpy()

xData = completeData[:, :-1]  # keep all but the last column
yData = completeData[:, -1]  # keep only the last column of labels

# Use the holdout method: split data into train/test 70/30 ratio
xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.3, random_state=42)

scaler = StandardScaler()
scaler.fit(xTrain)  # scale to 0 mean and unit variance
xTrain = scaler.transform(xTrain)  # transform xTrain
xTest = scaler.transform(xTest)  # transform xTest

# Use statsmodels package to get p-values corresponding to the beta coeffcients, for hypothesis testing (see report)
# Use p-values under P>|t| for two-tailed hypothesis testing
xTrain2 = sm.add_constant(xTrain) # add a constant in front of xTrain to get the intercept
linear_model = sm.OLS(yTrain, xTrain2)
results = linear_model.fit()

print(results.summary())

'''# Double check using sklearn's Linear Regression model
reg = LinearRegression().fit(xTrain, yTrain)
yHat = reg.predict(xTest)

print(r2_score(yHat, yTest))
print(np.sqrt(mean_squared_error(yHat, yTest))) '''

# 2. Get the significant corresponding significant coefs from the original list of features
# original list of feature names, without state, county and day to match the linear regressed features
featNames = ['mobility', 'population', 'density',	'num_households', 'income',	'poverty', 'white', 'black', 'native', 'asian', 'hispanic',	'no_college', 'blue',	'emergency', 'election_post', 'school_closure',	'individual_mask',	'public_mask', 'social_distancing', 'large_gathering_ban', 'gathering_ban', 'gathering_lifted', 'nonessential_bus_closure', 'nonessential_bus_lift', 'stay_home', 'stay_home_lifted','travel_quarantine', 'restaurant_closure','limited_gathering', 'prevCases']

# (not including intercept, beta 0)
# the ordinal positions of betas with alphas less than 0.05, determined to be significant
slices=[2,3,4,6,10,11,12,14,15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 30]

# scale back by one b/c list starts index count from 0, and statsmodel package uses ordinal numbers
new_slices = [x-1 for x in slices]

# use itemgetter to iterate through the new list and get the corresponding name to the index
significant_features = itemgetter(*new_slices)(featNames)
significant_features = list(significant_features)
print(significant_features)

# betas with alpha values larger than 0.05, deemed nonsignficant
non_sig = [1,5,7,8,9, 13, 22, 28, 29]
new_non_sig = [x-1 for x in non_sig]
non_sig_feat = itemgetter(*new_non_sig)(featNames)
non_sig_feat = list(non_sig_feat)
print(non_sig_feat)