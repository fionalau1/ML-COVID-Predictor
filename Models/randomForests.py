import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

#test nestSize, maxFeat, maxDepth, minLeaf parameters
def param_tuning(xTrain, yTrain):
    rfr = RandomForestRegressor()
    param_grid = {
        "n_estimators"      : [50,100,150],
        "max_features"      : ["sqrt", "log2"], #take out auto bc thats bagging, not a rf anymore
        "min_samples_leaf" : [3, 15, 22, 30],
        "max_depth": [5, 7, 10, 12]
    }

    CV_rfr = GridSearchCV(estimator=rfr, param_grid=param_grid, cv=5, scoring='r2')
    CV_rfr.fit(xTrain, yTrain)

    best_params = CV_rfr.best_params_ #get dict of best parameter values
    
    nestSize = best_params['n_estimators'] #store best nestSize
    maxFeat = best_params['max_features'] #store best maxFeat
    maxDepth = best_params['max_depth'] #store best maxDepth
    minLeaf = best_params['min_samples_leaf'] #store best minLeaf

    return nestSize, maxFeat, maxDepth, minLeaf


# Based on the ensemble method using linear regression results, (see report for process) drop the features found to be insignificant
# in addition, split data into test/train and scale to 0 mean 1 variance
def ensemble_preprocess(completeData):
    #features determined to be significant from linear regression hypothesis testing
    ensemble = ['population', 'density', 'num_households', 'poverty', 'asian', 'hispanic', 'no_college', 'emergency',
                'election_post', 'school_closure', 'individual_mask', 'public_mask', 'social_distancing',
                'large_gathering_ban', 'gathering_ban', 'nonessential_bus_closure', 'nonessential_bus_lift',
                'stay_home', 'stay_home_lifted', 'travel_quarantine', 'prevCases']

    ensemble_data = completeData[ensemble] #select based on these features for new ensemble dataset
    ensemble_data = ensemble_data.to_numpy() # convert dataframe into numpy to make splitting into xData and yData easier

    xData = ensemble_data[:, :-1]  # keep all rows, except the last column
    yData = ensemble_data[:, -1]  # keep all rows, only last col

    # split x data and y data using 30-70 test/train ratio (holdout method)
    xTrain_ensemble, xTest_ensemble, yTrain, yTest = train_test_split(xData, yData, test_size=0.3, random_state=42)

    # transform to 0 mean 1 variance
    scaler = StandardScaler()
    scaler.fit(xTrain_ensemble)
    xTrain_ensemble = scaler.transform(xTrain_ensemble)  # transform xTrain into xTrain_ensemble
    xTest_ensemble = scaler.transform(xTest_ensemble)  # transform xTest into xTest_ensemble
    return xTrain_ensemble, yTrain, xTest_ensemble, yTest

#Using the ensemble method significant features, train random forest
def ensemble_random_forest(xTrain_ensemble, yTrain, xTest_ensemble, yTest):
    print("Ensemble random forest")
    randomforest = RandomForestRegressor(n_estimators=50, max_features='sqrt', criterion='mse', max_depth=12,
                                         min_samples_leaf=3)
    # train random forest based on xTrain_ensemble
    trainedRF = randomforest.fit(xTrain_ensemble, yTrain)

    # for xTrain_ensemble data...
    # get R2 and root mean squared error as prediction accuracy metrics
    print("xTrain R2 and MSE:")
    yHatTrain = trainedRF.predict(xTrain_ensemble)
    print(r2_score(yHatTrain, yTrain))
    print(np.sqrt(mean_squared_error(yHatTrain, yTrain)))

    # for xTest_ensemble data...
    # get R2 and root mean squared error as prediction accuracy metrics
    yHat = trainedRF.predict(xTest_ensemble)
    print("xTest R2 and MSE:")
    print(r2_score(yTest, yHat))
    print(np.sqrt(mean_squared_error(yHat, yTest)))

#Based on Pearson correlation matrix, preprocess data features (drop features, split into 30-70 ratio, transform to 0 mean and 1 variance)
def pearson_preprocess(completeData):
    # drop values determined from Pearson Correlation matrix to be redundant (see Pearson section of report)
    pearson_data = completeData.drop(["population", "no_college", "emergency", "individual_mask", "stay_home_lifted"], axis=1)
    pearson_data = pearson_data.to_numpy()

    xVals = pearson_data[:, :-1]
    yVals = pearson_data[:, -1]

    # Split into 70/30 train-test ratio
    xTrain_pearson, xTest_pearson, yTrain, yTest = train_test_split(xVals, yVals, test_size=0.3, random_state=42)

    # transform dataset to 0 mean and unit variance
    scaler = StandardScaler()
    scaler.fit(xTrain_pearson)
    xTrain_pearson = scaler.transform(xTrain_pearson)  # apply to xTrain_pearson
    xTest_pearson = scaler.transform(xTest_pearson)  # apply to xTest_pearson
    return xTrain_pearson, yTrain, xTest_pearson, yTest

#Create random forest based on subset of features dropped based on Pearson Correlation matrix findings
def pearson_random_forest(xTrain_pearson, yTrain, xTest_pearson, yTest):
    print("Using dropped features from Pearson correlation matrix")
    randomforest = RandomForestRegressor(n_estimators=50, max_features='sqrt', criterion='mse', max_depth=12, min_samples_leaf=3)
    # train random forest on xTrain_pearson features
    trainedRF = randomforest.fit(xTrain_pearson, yTrain)

    # for xTrain_pearson data...
    # get R2 and root mean squared error as prediction accuracy metrics
    print("Pearson xTrain R2 and MSE:")
    yHatTrain = trainedRF.predict(xTrain_pearson)
    print(r2_score(yHatTrain, yTrain))
    print(np.sqrt(mean_squared_error(yHatTrain, yTrain)))

    # for xTest_pearson data...
    # get R2 and root mean squared error as prediction accuracy metrics
    yHat = trainedRF.predict(xTest_pearson)
    print("Pearson xTest R2 and MSE:")
    print(r2_score(yTest, yHat))
    print(np.sqrt(mean_squared_error(yHat, yTest)))

#general method for any random forest that uses PCA for reducing number of features
def PCA_random_forest(xTrain, yTrain, xTest, yTest):
    print("Random Forest trained on PCA features")
    randomforest = RandomForestRegressor(n_estimators=50, max_features='sqrt', criterion='mse', max_depth=10,
                                         min_samples_leaf=3)

    # Using best preprocessing results (PCA preprocessing section of report) from reducing 32 to 24 principal components, transform data using PCA
    pca = PCA(n_components=24)
    xTrain_PCA = pca.fit_transform(xTrain) # fit AND transform xTrain to xTrain_PCA in one step

    #fit random forest
    trainedRF = randomforest.fit(xTrain_PCA, yTrain)

    # for xTrain_PCA data...
    # get R2 and root mean squared error as prediction accuracy metrics
    print("PCA for xTrain")
    PCA_yhat_train = trainedRF.predict(xTrain_PCA)
    print("R2 and MSE:")
    print(r2_score(PCA_yhat_train, yTrain))
    print(np.sqrt(mean_squared_error(PCA_yhat_train, yTrain)))
    varianceRatios = pca.explained_variance_ratio_

    # for xTest data, transform and predict
    # get R2 and root mean squared error as prediction accuracy metrics
    print("PCA for xTest")
    xTest_PCA = pca.transform(xTest)
    PCA_yhat = trainedRF.predict(xTest_PCA)
    print("R2 and MSE:")
    print(r2_score(yTest, PCA_yhat))
    print(np.sqrt(mean_squared_error(PCA_yhat, yTest)))

    varianceRatios = pca.explained_variance_ratio_

#Preprocessing data (just train-test split, transform data to 0 mean and 1 variance) without reducing the number of features used for training a random forest
def regular_preprocess(completeData):
    completeData = completeData.to_numpy()
    xData = completeData[:, :-1]  # keep all but the last column
    yData = completeData[:, -1]  # keep only the last column of labels

    # HOLDOUT: split into 70:30 train:test
    xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    scaler.fit(xTrain)  # scale to 0 mean and unit variance
    xTrain = scaler.transform(xTrain)  # transform xTrain
    xTest = scaler.transform(xTest)  # transform xTest
    return xTrain, yTrain, xTest, yTest

#Create "normal" random forest without any feature reduction
def random_forest(xTrain, yTrain, xTest, yTest):
    print("Regular Random Forest")
    #Without PCA
    randomforest = RandomForestRegressor(n_estimators=50, max_features='sqrt', criterion='mse', max_depth=10, min_samples_leaf=3)
    trainedRF = randomforest.fit(xTrain, yTrain) # train model on xTrain

    print("xTrain:")
    print("R2 and RMSE:")
    # for xTrain data...
    # get R2 and root mean squared error as prediction accuracy metrics
    yHatTrain = trainedRF.predict(xTrain)
    print(r2_score(yHatTrain, yTrain))
    print(np.sqrt(mean_squared_error(yHatTrain, yTrain)))

    # for xTest data...
    # get R2 and root mean squared error as prediction accuracy metrics
    print("xTest:")
    yHat = trainedRF.predict(xTest)
    print("R2 and MSE:")
    print(r2_score(yTest, yHat))
    print(np.sqrt(mean_squared_error(yHat, yTest)))

def main():
    # Split data into features and labels
    completeData = pd.read_csv("labeledData.csv")
    completeData = completeData.drop(columns=['state'], axis=1)

    xTrain, yTrain, xTest, yTest = regular_preprocess(completeData)
    #nestSize, maxFeat, maxDepth, minLeaf = param_tuning(xTrain, yTrain)
    # print(nestSize, maxFeat, maxDepth, minLeaf)

    random_forest(xTrain, yTrain, xTest, yTest)
    PCA_random_forest(xTrain, yTrain, xTest, yTest)

    xTrain_pearson, yTrain, xTest_pearson, yTest = pearson_preprocess(completeData)
    # nestSize, maxFeat, maxDepth, minLeaf = param_tuning(xTrain_pearson, yTrain)
    # print(nestSize, maxFeat, maxDepth, minLeaf)
    pearson_random_forest(xTrain_pearson, yTrain, xTest_pearson, yTest)
    PCA_random_forest(xTrain_pearson, yTrain, xTest_pearson, yTest)

    xTrain_ensemble, yTrain, xTest_ensemble, yTest = ensemble_preprocess(completeData)
    ensemble_random_forest(xTrain_ensemble, yTrain, xTest_ensemble, yTest)
    # nestSize, maxFeat, maxDepth, minLeaf = param_tuning(xTrain_ensemble, yTrain)
    # print(nestSize, maxFeat, maxDepth, minLeaf)

if __name__ == "__main__":
    main()
