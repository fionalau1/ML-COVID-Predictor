import pandas as pd
import numpy as np
from sklearn import model_selection
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from scipy.ndimage.filters import uniform_filter1d
from sklearn import tree

# normalize xTrain and xTest using Standard Scaler
def normalize(xTrain, xTest):
     # NORMALIZE DATA 
    scaler = StandardScaler()
    scaler.fit(xTrain) # scale to 0 mean and unit variance
    xTrainNorm = scaler.transform(xTrain) # transform xTrain
    xTestNorm = scaler.transform(xTest) # transform xTest
    return xTrainNorm, xTestNorm

# create pearson correlation matrix, plot heatmap, print highly correlated features
# drop the features that are highly correlated according to the correlation matrix
def pearsonFeat(xTrain, xTest, yTrain, yTest):
    # # CORRELATION MATRIX
    featNames = ['county', 'day', 'mobility', 'population', 'density', 'num_households', 'income',	'poverty', 'white', 'black', 'native', 'asian', 'hispanic',	'no_college', 'blue',	'emergency', 'election_post', 'school_closure',	'individual_mask',	'public_mask', 'social_distancing', 'large_gathering_ban', 'gathering_ban', 'gathering_lifted', 'nonessential_bus_closure', 'nonessential_bus_lift', 'stay_home', 'stay_home_lifted','travel_quarantine', 'restaurant_closure','limited_gathering', 'prevCases']
    xTrainLabeled = pd.DataFrame(xTrain, columns = featNames) 
    xTrainLabeled["labels"] = yTrain # add labels to xTrain copy for correlation matrix
    correlations = xTrainLabeled.corr(method ='pearson') # make the pearson correlation matrix
    # print(correlations)

    # # HEATMAP
    heat_map = sb.heatmap(correlations, xticklabels=True, yticklabels=True) # plot correlations using a heatmap
    # plt.show()

    # # SELECT FEATS TO DROP
    # # Identify highly correlated features
    # # Print out the row/column feature pair that have high correlations 
    # print("Feature pairs that have an absolute correlation of at least .6")
    # for column in correlations.columns:
    #     for row in correlations.index:
    #         if abs(correlations.loc[row][column]) > .6  and row!=column: #dont print if row and column are the same value
    #             print("Feat1", column, "Feat2", row, abs(correlations.loc[row][column]), "Feat1's correlation with target", correlations.loc[column]['labels'])
    
    # # DROP CORRELATED FEATURES
    # # drop population (3), no_college(13), emergency(15), individual_mask(18), stay_home_lifted (27)
    xTrainPearson = np.delete(xTrain, [3, 13, 15, 18, 27], axis=1) # drop from xTrain
    xTestPearson = np.delete(xTest, [3, 13, 15, 18, 27], axis=1) # drop from xTest
    # return xTrainPearson and xTestPearson with only the uncorrelated features
    return xTrainPearson, xTestPearson

# Perform PCA dimension reduction on xTrain and xTest
def pcaFeat(xTrain, xTest):
    # # PCA FEATURE REDUCTION
    sklearn_pca = sklearnPCA(.95)
    # # fit pca on the training set, keeping 95% variance
    x_pca = sklearn_pca.fit(xTrain)
    # varianceRatios = sklearn_pca.explained_variance_ratio_
    # print("Variance ratios for 95 percent explained variance:")
    # print(varianceRatios)
    # print("First 3 principal components:")
    # print(sklearn_pca.components_[:3])
    # sortedIndices = np.ndarray.argsort(np.absolute(sklearn_pca.components_[:3])) # sort the pca component coefficients for the first 3 components
    # print(sortedIndices)
    # # transform xTrain and xTest using principal components
    xTrainPCA = sklearn_pca.transform(xTrain) 
    xTestPCA = sklearn_pca.transform(xTest)
    # # returned the PCA transformed xTrain and xTest
    return xTrainPCA, xTestPCA

# Use ensemble method to drop features based on coefficients of linear regression model
def ensembleFeat(xTrain, xTest):
    # Drop non-significant features according to Linear Regression model coefficients
    # county(0), day(1) mobility (2), income (6), white (8), black (9), native (10), blue(14), gathering_lifted(23), restaurant_closure (29), limited_gathering(30)
    xTrainEnsemble = np.delete(xTrain, [0, 1, 2, 6, 8, 9, 10, 14, 23, 29, 30], axis=1) # drop from xTrain
    xTestEnsemble = np.delete(xTest, [0, 1, 2, 6, 8, 9, 10, 14, 23, 29, 30], axis=1) # drop from xTest
    # return xTrainEnsemble and xTestEnsemble with only the significant features
    return xTrainEnsemble, xTestEnsemble

# return the best parameters for the regressor, using cross validation
def bestParams(xTrain, yTrain):
    # perform hyperperameter tuning on max depth and min samples leaf for decision tree
    dtClf = model_selection.GridSearchCV(DecisionTreeRegressor(), [{'max_depth': range(5,25,3), 'min_samples_leaf': range(1, 10, 2)}], cv=5, scoring='r2')
    dtClf.fit(xTrain, yTrain)
    print(dtClf.best_params_)
    # return the optimal hyperparameters
    return dtClf.best_params_

# train regressor on xTrain and yTrain, predict train and test sets, return the evaluation metrics
def dtPredict(regressor, xTrain, xTest, yTrain, yTest):
    # fit model
    regressor.fit(xTrain, yTrain)
    # evaluate train
    yHatTrain = regressor.predict(xTrain)
    r2_Train = metrics.r2_score(yTrain, yHatTrain)
    rmse_Train = metrics.mean_squared_error(yTrain, yHatTrain,  squared=False)
    # evaluate test
    yHatTest = regressor.predict(xTest)
    r2_Test = metrics.r2_score(yTest, yHatTest)
    rmse_Test = metrics.mean_squared_error(yTest, yHatTest, squared=False)
    # return evaluation metrics
    return r2_Train, rmse_Train, r2_Test, rmse_Test

# visualize the decision tree, save image as decision_tree.png
def visualizeTree(regressor, xTrain, yTrain, featNames):
    regressor.fit(xTrain, yTrain)
    # tree plot of decision tree, save as decision_tree.png
    fig, ax = plt.subplots(figsize=(50, 30))
    # show 3 layers of tree
    tree.plot_tree(regressor, filled = True, feature_names = featNames, max_depth= 3, fontsize = 30, rounded = True, precision = 1)
    fig.savefig("decision_tree.png")
    # text representation of the tree
    text_representation = tree.export_text(regressor, feature_names= featNames)
    print(text_representation)
   
def main():        
    # load data file using number of cases as label
    completeData = pd.read_csv("labeledData.csv") 
    # drop state column because this is not a feature (column only used for debugging purposes)
    completeData = completeData.drop(columns = ['state']).to_numpy() 
    # features
    xData = completeData[:,:-1]                         
    # labels            
    yData = completeData[:,-1]

    # HOLDOUT: split into 70:30 train:test
    xTrain, xTest, yTrain, yTest = model_selection.train_test_split(xData, yData, test_size=0.3, random_state=42)   

    # PEARSON CORRELATION FEATURE SELECTION
    xTrainPearson, xTestPearson = pearsonFeat(xTrain, xTest, yTrain, yTest) # remove highly correlated features

    # ENSEMBLE FEATURE SELECTION
    xTrainEnsemble, xTestEnsemble = ensembleFeat(xTrain, xTest) # remove features from linear regression significance teset
    
    # NORMALIZE ALL DATA 
    xTrain, xTest = normalize(xTrain, xTest) # normalize original set
    xTrainPearson, xTestPearson = normalize(xTrainPearson, xTestPearson) # normalize Pearson set
    xTrainEnsemble, xTestEnsemble = normalize(xTrainEnsemble, xTestEnsemble) # normalize ensemble set 

    # PCA FEATURE REDUCTION
    xTrainPCA, xTestPCA = pcaFeat(xTrain, xTest) # perform PCA on the normalized dataset
    xTrainPearsonPCA, xTestPearsonPCA = pcaFeat(xTrainPearson, xTestPearson) # perform PCA on the normalized and Pearson dropped dataset
    
    # # DECISION TREE MODELS
    # Decision Tree Regressor normalized dataset with all features
    # optimalParameters = bestParams(xTrain, yTrain)
    # regressor = DecisionTreeRegressor(max_depth=optimalParameters['max_depth'], min_samples_leaf= optimalParameters['min_samples_leaf'])
    regressor = DecisionTreeRegressor(max_depth = 11, min_samples_leaf = 1)
    r2_Train, rmse_Train, r2_Test, rmse_Test = dtPredict(regressor, xTrain, xTest, yTrain, yTest)
    print("DT train r2:", r2_Train)
    print("DT train rmse:", rmse_Train)
    print("DT test r2:", r2_Test)
    print("DT test rmse:", rmse_Test)

    # # Decision Tree Regressor normalized dataset with PCA
    # optimalParameters = bestParams(xTrainPCA, yTrain)
    # regressor = DecisionTreeRegressor(max_depth=optimalParameters['max_depth'], min_samples_leaf= optimalParameters['min_samples_leaf'])
    regressor = DecisionTreeRegressor(max_depth = 23, min_samples_leaf= 1)
    r2_Train, rmse_Train, r2_Test, rmse_Test = dtPredict(regressor, xTrainPCA, xTestPCA, yTrain, yTest)
    print("DT PCA train r2:", r2_Train)
    print("DT PCA train rmse:", rmse_Train)
    print("DT PCA test r2:", r2_Test)
    print("DT PCA test rmse:", rmse_Test)

    # # Decision Tree Regressor normalized dataset with Pearson Dropped Features
    # optimalParameters = bestParams(xTrainPearson, yTrain)
    # regressor = DecisionTreeRegressor(max_depth=optimalParameters['max_depth'], min_samples_leaf= optimalParameters['min_samples_leaf'])
    regressor = DecisionTreeRegressor(max_depth = 20, min_samples_leaf = 1)
    r2_Train, rmse_Train, r2_Test, rmse_Test = dtPredict(regressor, xTrainPearson, xTestPearson, yTrain, yTest)
    print("DT Pearson train r2:", r2_Train)
    print("DT Pearson train rmse:", rmse_Train)
    print("DT Pearson test r2:", r2_Test)
    print("DT Pearson test rmse:", rmse_Test)

    # # visualize this decision tree wiht pearson dropped features, save as decision_tree.png
    # featNames = ['county', 'day', 'mobility', 'density', 'num_households', 'income', 'poverty', 'white', 'black', 'native', 'asian', 'hispanic', 'blue', 'election_post', 'school_closure',	'public_mask', 'social_distancing', 'large_gathering_ban', 'gathering_ban', 'gathering_lifted', 'nonessential_bus_closure', 'nonessential_bus_lift', 'stay_home', 'travel_quarantine', 'restaurant_closure','limited_gathering', 'prevCases']
    # visualizeTree(regressor, xTrainPearson, yTrain, featNames)

    # # Decision Tree Regressor normalized datset with Pearson Dropped Features and PCA
    # optimalParameters = bestParams(xTrainPearsonPCA, yTrain)
    # regressor = DecisionTreeRegressor(max_depth=optimalParameters['max_depth'], min_samples_leaf= optimalParameters['min_samples_leaf'])
    regressor = DecisionTreeRegressor(max_depth = 20, min_samples_leaf = 1)
    r2_Train, rmse_Train, r2_Test, rmse_Test = dtPredict(regressor, xTrainPearsonPCA, xTestPearsonPCA, yTrain, yTest)
    print("DT Pearson PCA train r2:", r2_Train)
    print("DT Pearson PCA train rmse:", rmse_Train)
    print("DT Pearson PCA test r2:", r2_Test)
    print("DT Pearson PCA test rmse:", rmse_Test)

    # # Decision Tree Regressor normalized dataset with Ensemble Dropped Features
    # optimalParameters = bestParams(xTrainEnsemble, yTrain)
    # print(optimalParameters)
    # regressor = DecisionTreeRegressor(max_depth=optimalParameters['max_depth'], min_samples_leaf= optimalParameters['min_samples_leaf'])
    regressor = DecisionTreeRegressor(max_depth = 20, min_samples_leaf = 1)
    r2_Train, rmse_Train, r2_Test, rmse_Test = dtPredict(regressor, xTrainEnsemble, xTestEnsemble, yTrain, yTest)
    print("DT Ensemble train r2:", r2_Train)
    print("DT Ensemble train rmse:", rmse_Train)
    print("DT Ensemble test r2:", r2_Test)
    print("DT Ensemble test rmse:", rmse_Test)


if __name__ == "__main__":
    main()
    