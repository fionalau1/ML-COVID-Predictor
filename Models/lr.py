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
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression, Ridge , Lasso
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV

#labeleddata1 consists of dropped features from pearson correlation matrix
df = pd.read_csv("labeledData1.csv")

growthData = pd.read_csv("GRlabeledData.csv")

growthData = growthData.drop(columns = ['state']).to_numpy() 



"""xDataG = growthData[:,:-3] # Features                                
yDataG = growthData[:,-3:] # y data will store the number of cases, average 3 day growth rate, average 5 day growth rate

xTrainG, xTestG, yTrainG, yTestG = model_selection.train_test_split(xDataG, yDataG, test_size=0.3, random_state=42) """



"""regressor = linear_model.Lasso(alpha = 0.1)

regressor.fit(xTrainG, yTrainG[:,1])


yHatTrain = regressor.predict(xTrainG)
    # multiply previous number of cases by growth rate to get predicted number of cases
yHatCaseTrain = np.multiply(yHatTrain, xTrainG[:,-1]) 
    # evaluate by comparing true previous number of cases and predicted previous number of cases
r2_Train = metrics.r2_score(yTrainG[:,0], yHatCaseTrain)
rmse_Train = metrics.mean_squared_error(yTrainG[:,0], yHatCaseTrain,  squared=False)

    # evaluate test
yHatTest = regressor.predict(xTestG)
    # multiply previous number of cases by growth rate to get predicted number of cases
yHatCaseTest = np.multiply(yHatTest, xTestG[:,-1]) 
    # evaluate by comparing true previous number of cases and predicted previous number of cases
r2_Test = metrics.r2_score(yTestG[:,0], yHatCaseTest)
rmse_Test = metrics.mean_squared_error(yTestG[:,0], yHatCaseTest,  squared=False)

print("r2_Test growth: ", r2_Test)
print("r2_Train growth: ", r2_Train)
print("rmse_Test growth: ", rmse_Test)
print("rmse_Train growth: ", rmse_Train)"""



df = df.to_numpy()

xData = df[:,3:-1]

scaler = StandardScaler()
scaler.fit(xData)

yData = df[:,-1]



xTrain, xTest, yTrain, yTest = model_selection.train_test_split(xData, yData, test_size=0.3, random_state=42)


#CV cross validation for Ridge and Lasso Regularization
"""xTrain = xTrain[:-14]
yTrain = yTrain[14:]

xTest = xTest[:-14]
yTest = yTest[14:]

print (xTrain.shape)
print (yTrain.shape)"""
"""model = Lasso()

alpha = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
param_grid = dict(alpha=alpha)
grid = GridSearchCV(model, param_grid=param_grid, scoring ='neg_root_mean_squared_error', verbose=1, n_jobs=-1, cv=5)
grid_result = grid.fit(xTrain, yTrain)

print('Best Score: ', grid_result.best_score_)
print('Best Params: ', grid_result.best_params_)

model = Lasso()

alpha = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
param_grid = dict(alpha=alpha)
grid = GridSearchCV(model, param_grid=param_grid, scoring ='r2', verbose=1, n_jobs=-1, cv=5)
grid_result = grid.fit(xTrain, yTrain)

print('Best Score: ', grid_result.best_score_)
print('Best Params: ', grid_result.best_params_)

model = Ridge()

alpha = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
param_grid = dict(alpha=alpha)
grid = GridSearchCV(model, param_grid=param_grid, scoring ='r2', verbose=1, n_jobs=-1, cv=5)
grid_result = grid.fit(xTrain, yTrain)

print('Best Score: ', grid_result.best_score_)
print('Best Params: ', grid_result.best_params_)

model = Ridge()

alpha = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
param_grid = dict(alpha=alpha)
grid = GridSearchCV(model, param_grid=param_grid, scoring ='neg_root_mean_squared_error', verbose=1, n_jobs=-1, cv=5)
grid_result = grid.fit(xTrain, yTrain)

print('Best Score: ', grid_result.best_score_)
print('Best Params: ', grid_result.best_params_)"""



#Unregularized
regr = LinearRegression()

regr.fit(xTrain, yTrain)

y_pred = regr.predict(xTest)
for i in range (y_pred.shape[0]):
	if (y_pred[i] < 0):
		y_pred[i] = 0

print ("unregularized")
print("test rmse: ", metrics.mean_squared_error(yTest, y_pred, squared=False))
print (regr.coef_)
print("test r2: ", metrics.r2_score(yTest, y_pred))


y_train = regr.predict(xTrain)
print("train rmse:", metrics.mean_squared_error(yTrain, y_train, squared=False))
print("test rmse:", metrics.r2_score(yTrain, y_train)) 

pd.DataFrame(regr.coef_).to_csv("beta3.csv")



"""regr = linear_model.LassoCV(cv=5, random_state=0)

regr.fit(xTrain, yTrain)

y_pred = regr.predict(xTest)
for i in range (y_pred.shape[0]):
	if (y_pred[i] < 0):
		y_pred[i] = 0

print ("lasso alpha 0")
print("test rmse: ", metrics.mean_squared_error(yTest, y_pred, squared=False))
print (regr.coef_)
print("test r2: ", metrics.r2_score(yTest, y_pred))


y_train = regr.predict(xTrain)
print("train rmse:", metrics.mean_squared_error(yTrain, y_train, squared=False))
print("test rmse:", metrics.r2_score(yTrain, y_train)) """

#Lasso with 0.001 which is optimal
regr = linear_model.Lasso(alpha = 0.001)

regr.fit(xTrain, yTrain)

y_pred = regr.predict(xTest)
for i in range (y_pred.shape[0]):
	if (y_pred[i] < 0):
		y_pred[i] = 0

print ("lasso alpha 0.001")
print("test rmse: ", metrics.mean_squared_error(yTest, y_pred, squared=False))
print (regr.coef_)
print("test r2: ", metrics.r2_score(yTest, y_pred))


y_train = regr.predict(xTrain)
print("train rmse:", metrics.mean_squared_error(yTrain, y_train, squared=False))
print("test rmse:", metrics.r2_score(yTrain, y_train)) 


regr = linear_model.Ridge(alpha = 10)

regr.fit(xTrain, yTrain)

y_pred = regr.predict(xTest)
for i in range (y_pred.shape[0]):
	if (y_pred[i] < 0):
		y_pred[i] = 0

print ("ridge alpha 10")
print("test rmse: ", metrics.mean_squared_error(yTest, y_pred, squared=False))
print (regr.coef_)
print("test r2: ", metrics.r2_score(yTest, y_pred))


y_train = regr.predict(xTrain)
print("train rmse:", metrics.mean_squared_error(yTrain, y_train, squared=False))
print("test rmse:", metrics.r2_score(yTrain, y_train)) 

regr = linear_model.RidgeCV(cv=5, random_state=0)

regr.fit(xTrain, yTrain)

y_pred = regr.predict(xTest)
for i in range (y_pred.shape[0]):
	if (y_pred[i] < 0):
		y_pred[i] = 0

print ("ridge alpha 1.0")
print("test rmse: ", metrics.mean_squared_error(yTest, y_pred, squared=False))
print (regr.coef_)
print("test r2: ", metrics.r2_score(yTest, y_pred))


y_train = regr.predict(xTrain)
print("train rmse:", metrics.mean_squared_error(yTrain, y_train, squared=False))
print("test rmse:", metrics.r2_score(yTrain, y_train)) 




#pd.DataFrame(y_pred).to_csv("y_pred.csv")
#pd.DataFrame(yTest).to_csv("yTest.csv")

