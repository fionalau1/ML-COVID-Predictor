# README

This is a COVID-19 cases predictor by U.S. county. The machine learning model incorporates 32 different features and utilizes linear regression, decision tree, 
random forest, and other ensemble methods to predict future number of cases. The data includes 350,000 records and 70% was used for training and 30% was used 
for testing.

## How to run: 

````
git clone https://github.com/fionalau1/ML-COVID-Predictor.git
cd Models
python lr.py
python decisionTrees.py 
python randomForests.py
````

### Output:
Each python file corresponds to one of the linear regression, decision tree, or random forest models. Note: the linear regression output is used to inform
other ensemble methods within decisionTrees.py and randomForests.py.
After running ly.py, decisionTrees.py or randomForests.py, the output of the regression results and performance will be printed in the terminal. 


### Models folder:
- lr.py (builds linear regression models and performs hyperparameter tuning) 
- decisionTrees.py (performs additional preprocessing, builds decision tree models, builds other ensemble models)
- randomForests.py (performs additional preprocessing,  builds random forest models, builds other ensemble models)
- ensembleFeatSelect.py (performs significance testing, selects important features from linear regression for ensemble methods)

### Original data folder:
- Average Household Size and Population Density - County.csv
- us_covid19_states_policies.csv
- us-m50.csv
- Race and Ethnicity - County.csv
- 2016 County Election Data.csv
- covid_confirmed_usafacts.csv
- data.csv

### Data Cleaning folder:
- demographicOld.csv
- DemographicsKNN.py (reads demographicOld.csv, fills in missing values, and exports demographic.csv)
- mobilityOld.csv
- MobilityKNN.py (reads mobilityOld.csv, fills in missing values, and exports mobility.csv)
- cases.csv
- policies.csv
- demographic.csv
- mobility.csv
- dataCleaning.py (reads in cases.csv, policies.csv, demographic.csv, and mobility.csv and combines them into one final labeled dataset called labeledData.csv)
- labeledData.csv
- labeledData1.csv (different to labeledData.csv because government policies are set to binary instead of dates. Used specifically for linear regression models)

### Data Sources:
1) https://covid19.census.gov/datasets/21843f238cbb46b08615fc53e19e0daf_1/data?geometry=38.109%2C-0.672%2C-37.477%2C76.524&selectedAttribute=B25010_001E
(Population Density, Population Size, Number of Households) 

2) https://www.kaggle.com/c/mapping-the-impact-of-policy-on-covid-19-outbreaks/data
(COVID containment policies for all 50 states, Max-Distance Mobility)

3) https://covid19.census.gov/datasets/race-and-ethnicity-county/data
(Race/Ethnicity)

4) https://www.kaggle.com/johnwdata/2016-election-county-election-data
(Political Affiliation)

5) https://usafacts.org/visualizations/coronavirus-covid-19-spread-map/
(Number of daily COVID cases between 1/22/20 - 10/05/20 for each county in the U.S.)

6) https://www.kaggle.com/ady123/us-counties-covid19-dataset?select=data.csv
(Income, Poverty, Education Level)
