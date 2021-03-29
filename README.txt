Original data folder:
- Average Household Size and Population Density - County.csv
- us_covid19_states_policies.csv
- us-m50.csv
- Race and Ethnicity - County.csv
- 2016 County Election Data.csv
- covid_confirmed_usafacts.csv
- data.csv

Data Cleaning folder:
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


Models folder:
- lr.py (builds linear regression models and performs hyperparameter tuning) 
- decisionTrees.py (performs additional preprocessing and builds decision tree models)
- randomForests.py (performs additional preprocessing and builds random forest models)
- ensembleFeatSelect.py (selects significant features from linear regression for models in the previous two python files)


