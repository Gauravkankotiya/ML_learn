#### ML notes ####

'''Your raw data may not be setup to be in the best shape for modeling.
Sometimes you need to preprocess your data in order to best present the 
inherent structure of the problem in your data to the modeling algorithms'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler().fit(x)# this method is used for preprocessing the data


# # Cross Validation Classification LogLoss
# from pandas import read_csv
# from sklearn.model_selection import KFold
# from sklearn.model_selection import cross_val_score
# from sklearn.linear_model import LogisticRegression
# url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
# names = ['preg', 'plas', 'pres', 'skin',
#          'test', 'mass', 'pedi', 'age', 'class']
# dataframe = read_csv(url, names=names)
# array = dataframe.values
# X = array[:, 0:8]
# Y = array[:, 8]
# kfold = KFold(n_splits=10, random_state=7, shuffle=True)
# model = LogisticRegression(solver='liblinear')
# scoring = 'neg_log_loss'
# # for accuracy we can use accuracy in terms of scoring
# results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
# print(f"Logloss: {results.mean()} {results.std()}")


# Improving accuracy of algo by adjusting parameters

# Grid search ###
# Grid Search for Algorithm Tuning
from pandas import read_csv
import numpy
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin',
         'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]
alphas = numpy.array([1, 0.1, 0.01, 0.001, 0.0001, 0])
param_grid = dict(alpha=alphas)  # set alpha as key for alphas array
model = Ridge()
# grid searach try all parameters passed in param_grid and gives best feasible result for algo
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid.fit(X, Y)
print(grid.best_score_)
print(grid.best_estimator_.alpha)


######## Ensamble Predictions #########

# Random Forest Classification
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin',
         'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]
num_trees = 100
max_features = 3
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = RandomForestClassifier(
    n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
