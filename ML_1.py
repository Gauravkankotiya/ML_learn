'''Hello World in Machine Learning'''


from pandas import read_csv
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from matplotlib import pyplot as plt

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# print(dataset.shape)  # it will gives us an idea of rows and columns in dataset
# print(dataset.head(20))  # it'll print first 20 lines of dataset
# print(dataset.describe())  # this will gives all statistial info of data
# # this'll gives us number of instances in class column
# print(dataset.groupby('class').size())

## visulizing the data##
# dataset.plot(kind='box', subplots=True, layout=(
#     2, 2), sharex=False, sharey=False)
# plt.show()
# # histogram
# dataset.hist()
# plt.show()

# scatter plot matrix
# scatter_matrix(dataset)
# plt.show()

#### Evaluate some algorithms#####

# create a validation dataset

array = dataset.values  # returns only values of dataset not headers

print(array)
x = array[:, 0:4]
y = array[:, 4]
x_train, x_validation, y_train, y_validation = train_test_split(
    x, y, test_size=0.20, random_state=1)  # this method use to split the train and test models
models = []
models.append(('LR', LogisticRegression(
    solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(
        model, x_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(f'{name} : {cv_results.mean()} ({cv_results.std()})')

plt.boxplot(results, labels=names)
plt.title("Algorithm Comparison")
plt.show()

# select best fitting model for dataset by its accuracy(mean)
# make prediction on validatoin dataset

model = SVC(gamma='auto')
model.fit(x_train, y_train)
predictions = model.predict(x_validation)

# evaluate the predictoins
print(accuracy_score(y_validation, predictions))
print(confusion_matrix(y_validation, predictions))
print(classification_report(y_validation, predictions))
