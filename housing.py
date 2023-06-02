# %%
# imports
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from scipy import stats

# looking at the dataframe.
df = pd.read_csv('housing.csv')
display(df.head())
display(df.info())
display(df["ocean_proximity"].value_counts())
display(df.describe())
df.hist(bins=50, figsize=(20, 15))

# function to split dataset into training set and testing set.
def splitTrainTest(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

# setting values to split
train_set, test_set = splitTrainTest(df, 0.2)
display(len(train_set))
display(len(test_set))

# calculates a short fixed length binary sequence and forms a codeword.
# multiple codewords are compared with residue constants to ensure that
# the block does not contain any data errors. same residue (remainder) = 
# no error. 
# returns boolean
def testSetCheck(identifier, testRatio):
    return crc32(np.int64(identifier)) & 0xffffffff < testRatio * 2**32

def splitTrainTestById(data, testRatio, idColumn):
    ids = data[idColumn]
    inTestSet = ids.apply(lambda id_: testSetCheck(id_, testRatio))
    return data.loc[~inTestSet], data.loc[inTestSet]

housingWithId = df.reset_index()
train_set, test_set = splitTrainTestById(housingWithId, 0.2, "index")

# same as splitTrainTest but with machine learning.
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

# %%
# spreads out median income histogram for us to understand it better.
df["income_cat"] = pd.cut(df["median_income"], 
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
display(df["income_cat"].hist())
# plt.show(df["income_cat"].hist())

# %%
# stratified sampling based on the income category.
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for trainIndex, testIndex in split.split(df, df["income_cat"]):
    stratTrainSet = df.loc[trainIndex]
    stratTestSet = df.loc[testIndex]
    
# seeing if it worked.
stratTestSet["income_cat"].value_counts() / len(stratTestSet)

# %%
# removing income cat attribute so the data is back to its og state.
for set_ in (stratTrainSet, stratTestSet):
    set_.drop("income_cat", axis='columns', inplace=True)

# %%
# making a cpoy of the training set to manipulate it without harming the og.
df = stratTrainSet.copy()

# %%
# scatterplot to visualize geographical info. 
# alpha manipulates the opacity
df.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)

# %%
# including housing prices in our scatterplot
# cmap = 'jet' blue (low) -> red (high)
# s = size of markers
# c = array of values for marker colors
df.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, 
    s=df["population"]/100, label="population", figsize=(10, 7), 
    c="median_house_value", cmap=plt.get_cmap('jet'), colorbar=True)
plt.legend()

# %%
# tells us the correlation of other attributes with the median house value
# 1 = strong positive correlation, 0 = no correlation, 
# -1 = strong negative correlation
# only measures linear correlation (2 vars)
corrMatrix = df.corr()
corrMatrix["median_house_value"].sort_values(ascending=False)

# %%
# scatter matrix for correlation constant
attributes = ["median_house_value", "median_income", "total_rooms",
    "housing_median_age"]
scatter_matrix(df[attributes], figsize=(12, 8))

# %%
# zooming in on median_income X median_house_value
df.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.1)

# %%
# more correlations
df["rooms_per_household"] = df["total_rooms"]/df["households"]
df["bedrooms_per_room"] = df["total_bedrooms"]/df["total_rooms"]
df["population_per_household"] = df["population"]/df["households"]

corrMatrix = df.corr()
corrMatrix["median_house_value"].sort_values(ascending=False)

# %%
# preparing data for ML algorithms
df = stratTrainSet.drop("median_house_value", axis='columns')
dfLabels = stratTrainSet["median_house_value"].copy()

# %%
# # data cleaning
# # total_bedrooms has some missing attributes
# df.drop("total_bedrooms", axis='columns')
imputer = SimpleImputer(strategy='median')
# this only works on numerical attributes, therefore 
# getting rid of ocean_proximity
dfNums = df.drop("ocean_proximity", axis='columns')
# fitting this instance of the training data
imputer.fit(dfNums)
imputer.statistics_
dfNums.median().values
X = imputer.transform(dfNums)
housing_tr = pd.DataFrame(X, columns=dfNums.columns)

# %%
# converting ocean_proximity obj attributes to numerical attributes
housingCat = df[["ocean_proximity"]]
display(housingCat.head(10))
ordinalEncoder = OrdinalEncoder()
housingCatEncoded = ordinalEncoder.fit_transform(housingCat)
housingCatEncoded[:10]
ordinalEncoder.categories_

# %%
# converting categorical values into one hot vectors
# creating an instance of one hot encoder
catEncoder = OneHotEncoder()
HousingCat1Hot = catEncoder.fit_transform(housingCat)
HousingCat1Hot

# %%
# assigning indices
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
# custom transformer
# named attribsAdder for pipeline
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    # constructor method (self = this)
    def __init__(self, addBedroomsPerRoom=True):
        self.addBedroomsPerRoom = addBedroomsPerRoom
    # returns instance of transformer
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.addBedroomsPerRoom:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, 
                bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
    
attr_adder = CombinedAttributesAdder(addBedroomsPerRoom=False)
housing_extra_attribs = attr_adder.transform(df.values)

# %%
# pipeline to specify the order of steps

numPipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('attribsAdder', CombinedAttributesAdder()),
    ('stdScaler', StandardScaler())
])
housing_num_tr = numPipeline.fit_transform(dfNums)

# %%
# full pipeline to apply all transformations on housing data
numAttribs = list(dfNums)
catAttribs = ["ocean_proximity"]

fullPipeline = ColumnTransformer([
    ('num', numPipeline, numAttribs),
    ('cat', OneHotEncoder(), catAttribs)
])

housingPrepared = fullPipeline.fit_transform(df)

# %%
# creating a working linear regression model
# instance
linReg = LinearRegression()
linReg.fit(housingPrepared, dfLabels)

# %%
# trying the model out on a few instances from the training set
someData = df.iloc[:5]
someLabels = dfLabels.iloc[:5]
someDataPrepared = fullPipeline.transform(someData)
print("Predictions: ", linReg.predict(someDataPrepared))
print("Labels: ", list(someLabels))

# %% 
# calculating the RMSE
housingPredictions = linReg.predict(housingPrepared)
linMSE = mean_squared_error(dfLabels, housingPredictions)
linRMSE = np.sqrt(linMSE)
linRMSE
# conclusion: underfitting

# %%
# creating a working decision tree regressor model
# instance
decReg = DecisionTreeRegressor()
decReg.fit(housingPrepared, dfLabels)

# %%
# evaluating the model out on a few instances from the training set
housingPredictions = decReg.predict(housingPrepared)
treeMSE = mean_squared_error(dfLabels, housingPredictions)
treeRMSE = np.sqrt(treeMSE)
treeRMSE
# conclusion: possibly overfitting, not the best way to evaluate tree model

# %%
# evaluating decision tree scores using k-fold cross validation feature
scores = cross_val_score(decReg, housingPrepared, dfLabels,
    scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)

# evaluating linear regression scores using k-fold cross validation feature

lin_scores = cross_val_score(linReg, housingPrepared, dfLabels,
    scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

# %%
# ensemble learning
# forest regression averages out decision trees
forestReg = RandomForestRegressor()
forestReg.fit(housingPrepared, dfLabels)
housingPredictions = forestReg.predict(housingPrepared)
forestMSE = mean_squared_error(dfLabels, housingPredictions)
forestRMSE = np.sqrt(forestMSE)
forestRMSE

forest_scores = cross_val_score(forestReg, housingPrepared, dfLabels,
    scoring='neg_mean_squared_error', cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)

# %%
# grid search cv
# parameter grid takes in 2 dictionaries with parameter names as keys and 
# lists of parameter values.
paramGrid = [
 {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
 {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
 ]
# instance of forest regressor
forestReg = RandomForestRegressor()
# performs grid search with cross validation
# cv specifies number of cross validation folds
# return train scores gives the training scores in the results
gridSearch = GridSearchCV(forestReg, paramGrid, cv=5, 
    scoring='neg_mean_squared_error', return_train_score=True)

# %%
# fits results of the grid search, like best hyperparameters and 
# performance scores for each combination of hyperparameters. 
gridSearch.fit(housingPrepared, dfLabels)

# %%
# getting the best combination of parameters
gridSearch.best_params_

# %%
# getting the best estimator
display(gridSearch.best_estimator_)

# %%
# getting the evaluation scores
cvRes = gridSearch.cv_results_

for i, j in zip(cvRes['mean_test_score'], cvRes['params']):
    print(np.sqrt(-i), j)
    
# %%
# calculates a score for all input features for a given model. this score
# calculates the importance of each feature. the higher the score, the 
# higher the effect of the feature on the given model that is being used to
# predict a specific variable
feature_importances = gridSearch.best_estimator_.feature_importances_
feature_importances

# %%
# displaying importance scores next to their attribute names.
extra_attribs = ['rooms_per_hhold', 'pop_per_hhold', 'bedrooms_per_room']
cat_encoder = fullPipeline.named_transformers_['cat']
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = numAttribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)

# %%
# evlauating the final model on the test set
final_model = gridSearch.best_estimator_
X_test = stratTestSet.drop('median_house_value', axis='columns')
y_test = stratTestSet['median_house_value'].copy()
X_test_prepared = fullPipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse

# %%
# abstraction at its peak
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1, 
    loc=squared_errors.mean(), scale=stats.sem(squared_errors)))

# %% 
'''AND WE ARE DONE!!!'''
