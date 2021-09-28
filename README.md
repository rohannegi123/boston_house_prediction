# boston_house_prediction
Context
 - To Explore more on Regression Algorithm
 
# Introduction
In this project, we will develop and evaluate the performance and the predictive power of a model trained and tested on data collected from houses in Boston’s suburbs.
Once we get a good fit, we will use this model to predict the monetary value of a house located at the Boston’s area.
A model like this would be very valuable for a real state agent who could make use of the information provided in a daily basis.


# Dataset

I have taken Housing dataset which contains information about different houses in Boston. This data was originally a part of UCI Machine Learning Repository and has been removed now. We can also access this data from the scikit-learn library. There are 506 samples and 13 feature variables in this dataset. The objective is to predict the value of prices of the house using the given features.The following describes the dataset columns:
CRIM - per capita crime rate by town
ZN - proportion of residential land zoned for lots over 25,000 sq.ft.;
INDUS - proportion of non-retail business acres per town.;
CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise);
NOX - nitric oxides concentration (parts per 10 million);
RM - average number of rooms per dwelling;
AGE - proportion of owner-occupied units built prior to 1940;
DIS - weighted distances to five Boston employment centres;
RAD - index of accessibility to radial highways;
TAX - full-value property-tax rate per $10,000;
PTRATIO - pupil-teacher ratio by town;
B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town;
LSTAT - % lower status of the population;
MEDV - Median value of owner-occupied homes in $1000's  (Target)


# Liner Regression models
I have build 6 models in the project-
* Linear regression
* Lasso 
* Ridge
* Elastic Regressor
* XGBoost
* Random forest

The random forest model has the highest accuracy and best prediction , So i have used the random forest model for the prediction




# Project deployed in heroku 
- https://bostomn-house-prediction.herokuapp.com/
