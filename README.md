# boston_house_prediction
Context
 - To Explore more on Regression Algorithm
 
#Introduction
In this project, we will develop and evaluate the performance and the predictive power of a model trained and tested on data collected from houses in Boston’s suburbs.
Once we get a good fit, we will use this model to predict the monetary value of a house located at the Boston’s area.
A model like this would be very valuable for a real state agent who could make use of the information provided in a daily basis.


# linear_regression_models

I have taken Housing dataset which contains information about different houses in Boston. This data was originally a part of UCI Machine Learning Repository and has been removed now. We can also access this data from the scikit-learn library. There are 506 samples and 13 feature variables in this dataset. The objective is to predict the value of prices of the house using the given features.

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
