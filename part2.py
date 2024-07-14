# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
import numpy as np
import pandas as pd
import rfit

# world1 = rfit.dfapi('World1', 'id')
# world1.to_csv("world1.csv")
world1 = pd.read_csv("world1.csv", index_col="id") # use this instead of hitting the server if csv is on local
# world2 = rfit.dfapi('World2', 'id')
# world2.to_csv("world2.csv")
world2 = pd.read_csv("world2.csv", index_col="id") # use this instead of hitting the server if csv is on local


print("\nReady to continue.")

#%% [markdown]
# # Two Worlds (Continuation from midterm: Part I - 25%)
# 
# In the midterm project, we used statistical tests and visualization to 
# study these two worlds. Now let us use the modeling techniques we now know
# to give it another try. 
# 
# Use appropriate models that we learned in this class or elsewhere, 
# elucidate what the current state of each world looks like. 
# 
# However, having an accurate model does not tell us if the worlds are 
# utopia or not. Is it possible to connect these concepts together? (Try something called 
# "feature importance")
# 
# Data dictionary:
# * age00: the age at the time of creation. This is only the population from age 30-60.  
# * education: years of education they have had. Education assumed to have stopped. A static data column.  
# * marital: 0-never married, 1-married, 2-divorced, 3-widowed  
# * gender: 0-female, 1-male (for simplicity)  
# * ethnic: 0, 1, 2 (just made up)  
# * income00: annual income at the time of creation   
# * industry: (ordered with increasing average annual salary, according to govt data.)   
#   0. leisure n hospitality  
#   1. retail   
#   2. Education   
#   3. Health   
#   4. construction   
#   5. manufacturing   
#   6. professional n business   
#   7. finance   
# 
#%%
# Basic Overview
print("World1 Data Preview:")
print(world1.head())
print("\nWorld1 Data Types and Missing Values:")
print(world1.info())
# Summary Statistics
print("\nWorld1 Summary Statistics:")
print(world1.describe())

#%%
print("\nWorld2 Data Preview:")
print(world2.head())
print("\nWorld2 Data Types and Missing Values:")
print(world2.info())
print("\nWorld2 Summary Statistics:")
print(world2.describe())

#%%
import seaborn as sns
import matplotlib.pyplot as plt




#%%
# WORLD 1
# Education Analysis
plt.figure(figsize=(10, 5))
sns.countplot(x='education', data=world1)
plt.title('Education Level Distribution in World1')
plt.xlabel('Years of Education')
plt.ylabel('Count')
plt.show()

# Gender Analysis
plt.figure(figsize=(10, 5))
sns.countplot(x='gender', data=world1)
plt.title('Gender Distribution in World1')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# Ethnic Analysis
plt.figure(figsize=(10, 5))
sns.countplot(x='ethnic', data=world1)
plt.title('Ethnic Distribution in World1')
plt.xlabel('Ethnic Group')
plt.ylabel('Count')
plt.show()

# Industry Analysis
plt.figure(figsize=(10, 5))
sns.countplot(x='industry', data=world1)
plt.title('Industry Distribution in World1')
plt.xlabel('Industry Code')
plt.ylabel('Count')
plt.show()

# Age Analysis
plt.figure(figsize=(10, 5))
sns.histplot(world1['age00'], kde=True)
plt.title('Age Distribution in World1')
plt.xlabel('Age')
plt.ylabel('Density')
plt.show()

# Correlation Analysis
corr_matrix_world1 = world1.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix_world1, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix for World1')
plt.show()

#%%
# WORLD 2
# Education Analysis
plt.figure(figsize=(10, 5))
sns.countplot(x='education', data=world2)
plt.title('Education Level Distribution in World2')
plt.xlabel('Years of Education')
plt.ylabel('Count')
plt.show()

# Gender Analysis
plt.figure(figsize=(10, 5))
sns.countplot(x='gender', data=world2)
plt.title('Gender Distribution in World2')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# Ethnic Analysis
plt.figure(figsize=(10, 5))
sns.countplot(x='ethnic', data=world2)
plt.title('Ethnic Distribution in World2')
plt.xlabel('Ethnic Group')
plt.ylabel('Count')
plt.show()

# Industry Analysis
plt.figure(figsize=(10, 5))
sns.countplot(x='industry', data=world2)
plt.title('Industry Distribution in World2')
plt.xlabel('Industry Code')
plt.ylabel('Count')
plt.show()

# Age Analysis
plt.figure(figsize=(10, 5))
sns.histplot(world2['age00'], kde=True)
plt.title('Age Distribution in World2')
plt.xlabel('Age')
plt.ylabel('Density')
plt.show()

# Correlation Analysis
corr_matrix_world2 = world2.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix_world2, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix for World2')
plt.show()

# INCOME
plt.figure(figsize=(18, 12))

# Histograms
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(18, 15))
variables = ['age00', 'education', 'income00']
for i, var in enumerate(variables):
    sns.histplot(world1[var], bins=30, kde=True, ax=axes[i, 0], color='skyblue').set_title(f'World1 {var} Distribution')
    sns.histplot(world2[var], bins=30, kde=True, ax=axes[i, 1], color='lightgreen').set_title(f'World2 {var} Distribution')

# Boxplots for income by categories
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
categories = ['gender', 'marital', 'ethnic', 'industry']
for i, cat in enumerate(categories):
    sns.boxplot(x=cat, y='income00', data=world1, ax=axes[0, i]).set_title(f'World1 Income by {cat.title()}')
    sns.boxplot(x=cat, y='income00', data=world2, ax=axes[1, i]).set_title(f'World2 Income by {cat.title()}')

plt.tight_layout()

# Correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(world1.corr(), annot=True, cmap='coolwarm').set_title('World1 Correlation Matrix')
plt.show()

plt.figure(figsize=(12, 10))
sns.heatmap(world2.corr(), annot=True, cmap='coolwarm').set_title('World2 Correlation Matrix')
plt.show()
#%% [markdown]
# HISTOGRAM

# - Both World1 and World2 have a fairly uniform distribution of age, which suggests a balanced representation 
# across the 30-60 age range.

# - There are peaks at certain education levels, possibly indicating standard education stopping points (like high school, bachelor's, and master's degrees).

# - Income in both worlds appears right-skewed, meaning there is a larger proportion of the population earning a lower income, with fewer individuals 
# earning higher incomes.

# BOXPLOT

# - In both worlds, males (1) appear to have a higher income distribution than females (0), though the difference seems less pronounced in World2.

# - Income by Marital Status: There does not appear to be a stark difference in income based on marital status, though single (0) and divorced (2) individuals 
# might have slightly lower median incomes in World1.

# - Income by Ethnicity: World1 shows some variation in income across ethnic groups, which is less apparent in World2.

# - Income by Industry: There's a clear upward trend in median income as industry codes increase, reflecting the ordering of industries 
# by average annual salary. Higher industry codes, representing fields like finance, tend to have a higher income range.

# CORRELATION

# - The strongest correlation in both worlds is between industry and income, which is very high (0.89). This suggests that industry is a major determinant 
# of income levels.

# - There are minor differences in correlations with gender and ethnic in both worlds, but these are not as strong as the correlation with industry.


#%%
# MODELING

"""
The choice to target income in the predictive model stems from income being a quantifiable and commonly used indicator of economic well-being in a society.
Also with the data that we have for World1 and World2, income has been the main focus because it's a clear numerical outcome that can be directly predicted.

Education as a target could reveal the factors that contribute to higher educational attainment. 

Predicting the industry sector in which individuals work can give insights into the workforce distribution and how demographic factors influence career.
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

#%%
# W1 Income Random Forest

# Dropping 'income00' to use as the target variable
X = world1.drop('income00', axis=1) 
y = world1['income00']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Prediction
y_pred = rf.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R^2 Score: {r2}')


feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index = X_train.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)


plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y=feature_importances.index, data=feature_importances)
plt.title('Feature Importances in World1')
plt.show()

#%% [markdown]
# MSE and RMSE: These values indicate the average error of the model's predictions. The RMSE, in particular, tells us that, on average, 
# the model's predictions are about 11,611.84 units (the currency isn't specified) away from the actual income values. 

# R² Score: An R² score of 0.828 means that approximately 82.8% of the variation in the income variable can be explained by the model's inputs. 
# This is generally considered a strong score, suggesting that the model has a good fit to the data.

# Feature Importance:
# - Industry: It has the highest importance score by a significant margin, confirming the earlier EDA finding that the industry is a major determinant 
# of income levels in World1.
# - Age and Education: The next two important features are age and education, but their importance scores are notably lower than that of the industry, 
# which suggests they have a much smaller impact on income.
# - Marital status, ethnicity, and gender have even lower importance scores, indicating they have minimal impact on income relative to industry in this model.


#%%
# W2 Income Random Forest

# Dropping 'income00' to use as the target variable
X = world2.drop('income00', axis=1) 
y = world2['income00']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Prediction
y_pred = rf.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R^2 Score: {r2}')


feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index = X_train.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)


plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y=feature_importances.index, data=feature_importances)
plt.title('Feature Importances in World2')
plt.show()

#%% [markdown]

# The MSE and RMSE are slightly higher for World2, suggesting the model predictions for World2 are, on average, a bit further from the true income values 
# than for World1. The R² score is also slightly lower, indicating that the model for World2 explains a bit less of the variance in income compared to World1.

# Feature Importance:
# - Industry is, once again, the most significant predictor of income. This reinforces the findings from World1 and suggests that the type of industry 
# an individual works in is the most determining factor for income in both worlds.
# - Age and education follow in importance but are significantly less influential than the industry, mirroring the results from World1.


#%%
# W1 Education RF
from sklearn.ensemble import RandomForestClassifier


X = world1.drop(['income00', 'education'], axis=1)
y = world1['education']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)


rf_classifier.fit(X_train, y_train)

importances = rf_classifier.feature_importances_

indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12, 8))
plt.title('World1 Feature Importances for Predicting Education Level')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()

#%%
# W2 Education RF
X = world2.drop(['income00', 'education'], axis=1)
y = world2['education']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)


rf_classifier.fit(X_train, y_train)

importances = rf_classifier.feature_importances_

indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12, 8))
plt.title('World2 Feature Importances for Predicting Education Level')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()

#%%
# W1 Industry RF
from sklearn.metrics import accuracy_score, classification_report

X = world1.drop(['income00', 'industry'], axis=1)  # drop 'industry' to predict it
y = world1['industry']  # 'industry' as the target variable

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'World1 Classification Accuracy: {accuracy}')
print(class_report)

# Get and plot feature importances
importances = rf_classifier.feature_importances_
sorted_indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 8))
plt.title('World1 Feature Importances for Predicting Industry')
plt.bar(range(X_train.shape[1]), importances[sorted_indices], align='center')
plt.xticks(range(X_train.shape[1]), X_train.columns[sorted_indices], rotation=90)
plt.tight_layout()
plt.show()

#%%
# W2 Industry RF
from sklearn.metrics import accuracy_score, classification_report

X = world2.drop(['income00', 'industry'], axis=1)  # drop 'industry' to predict it
y = world2['industry']  # 'industry' as the target variable

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'World2 Classification Accuracy: {accuracy}')
print(class_report)

# Get and plot feature importances
importances = rf_classifier.feature_importances_
sorted_indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 8))
plt.title('World2 Feature Importances for Predicting Industry')
plt.bar(range(X_train.shape[1]), importances[sorted_indices], align='center')
plt.xticks(range(X_train.shape[1]), X_train.columns[sorted_indices], rotation=90)
plt.tight_layout()
plt.show()

#%% [markdown]
# - Both worlds appear to have economies where the type of industry  heavily influences  income. 

# - Age as a predictor for both industry and education in both worlds suggests a dynamic where the workforce's distribution across industries 
# and education levels changes with age demographics, possibly reflecting the natural progression of careers and educational access over time.

# - The current state of both World1 and World2 is characterized by an economy that rewards industry affiliation significantly,
# alongside a notable influence of age on both industry and education. The nuanced relationship between demographic features and sectors of employment 
# speaks to a need to promote diversity and equal opportunity across all industries and age groups to move towards a more utopian society.



#%% [markdown]
#
# # Free Worlds (Continuation from midterm: Part II - 25%)
# 
# To-do: Complete the method/function "predictFinalIncome" towards the end of this Part II codes.  
#  
# The worlds are gifted with freedom. Sort of.  
# I have a model built for them. It predicts their MONTHLY income/earning growth, 
# base on the characteristics of the individual. Your task is to first examine and 
# understand the model. If you don't like it, build your own world and own model. 
# For now, please help me finish the last piece.  
# 
# My model will predict what the growth factor for each person is in the immediate month ahead. 
# Along the same line, it also calculates what the expected (average) salary will be after 1 month with 
# that growth rate. You need to help make it complete, by producing a method/function that will 
# calculate what the salary will be after n months. (Method: predictFinalIncome )  
# 
# Then try this model on people like Plato, and also create a few hypothetical characters
# with different demographic traits, and see what their growth rates / growth factors are.
# Use the sample codes after the class definition below.  
# 
#%%
class Person:
  """ 
  a person with properties in the utopia 
  """

  def __init__(self, personinfo):
    self.age00 = personinfo['age00'] # age at creation or record. Do not change.
    self.age = personinfo['age00'] # age at current time. 
    self.income00 = personinfo['income00'] # income at creation or record. Do not change.
    self.income = personinfo['income00'] # income at current time.
    self.education = personinfo['education']
    self.gender = personinfo['gender']
    self.marital = personinfo['marital']
    self.ethnic = personinfo['ethnic']
    self.industry = personinfo['industry']
    # self.update({'age00': self.age00, 
    #         'age': self.age,
    #         'education': self.education,
    #         'gender': self.gender,
    #         'ethnic': self.ethnic,
    #         'marital': self.marital,
    #         'industry': self.industry,
    #         'income00': self.income00,
    #         'income': self.income})
    return
  
  def update(self, updateinfo):
    for key,val in updateinfo.items():
      if key in self.__dict__ : 
        self.__dict__[key] = val
    return
        
  def __getitem__(self, item):  # this will allow both person.gender or person["gender"] to access the data
    return self.__dict__[item]

  
#%%  
class myModel:
  """
  The earning growth model for individuals in the utopia. 
  This is a simplified version of what a model could look like, at least on how to calculate predicted values.
  """

  # ######## CONSTRUCTOR  #########
  def __init__(self, bias) :
    """
    :param bias: we will use this potential bias to explore different scenarios to the functions of gender and ethnicity

    :param b_0: the intercept of the model. This is like the null model. Or the current average value. 

    :param b_age: (not really a param. it's more a function/method) if the model prediction of the target is linearly proportional to age, this would the constant coefficient. In general, this does not have to be a constant, and age does not even have to be numerical. So we will treat this b_age as a function to convert the value (numerical or not) of age into a final value to be combined with b_0 and the others 
    
    :param b_education: similar. 
    
    :param b_gender: similar
    
    :param b_marital: these categorical (coded into numeric) levels would have highly non-linear relationship, which we typically use seaparate constants to capture their effects. But they are all recorded in this one function b_martial
    
    :param b_ethnic: similar
    
    :param b_industry: similar
    
    :param b_income: similar. Does higher salary have higher income or lower income growth rate as lower salary earners?
    """

    self.bias = bias # bias is a dictionary with info to set bias on the gender function and the ethnic function

    # ##################################################
    # The inner workings of the model below:           #
    # ##################################################

    self.b_0 = 0.0023 # 0.23% MONTHLY grwoth rate as the baseline. We will add/subtract from here

    # Technically, this is the end of the constructor. Don't change the indent

  # The rest of the "coefficients" b_1, b_2, etc are now disguised as functions/methods
  def b_age(self, age): # a small negative effect on monthly growth rate before age 45, and slight positive after 45
    effect = -0.00035 if (age<40) else 0.00035 if (age>50) else 0.00007*(age-45)
    return effect

  def b_education(self, education): 
    effect = -0.0006 if (education < 8) else -0.00025 if (education <13) else 0.00018 if (education <17) else 0.00045 if (education < 20) else 0.0009
    return effect

  def b_gender(self, gender):
    effect = 0
    biasfactor = 1 if ( self.bias["gender"]==True or self.bias["gender"] > 0) else 0 if ( self.bias["gender"]==False or self.bias["gender"] ==0 ) else -1  # for bias, no-bias, and reverse bias
    effect = -0.00045 if (gender<1) else 0.00045  # This amount to about 1% difference annually
    return biasfactor * effect 

  def b_marital(self, marital): 
    effect = 0 # let's assume martial status does not affect income growth rate 
    return effect

  def b_ethnic(self, ethnic):
    effect = 0
    biasfactor = 1 if ( self.bias["ethnic"]==True or self.bias["ethnic"] > 0) else 0 if ( self.bias["ethnic"]==False or self.bias["ethnic"] ==0 ) else -1  # for bias, no-bias, and reverse bias
    effect = -0.0006 if (ethnic < 1) else -0.00027 if (ethnic < 2) else 0.00045 
    return biasfactor * effect

  def b_industry(self, industry):
    effect = 0 if (industry < 2) else 0.00018 if (industry <4) else 0.00045 if (industry <5) else 0.00027 if (industry < 6) else 0.00045 if (industry < 7) else 0.00055
    return effect

  def b_income(self, income):
    # This is the kicker! 
    # More disposable income allow people to invest (stocks, real estate, bitcoin). Average gives them 6-10% annual return. 
    # Let us be conservative, and give them 0.6% return annually on their total income. So say roughly 0.0005 each month.
    # You can turn off this effect and compare the difference if you like. Comment in-or-out the next two lines to do that. 
    # effect = 0
    effect = 0 if (income < 50000) else 0.0001 if (income <65000) else 0.00018 if (income <90000) else 0.00035 if (income < 120000) else 0.00045 
    # Notice that this is his/her income affecting his/her future income. It's exponential in natural. 
    return effect

    # ##################################################
    # end of black box / inner structure of the model  #
    # ##################################################

  # other methods/functions
  def predictGrowthFactor( self, person ): # this is the MONTHLY growth FACTOR
    factor = 1 + self.b_0 + self.b_age( person["age"] ) + self.b_education( person['education'] ) + self.b_ethnic( person['ethnic'] ) + self.b_gender( person['gender'] ) + self.b_income( person['income'] ) + self.b_industry( person['industry'] ) + self.b_marital( ['marital'] )
    # becareful that age00 and income00 are the values of the initial record of the dataset/dataframe. 
    # After some time, these two values might have changed. We should use the current values 
    # for age and income in these calculations.
    return factor

  def predictIncome( self, person ): # perdict the new income one MONTH later. (At least on average, each month the income grows.)
    return person['income']*self.predictGrowthFactor( person )

  def predictFinalIncome( self, n, person ): 
    # predict final income after n months from the initial record.
    # the right codes should be no longer than a few lines.
    # If possible, please also consider the fact that the person is getting older by the month. 
    # The variable age value keeps changing as we progress with the future prediction.
    # return self.predictFinalIncome(n-1,person)*self.predictGrowthFactor(person) if n>0 else self.income
    final_income = person['income']
   
    for month in range(n):
        person.update({'age': person['age'] + 1/12})
        growth_factor = self.predictGrowthFactor(person)
        final_income *= growth_factor
    return final_income



print("\nReady to continue.")

#%%
# SAMPLE CODES to try out the model
utopModel = myModel( { "gender": False, "ethnic": False } ) # no bias Utopia model
biasModel = myModel( { "gender": True, "ethnic": True } ) # bias, flawed, real world model

print("\nReady to continue.")

#%%
# Now try the two models on some versions of different people. 
# See what kind of range you can get. For starters, I created a character called Plato for you as an example.
# industry: 0-leisure n hospitality, 1-retail , 2- Education 17024, 3-Health, 4-construction, 5-manufacturing, 6-professional n business, 7-finance
# gender: 0-female, 1-male
# marital: 0-never, 1-married, 2-divorced, 3-widowed
# ethnic: 0, 1, 2 
# age: 30-60, although there is no hard limit what you put in here.
# income: no real limit here.

months = 12 # Try months = 1, 12, 60, 120, 360
# In the ideal world model with no bias
plato = Person( { "age": 58, "education": 20, "gender": 1, "marital": 0, "ethnic": 2, "industry": 7, "income": 100000 } )
print(f'utop: {utopModel.predictGrowthFactor(plato)}') # This is the current growth factor for plato
print(f'utop: {utopModel.predictIncome(plato)}') # This is the income after 1 month
# Do the following line when your new function predictFinalIncome is ready
print(f'utop: {utopModel.predictFinalIncome(months,plato)}')
#
# If plato ever gets a raise, or get older, you can update the info with a dictionary:
plato.update( { "age": 59, "education": 21, "marital": 1, "income": 130000 } )

# In the flawed world model with biases on gender and ethnicity 
aristotle = Person( { "age": 58, "education": 20, "gender": 1, "marital": 0, "ethnic": 2, "industry": 7, "income": 100000 } )
print(f'bias: {biasModel.predictGrowthFactor(aristotle)}') # This is the current growth factor for aristotle
print(f'bias: {biasModel.predictIncome(aristotle)}') # This is the income after 1 month
# Do the following line when your new function predictFinalIncome is ready
print(f'bias: {biasModel.predictFinalIncome(months,aristotle)}')

hypothetical_characters = [
    {"name": "Peter", "age": 30, "education": 12, "gender": 1, "marital": 0, "ethnic": 1, "industry": 4, "income": 45000},
    {"name": "Ning", "age": 45, "education": 16, "gender": 0, "marital": 1, "ethnic": 0, "industry": 2, "income": 60000},
    {"name": "Anh", "age": 52, "education": 18, "gender": 1, "marital": 2, "ethnic": 2, "industry": 6, "income": 85000},
    {"name": "Taylor", "age": 40, "education": 20, "gender": 0, "marital": 1, "ethnic": 1, "industry": 7, "income": 95000},
]
for character_info in hypothetical_characters:
    character = Person(character_info)
    
    utop_growth_factor = utopModel.predictGrowthFactor(character)
    utop_projected_income = utopModel.predictFinalIncome(12, character)

    bias_growth_factor = biasModel.predictGrowthFactor(character)
    bias_projected_income = biasModel.predictFinalIncome(12, character)
    print(f"--- {character_info['name']} ---")
    print(f"Utopia Model - Growth factor: {utop_growth_factor}, Projected income after 12 months: {utop_projected_income}")
    print(f"Bias Model - Growth factor: {bias_growth_factor}, Projected income after 12 months: {bias_projected_income}\n")


print("\nReady to continue.")


#%% [markdown]
# # Evolution (Part III - 25%)
# 
# We want to let the 24k people in WORLD#2 to evolve, for 360 months. You can either loop them through, and 
# create a new income or incomeFinal variable in the dataframe to store the new income level after 30 years. Or if you can figure out a way to do 
# broadcasting the predict function on the entire dataframe that can work too. If you loop through them, you can also consider 
# using Person class to instantiate the person and do the calculations that way, then destroy it when done to save memory and resources. 
# If the person has life changes, it's much easier to handle it that way, then just transform the data frame directly.
# 
# We have just this one goal, to see what the world looks like after 30 years, according to the two models (utopModel and biasModel). 
# 
# Remember that in the midterm, we found that there is not much gender or ethnic bias in income distribution in world2.
# Now if we let the world to evolve under the utopia model "utopmodel", and the biased model "biasmodel", what will the income distributions 
# look like after 30 years?
# 
# Answer this in terms of distribution of income only. I don't care about 
# other utopian measures in this question here. 
# 
#%%
print(world2.columns)

#%%
def simulate_income_evolution(dataframe, months):
    
    df_evolved = dataframe.copy()
    
    df_evolved['income_evolved_utop'] = 0
    df_evolved['income_evolved_bias'] = 0

    for index, row in df_evolved.iterrows():
        person = Person(row.to_dict())
        df_evolved.at[index, 'income_evolved_utop'] = utopModel.predictFinalIncome(months, person)
        df_evolved.at[index, 'income_evolved_bias'] = biasModel.predictFinalIncome(months, person)

    return df_evolved

# Simulate the income evolution for 30 years (360 months)
world2_evolved_df = simulate_income_evolution(world2, 360)

print(world2_evolved_df[['income_evolved_utop', 'income_evolved_bias']].head())

print(world2_evolved_df[['income_evolved_utop', 'income_evolved_bias']].describe())

#%%
plt.figure(figsize=(14, 7))
plt.hist(world2_evolved_df['income_evolved_utop'], bins=50, alpha=0.5, label='Utopia Model')
plt.hist(world2_evolved_df['income_evolved_bias'], bins=50, alpha=0.5, label='Bias Model')
plt.title('Comparison of Income Distributions After 30 Years')
plt.xlabel('Income')
plt.ylabel('Frequency')
plt.legend()
plt.show()


plt.figure(figsize=(14, 7))
sns.kdeplot(world2_evolved_df['income_evolved_utop'], label='Utopia Model', bw_adjust=0.5)
sns.kdeplot(world2_evolved_df['income_evolved_bias'], label='Bias Model', bw_adjust=0.5)
plt.title('Density Plot of Income Distributions After 30 Years')
plt.xlabel('Income')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=world2_evolved_df[['income_evolved_utop', 'income_evolved_bias']], orient='h')
plt.title('Box Plot of Income Distributions After 30 Years')
plt.show()

#%% [markdown]
# HISTOGRAM

# We see that both distributions seem to have a similar shape, indicating that while the overall structure of income distribution 
# remains similar, there are subtle differences. There could be a slightly higher frequency of higher incomes in the bias model, 
# suggesting that biases may lead to higher incomes for certain segments of the population.

# DENSITY PLOT

# The density plot provides a smoother curve representing the distribution of incomes, confirming what we observed in the histogram. 
# Both distributions follow a similar pattern, but the bias model seems to allow for higher incomes, as indicated by the long tail 
# on the right. This suggests that biases could lead to greater income disparity, with a minority of the population potentially achieving 
# significantly higher incomes.

# BOX PLOT

# The box plot provides a summary of the distribution of incomes. The length of the boxes and the whiskers indicate the range and 
# spread of the data. Here, we can observe that the median income is similar for both models, but the interquartile range is slightly 
# broader for the bias model, and there are more outliers on the higher income side. This indicates that the bias model may result 
# in a wider spread of incomes, with a tendency towards higher extremes.


# These visualizations together suggest that both models allow for income growth over 30 years, but the bias model lead to a more 
# unequal society with a greater disparity in incomes.

#%% 
# # Reverse Action (Part IV - 25%)
# 
# Now let's turn our attention to World 1, which you should have found in the midterm that it is far from being fair in terms of income inequality across gender & ethnicity groups.
# Historically, federal civil rights laws such as Affirmative Action and Title IX were established to fight real-world bias that existed against females and ethnic minorities.
# Programs that provide more learning and development opportunities for underrepresented groups can help reduce bias the existed in the society.
# Although achieving an absolute utopia might be infeasible, we should all aspire to create a better society that provides equal access to education, employment and healthcare and growth
# opportunities for everyone regardless of race, ethnicity, age, gender, religion, sexual orientation, and other diverse backgrounds. 

# %%
# Let us now put in place some policy action to reverse course, and create a reverse bias model:
revbiasModel = myModel( { "gender": -1, "ethnic": -1 } ) # revsered bias, to right what is wronged gradually.

# If we start off with World 1 on this revbiasModel, is there a chance for the world to eventual become fair like World #2?
# If so, how long does it take, to be fair for the different genders? How long for the different ethnic groups? 

# If the current model cannot get the job done, feel free to tweak the model with more aggressive intervention to change the growth rate percentages on gender and ethnicity to make it work. 

#%%

# newWorld2 = # world2.
# def simulate(n, row):
#   newPerson = Person(row)
#   revbiasModel.predictFinalIncome(n, newPerson) # newPerson.income
#   return newPerson.income
  

# newlist = world2.apply( simulate() )

# world2.income = newlist 

#%%
def simulate_reversed_bias(n_months, df, model):
    df_evolved = df.copy()
    df_evolved['income_evolved_revbias'] = 0
    
    for index, row in df_evolved.iterrows():
        person = Person(row.to_dict())
        df_evolved.at[index, 'income_evolved_revbias'] = model.predictFinalIncome(n_months, person)
        
    return df_evolved

def compare_income_distributions(df1, df2, model, time_periods):
    simulation_results = {}

    for months in time_periods:
        df1_evolved = simulate_reversed_bias(months, df1, model)
        
        summary_stats = df1_evolved['income_evolved_revbias'].describe()
        
        summary_stats_world2 = df2['income00'].describe()
        comparison_stats = summary_stats - summary_stats_world2
        
        simulation_results[months] = comparison_stats

    results_df = pd.DataFrame(simulation_results)
    
    return results_df

time_periods = [60, 120, 180, 240, 300, 360]  # up to 30 years

comparison_results = compare_income_distributions(world1, world2, revbiasModel, time_periods)

print(comparison_results)

#%% [markdown]
# - Mean Income Difference: Increases over time, indicating that the reversed bias interventions are gradually leading to 
# higher average incomes in World 1.
# - Standard Deviation Difference: Also increases, suggesting that there's a greater spread of incomes in World 1 as the simulation 
# progresses. This might imply more variability and possibly greater opportunity for upward economic mobility.
# - Minimum Income Difference: Grows over time, which could mean that the lowest incomes in World 1 are catching up or surpassing 
# those in World 2.
# - Quartiles and Maximum Income Difference: The increases here suggest that not only the average but also the range of incomes in World 1 
# is expanding, which could be interpreted as an improvement in economic outcomes across different segments of the population.

# So yes, World1 can be fair like World2, and through the time World1 will catch up

# %%
def simulate_reversed_bias_gender_ethnic(n_months, df, model, gender_ethnic_groups):
    results = {}
    
    for (gender, ethnic) in gender_ethnic_groups:
        df_group = df[(df['gender'] == gender) & (df['ethnic'] == ethnic)]
        df_group_evolved = simulate_reversed_bias(n_months, df_group, model)
        
        mean_income_world1 = df_group_evolved['income_evolved_revbias'].mean()
    
        mean_income_world2 = world2.groupby(['gender', 'ethnic'])['income00'].mean()
        
        # Store the differences in mean income for this gender and ethnic group
        results[(gender, ethnic)] = mean_income_world1 - mean_income_world2
    
    return results

# Define the different gender and ethnic groups 
gender_ethnic_groups = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

time_periods = [60, 120, 180, 240, 300, 360]  # Example time points
fairness_over_time = {}

for months in time_periods:
    fairness_over_time[months] = simulate_reversed_bias_gender_ethnic(months, world1, revbiasModel, gender_ethnic_groups)

# %%
for months, differences in fairness_over_time.items():
    print(f"Time: {months} months")
    for group, difference in differences.items():
        gender, ethnic = group
        print(f"Gender: {gender}, Ethnic: {ethnic}, Mean Income Difference: {difference}")
    print("\n")
# %%
print(fairness_over_time)



# %%
