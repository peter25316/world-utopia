#!/usr/bin/env python
# coding: utf-8

# In[1]:


# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%%[markdown]'
# First, install the rfit package using pip install rfit
# This package (rfit - regression.fit) is for use at The George Washington University Data Science program. 
# There are some useful and convenient functions we use in our python classes.
# One of them is dfapi, which connects to the api.regression.fit endpoint, to load data frames used in our classes.
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rfit 
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency

# world1 = rfit.dfapi('World1', 'id')
# world1.to_csv("world1.csv")
world1_df = pd.read_csv("world1.csv", index_col="id") # use this instead of hitting the server if csv is on local
# world2 = rfit.dfapi('World2', 'id')
# world2.to_csv("world2.csv")
world2_df = pd.read_csv("world2.csv", index_col="id") # use this instead of hitting the server if csv is on local

print("\nReady to continue.")

#%%[markdown]
# # Two Worlds 
# 
# I was searching for utopia, and came to this conclusion: If you want to do it right, do it yourself. 
# So I created two worlds. 
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
# 
# Please do whatever analysis you need, convince your audience both, one, or none of these 
# worlds is fair, or close to a utopia. 
# Use plots, maybe pivot tables, and statistical tests (optional), whatever you deem appropriate 
# and convincing, to draw your conclusions. 
# 
# There are no must-dos (except plots), should-dos, cannot-dos. The more convenicing your analysis, 
# the higher the grade. It's an art.
#


# In[2]:


world1_df


# In[3]:


#%%
world1_desc = world1_df.describe()
world2_desc = world2_df.describe()
world1_desc


# In[4]:


world2_desc


# In[5]:


#%%
def plot_histograms(df, title):
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(['age00', 'education', 'income00'], start=1):
        plt.subplot(2, 3, i)
        sns.histplot(df[column], kde=True)
        plt.title(f'{title} - {column}')
    plt.tight_layout()
    plt.show()
plot_histograms(world1_df, "World1")
plot_histograms(world2_df, "World2")


# In[6]:


#%%[markdown]
# Descriptive Statistics
#
# World1:
# Age: Ranges from 30 to 60 years, with an average of 44.29 years.
# Education: On average, individuals have 15.07 years of education.
# Income: Average income is about 60,642, with a wide distribution.
#
# World2:
# Age: Similar to World1, ranging from 30 to 60 years, with an average of 44.31 years.
# Education: Slightly higher average education at 15.11 years.
# Income: Very similar to World1 with an average income of around 60,635.
#
# Observations from Histograms
# Both worlds show similar distributions for age, education, and income.
# The distribution of income in both worlds seems to be right-skewed, indicating that a larger proportion of the population earns a lower income, with fewer individuals earning higher incomes.


# In[7]:


#%%
#%%[markdown]
# # Correlation Analysis

corr_world1 = world1_df.corr()
corr_world2 = world2_df.corr()


def plot_correlation_heatmap(df, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(df, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f'Correlation Matrix for {title}')
    plt.show()

plot_correlation_heatmap(corr_world1, "World1")


# In[8]:


#%%[markdown]
# The correlation coefficient ranges from -1 to 1, where:

# 1 indicates a perfect positive linear relationship.
# 0 indicates no linear relationship.
# -1 indicates a perfect negative linear relationship.
# Diagonal (from top left to bottom right):
# The diagonal is filled with 1s because every variable has a perfect positive correlation with itself.
#
# Colors:
# The color scale on the right side indicates the strength of the correlation, with red representing stronger positive correlations and blue representing stronger negative correlations. Lighter shades closer to white indicate weaker correlations.
#
# # World1 
#
# Age (age00):
# The correlations between age and all other variables are close to 0, suggesting no linear relationship.
#
# Education:
# Similarly, education appears to have very little to no linear relationship with other variables, as indicated by the values close to 0.
#
# Marital Status (marital):
# There's no indication of a strong linear relationship with other variables since the coefficients are around 0.
#
# Gender:
# Gender has a weak to moderate positive correlation with industry (0.22) and income (0.19), suggesting that as one goes from female to male (coded 0 to 1), there tends to be a slight increase in both industry and income.
#
# Ethnicity (ethnic):
# Ethnicity shows weak positive correlations with industry (0.23) and income (0.22). This suggests that different ethnic groups may be represented differently across industries and income levels, but the relationship is not strong.
#
# Industry:
# Industry has a strong positive correlation with income (0.89), suggesting that the type of industry someone is employed in is strongly associated with their income level.
#
#There are noticeable correlations between income and education, and between income and industry. This suggests that higher education and certain industries are associated with higher income levels. Gender and marital status show weaker correlations with income.


# In[9]:


plot_correlation_heatmap(corr_world2, "World2")


# In[10]:


#%%[markdown]
# # World2
#
# Age (age00):
# The correlations between age and other variables are negligible (around 0), suggesting no clear linear relationship with age.
#
# Education:
# Similar to age, education does not show a significant correlation with the other variables, with coefficients around 0.
#
# Marital Status (marital):
# Marital status shows a small negative correlation with gender (-0.13), which might suggest that one gender could be slightly less likely to be married than the other in this world.
#
# Gender:
# Gender has a negative correlation with marital status, as noted above.
# There is almost no correlation with other variables, except for a very small positive correlation with income (0.01), which is negligible.
#
# Ethnicity (ethnic):
# Ethnicity does not exhibit any strong correlation with the other variables in the dataset, with all values being close to 0.
#
# Industry:
# Similar to World1, there is a very strong positive correlation between industry and income (0.89), indicating that the type of industry has a significant impact on the income level, just as in World1.
#
#Similar to World1, education and industry show a significant correlation with income. The patterns of correlation in both worlds are quite alike, suggesting similar dynamics in terms of how education, industry, and other factors relate to income.


# In[11]:


#%%
#%%[markdown]
# # Income Distribution
def plot_income_boxplots(df, title):
    plt.figure(figsize=(18, 12))
    categorical_vars = ['gender', 'ethnic', 'marital', 'industry', 'education']
    
    for i, var in enumerate(categorical_vars, start=1):
        plt.subplot(2, 3, i)
        sns.boxplot(x=var, y='income00', data=df)
        plt.title(f'{title} - Income by {var.title()}')

    plt.tight_layout()
    plt.show()

plot_income_boxplots(world1_df, "World1")


# In[12]:


#%%[markdown]
# Bos Plot Explanation:
# - The box in the box plot shows us where the "middle-class" incomes fall — the range between the lower-middle and upper-middle earners.
# - The line in the middle of the box marks the "median income," which is the income smack in the middle of the group — half the people earn more than this amount, and half earn less.
# - The whiskers are like the income limits for the majority of people. They stretch out to show the income of people who earn a bit less or a bit more than the middle group, but not the super-low or super-high earners.
# - The dots or points beyond the whiskers are the incomes that are unusually low or high — I think of them as the guys who are earning way less or way more than most people, like someone making minimum wage compared to a CEO's income.
# 
# # World1:
#
# Income by Gender:
# - There are two categories (0 for female and 1 for male).
# - The median income for males is slightly higher than for females, as indicated by the line within the box.
# - The distribution for males has more outliers on the higher income side, suggesting more males have higher incomes than the general population.
# 
# Income by Ethnic:
# - There are three ethnic groups (0, 1, 2).
# - The medians are relatively similar across all groups, but group 2 has a slightly higher median income.
# - Group 2 also shows more high-income outliers, implying that individuals in this group tend to have higher incomes compared to other ethnic groups.
# 
# Income by Marital:
# - Four marital statuses are represented (0 for never married, 1 for married, 2 for divorced, 3 for widowed).
# - The median income seems to be higher for married individuals, with the box plot for marital status 1 being slightly higher.
# - The spread of income is similar across the groups, but with different numbers of high-income outliers.
# 
# Income by Industry:
# - There are eight industries, coded from 0 to 7.
# - There's a general upward trend in median income as the industry code increases, indicating that certain industries tend to pay more on average.
# - Industry 7 has a significantly higher median income, consistent with the strong correlation between industry and income seen in the correlation matrix.
# 
# Income by Education:
# - Education levels range from 0 to 22 years.
# - There's a trend where higher education levels tend to have higher median incomes, especially noticeable after 16 years of education.
# - The spread of incomes increases with education, and the number of high-income outliers is more prominent at higher education levels.


# In[13]:


plot_income_boxplots(world2_df, "World2")


# In[14]:


#%%[markdown]
# # World2:
#
# Income by Gender:
# - There are two boxes representing two groups: let's say, group 0 might be women and group 1 might be men.
# - Both groups have a wide range of incomes, but men have more "very high earners" (shown by the dots at the top).
#
# Income by Ethnic:
# - Three boxes for three different ethnic groups.
# - All groups have a similar range of incomes from low to high, but group 2 has a few more "very high earners."
#
# Income by Marital:
# - Four boxes showing four marital statuses, maybe from single to widowed.
# - They all have similar income ranges, but married people (group 1) tend to have more "high earners."
#
# Income by Industry:
# - Each box is a different type of job industry.
# - Some industries, like the last one (number 7), pay a lot more on average.
#
# Income by Education:
# - The more education people have, generally the more they earn, but there's a big mix.
# - After a certain point (around 16 years of education), people start to earn significantly more, and there are many "very high earners."
#
#
# # On both worlds:
# Gender: In both worlds, there seems to be a disparity in income distribution between genders, with males generally earning more.
# Ethnicity: There are variations in income across different ethnic groups, indicating potential disparities.
# Marital Status: Income distribution varies with marital status. Married individuals tend to have higher incomes.
# Industry: A clear gradient is observable in income across industries, with certain industries (like finance and professional & business) having higher median incomes.
# Education: Higher education levels are associated with higher income, which aligns with the correlations observed.


# In[15]:


#%%
#%%[markdown]
# # T-test
# For World1, comparing the average incomes between genders
ttest_result_world1 = ttest_ind(
    world1_df[world1_df['gender'] == 0]['income00'], 
    world1_df[world1_df['gender'] == 1]['income00']
)

# For World2, doing the same comparison
ttest_result_world2 = ttest_ind(
    world2_df[world2_df['gender'] == 0]['income00'], 
    world2_df[world2_df['gender'] == 1]['income00']
)


# In[16]:


ttest_result_world1


# In[17]:


#%%[markdown]
# # World1:
#
# The t-statistic is a big negative number, which means one group consistently has lower incomes than the other group in this world.
#
# The p-value is a really tiny number, way smaller than the standard cutoff we usually use to decide if something is probably not due to random chance. This tells us that the difference in incomes between the two groups isn't just a fluke; it's something we can be pretty sure about.


# In[18]:


ttest_result_world2


# In[19]:


#%%[markdown]
# # World2:
#
# The t-statistic is a small negative number. This suggests that there might be a slight difference in average income between the two groups (likely men and women), with the first group possibly earning a bit less than the second.
#
# The p-value is about 0.397, which is not small enough to be considered significant by typical standards (usually it should be less than 0.05 to be noteworthy).
#
# The incomes between the two groups in World2 might not be equal, but the difference isn't strong enough.
# So, in this world, I can't confidently say that men or women earn more. The difference between their earnings isn't clear cut.


# In[21]:


#%%
#%%[markdown]
# # Chi-Square-Test of each categories with each other
def perform_chi_square_tests(df):
    results = {}
    columns = ['gender', 'marital', 'ethnic', 'industry']  # Add other categorical columns if needed
    
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            contingency_table = pd.crosstab(df[columns[i]], df[columns[j]])
            chi2_result = chi2_contingency(contingency_table)
            results[f'{columns[i]} & {columns[j]}'] = chi2_result
    
    return results

chi_square_tests_world1 = perform_chi_square_tests(world1_df)
chi_square_tests_world2 = perform_chi_square_tests(world2_df)


# In[22]:


print("World1 Chi-square Tests:")
for test, result in chi_square_tests_world1.items():
    print(test, result)


# In[ ]:


#%%[markdown]
# # World1:
#
# Gender & Marital Status:
# - Result: Not significant (p-value = 0.584)
# - Meaning: There's no strong evidence to suggest that gender and marital status are related. In other words, whether someone is married, single, etc., doesn't seem to depend on their gender.
#
# Gender & Ethnic:
# - Result: Barely significant (p-value = 0.043)
# - Meaning: There's a slight indication that gender and ethnicity might be related. This could mean that the proportion of males and females might differ slightly across different ethnic groups, but this association is not very strong.
#
# Gender & Industry:
# - Result: Highly significant (p-value = 0.0)
# - Meaning: There's a very strong relationship between gender and industry. This suggests that certain industries might be predominantly male or female.
#
# Marital & Ethnic:
# - Result: Not significant (p-value = 0.367)
# - Meaning: Marital status and ethnicity don't seem to be related. This means that the distribution of marital status is likely similar across different ethnic groups.
#
# Marital & Industry:
# - Result: Not significant (p-value = 0.883)
# - Meaning: There's no significant relationship between what industry people work in and their marital status.
#
# Ethnic & Industry:
# - Result: Highly significant (p-value = 0.0)
# - Meaning: There's a significant relationship between ethnicity and industry. Certain ethnic groups might be more prevalent in certain industries.
#
# In World1, the most notable associations are between gender and industry, and ethnicity and industry, suggesting that these aspects are not evenly distributed across these categories. Meanwhile, marital status doesn't show a significant association with either gender or industry, indicating more uniformity in these aspects.


# In[24]:


print("\nWorld2 Chi-square Tests:")
for test, result in chi_square_tests_world2.items():
    print(test, result)


# In[ ]:


#%%[markdown]
# # World2:
#
# Gender & Marital Status:
# - Result: Highly significant (p-value ≈ 0).
# - Meaning: There's a very strong relationship between gender and marital status. This could mean that the marital status distribution is quite different for men and women in this world.
#
# Gender & Ethnic:
# - Result: Not significant (p-value = 0.441).
# - Meaning: There doesn't seem to be a notable relationship between gender and ethnicity. The mix of men and women is likely similar across different ethnic groups.
#
# Gender & Industry:
# - Result: Slightly significant (p-value = 0.011).
# - Meaning: There's a hint of a relationship between gender and industry, but it's not very strong. It suggests there might be some industries with slightly more men or women, but the difference isn't dramatic.
#
# Marital & Ethnic:
# - Result: Not significant (p-value = 0.305).
# - Meaning: Marital status doesn't seem to be related to ethnicity. The distribution of marital statuses (like being single, married, etc.) is probably similar among different ethnic groups.
#
# Marital & Industry:
# - Result: Slightly significant (p-value = 0.017).
# - Meaning: There might be a small connection between what industry people work in and their marital status, but it's not a very strong link.
#
# Ethnic & Industry:
# - Result: Not significant (p-value = 0.736).
# - Meaning: Ethnicity and industry don't seem to be related. People from different ethnic backgrounds likely work across various industries at similar rates.
#
# In World2, the strongest association is seen between gender and marital status, while other variable pairs like gender and ethnic, and ethnic and industry show no significant association. The slight associations observed in gender and industry, and marital and industry, indicate some differences in these categories, but they're not as pronounced.
#
# World1:
# Significant Gender and Industry Bias: The strong association between gender and industry suggests a gender-based division in the workforce, which could indicate a lack of equal opportunity or stereotypical job roles based on gender.
# Ethnic Groups Concentrated in Certain Industries: This might point to a lack of diversity in certain sectors or potential barriers for certain ethnic groups in accessing a wide range of industries.
# Lack of Association in Other Areas: The absence of strong associations between gender, marital status, and ethnicity suggests some level of equality in these areas.
# 
# World2:
# Strong Association Between Gender and Marital Status: This could imply traditional or stereotypical roles based on gender, which might not align with the ideals of a utopian society.
# Slight Bias in Industry Based on Gender: While not as pronounced as in World1, this still indicates some level of inequality in employment opportunities.
# More Equality in Other Aspects: The lack of strong associations in other areas could be indicative of fairness in those dimensions.


# In[25]:


#%%
#%%[markdown]
# # Chi-Square-Test of each categories with income
def categorize_income(df):
    labels = ['low', 'medium', 'high']
    df['income_category'] = pd.qcut(df['income00'], q=3, labels=labels)
    return df

world1_df = categorize_income(world1_df)
world2_df = categorize_income(world2_df)

def perform_chi_square_tests_income_category(df):
    results = {}
    columns = ['gender', 'marital', 'ethnic', 'industry']  # Add other categorical columns if needed
    
    for col in columns:
        contingency_table = pd.crosstab(df[col], df['income_category'])
        chi2_result = chi2_contingency(contingency_table)
        results[col] = chi2_result
    
    return results

chi_square_tests_income_category_world1 = perform_chi_square_tests_income_category(world1_df)
chi_square_tests_income_category_world2 = perform_chi_square_tests_income_category(world2_df)


# In[26]:


print("World1 Chi-square Tests for Income Category:")
for category, result in chi_square_tests_income_category_world1.items():
    print(category, result)


# In[ ]:


#%%[markdown]
# # World1:
#
# Gender:
# - Result: Highly significant (p-value ≈ 0).
# - Meaning: There's a very strong connection between gender and income level. This suggests that in World1, men and women are likely to be in different income categories, indicating a gender income gap.
#
# Marital Status:
# - Result: Not significant (p-value = 0.689).
# - Meaning: Marital status doesn't seem to have a strong connection with income levels. This means whether someone is single, married, divorced, etc., doesn't significantly affect which income category they are in.
#
# Ethnic Group:
# - Result: Highly significant (p-value = 0).
# - Meaning: There's a significant relationship between a person's ethnic group and their income level. This could suggest that certain ethnic groups are more likely to be in higher or lower income categories.
#
# Industry:
# - Result: Extremely significant (p-value = 0).
# - Meaning: The type of industry a person works in is strongly linked to their income level. This means that some industries in World1 might pay a lot more or a lot less, leading to clear income differences among them.


# In[28]:


print("\nWorld2 Chi-square Tests for Income Category:")
for category, result in chi_square_tests_income_category_world2.items():
    print(category, result)


# In[ ]:


#%%[markdown]
# # World2:
#
# Gender:
# - Result: Not significant (p-value = 0.206).
# - Meaning: Gender doesn't seem to play a significant role in determining income levels in World2. In other words, men and women are likely to be found in similar proportions across low, medium, and high-income categories.
#
# Marital Status:
# - Result: Not significant (p-value = 0.201).
# - Meaning: Marital status (like being single, married, etc.) isn't strongly related to what income category people fall into. So, someone's income level isn't significantly affected by their marital status.
#
# Ethnic Group:
# - Result: Not significant (p-value = 0.414).
# - Meaning: Ethnicity doesn't have a significant influence on income levels. Different ethnic groups appear to be equally distributed across the various income categories.
#
# Industry:
# - Result: Extremely significant (p-value = 0.0).
# - Meaning: The industry a person works in is highly indicative of their income level. This suggests that some industries pay significantly more or less than others, leading to clear distinctions in income based on industry.
#
# World1 shows significant differences in income based on gender and ethnicity, suggesting inequalities in these aspects. This might challenge the notion of World1 being a utopia.
# World2 is more equal in terms of gender and ethnicity regarding income, but industry still plays a major role in determining how much people earn.
# In both worlds, the type of industry is crucial for income levels. However, World2 appears to be more balanced in terms of gender and ethnic equality.


# In[ ]:


#%%[markdown]
# # Final Conclusion:
# World1:
# - Significant Gender and Income Gap: Men and women have a noticeable difference in income.
# - Ethnic Disparity in Income: Different ethnic groups earn significantly different incomes.
# - Industry-Based Income Differences: What job you do greatly affects your income.
# - Conclusion: World1 shows clear inequalities in terms of gender and ethnicity, which are key markers against a utopian society. It’s not very close to the ideal of a utopia.
#
# World2:
# - More Gender Equality: Men and women have more similar incomes.
# - Ethnic Equality in Income: Income levels are similar across different ethnic groups.
# - Still Industry-Based Income Differences: Like World1, your job type greatly impacts your income.
# - Conclusion: World2 is closer to being a utopia than World1, especially in terms of gender and ethnic equality. However, the significant income disparity based on industry keeps it from being a perfect utopia.
#
# Neither World1 nor World2 is a perfect utopia.
# While World2 shows more signs of equality and fairness, particularly in gender and ethnic terms, the continued significant disparity in income based on industry suggests that it's not completely equitable.
# World1 has more pronounced disparities in gender and ethnicity, making it further from the utopian ideal.
#
# A utopian society would ideally have no significant differences in income based on gender, ethnicity, or industry. World2 does better in some aspects but still falls short in others, and World1 shows more obvious inequalities. So, while neither world is a perfect utopia, World2 is somewhat closer to this ideal.

