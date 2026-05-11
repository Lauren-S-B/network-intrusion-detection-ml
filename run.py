#!/usr/bin/env python
# coding: utf-8

# In[169]:


#For Preprocessing & ML
import pandas as pd
import numpy as np
import csv
# For Missing Values
import seaborn as sns
#For Data Types
#Irrelevant Attributes
from sklearn.feature_selection import VarianceThreshold
#Scaling
from sklearn.preprocessing import StandardScaler
#Feature Selection
#from mlxtend.feature_selection import SequentialFeatureSelector as SFS
#from sklearn.neighbors import KNeighborsClassifier
#Feature Engineering
import matplotlib.pyplot as plt
#Train Test Split
from sklearn.model_selection import train_test_split
#Undersampling
from imblearn.under_sampling import RandomUnderSampler
#Oversampling
from imblearn.over_sampling import RandomOverSampler
#Decision Tree Model
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
#K Nearest Neighbours
from sklearn.neighbors import KNeighborsClassifier


# In[170]:


test_1_df = pd.read_excel('Test-data-2025-1.xlsx')
test_2_df = pd.read_excel('Test-data-2025-2.xlsx')
train_df = pd.read_excel('Training-data-2025.xlsx')


# In[171]:


test_1_df.head()


# In[172]:


test_1_df.shape


# In[173]:


test_2_df.head()


# In[174]:


test_2_df.shape


# In[175]:


train_df.head()


# In[176]:


train_df.shape


# ## Handling Duplicates

# #### Training Data

# In[177]:


duplicates_train_df = train_df.duplicated(keep=False)
print("\nDuplicate rows based on all columns: ", duplicates_train_df)


# In[178]:


num_duplicates = train_df.duplicated(keep=False).sum()
print("Number of duplicate rows : ", num_duplicates)


# In[179]:


dropped_duplicates_train_df = train_df.drop_duplicates(keep='first')
print("\nDataframe after removing all duplicates except the first occurence of the duplicate: \n", dropped_duplicates_train_df)


# In[180]:


#Checking for duplicates after keep the first occurance and removing the rest
num_duplicates1 = dropped_duplicates_train_df.duplicated(keep=False).sum()
print("Number of duplicate rows after removing all duplicates after the \nfirst occurence: ", num_duplicates1)


# #### Test Data 1

# In[181]:


duplicates_test_1_df = test_1_df.duplicated(keep=False)
print("\nDuplicate rows based on all columns: ", duplicates_test_1_df)


# In[182]:


num_duplicates_1 = test_1_df.duplicated(keep=False).sum()
print("Number of duplicate rows : ", num_duplicates_1)


# In[183]:


dropped_duplicates_test_1_df = test_1_df.drop_duplicates(keep='first')
print("\nDataframe after removing all duplicates except the first occurence of the duplicate: \n", dropped_duplicates_test_1_df)


# In[184]:


#Checking for duplicates after keep the first occurance and removing the rest
num_duplicates_test_1 = dropped_duplicates_test_1_df.duplicated(keep=False).sum()
print("Number of duplicate rows after removing all duplicates after the \nfirst occurence: ", num_duplicates_test_1)


# #### Test Data 2 

# In[185]:


duplicates_test_2_df = test_1_df.duplicated(keep=False)
print("\nDuplicate rows based on all columns: ", duplicates_test_2_df)


# In[186]:


num_duplicates_2 = test_2_df.duplicated(keep=False).sum()
print("Number of duplicate rows : ", num_duplicates_2)


# In[187]:


dropped_duplicates_test_2_df = test_2_df.drop_duplicates(keep='first')
print("\nDataframe after removing all duplicates except the first occurence of the duplicate: \n", dropped_duplicates_test_2_df)


# In[188]:


#Checking for duplicates after keep the first occurance and removing the rest
num_duplicates_test_2 = dropped_duplicates_test_2_df.duplicated(keep=False).sum()
print("Number of duplicate rows after removing all duplicates after the \nfirst occurence: ", num_duplicates_test_2)


# ## Data Types

# #### Training Data

# In[189]:


print(dropped_duplicates_train_df.dtypes)


# In[196]:


#Convert to string & clean whitespace/casing
dropped_duplicates_train_df['service'] = dropped_duplicates_train_df['service'].astype(str).str.strip().str.lower()
#Replace known missing or rare indicators
dropped_duplicates_train_df['service'] = dropped_duplicates_train_df['service'].replace({'-': 'rare_service'})#, '': 'unknown', 'nan': 'unknown'
#dropped_duplicates_train_df['service'] = dropped_duplicates_train_df['service'].fillna('unknown')
freq = dropped_duplicates_train_df['service'].value_counts(normalize=True)
dropped_duplicates_train_df['service'] = dropped_duplicates_train_df['service'].map(freq)
dropped_duplicates_train_df.head()


# In[198]:


#Identify all categorical columns
cat_cols = dropped_duplicates_train_df.select_dtypes(include=['object']).columns.tolist()
#Clean & temporarily fill missing to compute frequencies
for col in cat_cols:
    dropped_duplicates_train_df[col] = dropped_duplicates_train_df[col].astype(str).str.lower().str.strip()
#Apply frequency encoding for all categorical columns
for col in cat_cols:
    freq = dropped_duplicates_train_df[col].value_counts(normalize=True) #normalise for proportions
    dropped_duplicates_train_df[col] = dropped_duplicates_train_df[col].map(freq)
print(dropped_duplicates_train_df)


# #### Test Data 1

# In[165]:


print(dropped_duplicates_test_1_df.dtypes)


# In[168]:


print(dropped_duplicates_test_2_df.dtypes)


# ## Missing Values

# In[195]:


sns.heatmap(dropped_duplicates_train_df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[164]:


dropped_duplicates_train_df.isnull().sum()


# In[166]:


sns.heatmap(dropped_duplicates_test_1_df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[167]:


sns.heatmap(dropped_duplicates_test_2_df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# ## Irrelevant attributes

# #### Training Data

# In[273]:


selector = VarianceThreshold(threshold=0.05)
# Fit selector to data and transforms it
# Return new array with selected features
df_selected_array = selector.fit_transform(dropped_duplicates_train_df)
# Get the names of the selected features
selected_features = dropped_duplicates_train_df.columns[selector.get_support()]
# Create a new DataFrame with only the selected features
train_df_selected = pd.DataFrame(df_selected_array, columns=selected_features)
print("\nDataFrame after VarianceThreshold:")
train_df_selected.head()


# In[274]:


#Identifying irrelevant attribute removed using Variance Threshold method
feature_mask = selector.get_support()
original_columns = train_df.columns
removed_columns = original_columns[~feature_mask]
print("Columns removed by VarianceThreshold:", removed_columns)


# In[275]:


#Checking whether the irrelevant attributes mentioned above were removed
cols_df2 = train_df_selected.columns
print("Columns after removing irrelevant attributes: ", cols_df2)


# #### Test Data 1

# In[ ]:





# #### Test Data 2 

# In[ ]:





# ## Visualisation

# In[216]:


sns.pairplot(train_df_selected)


# In[217]:


train_df_selected.plot()


# ## Scaling

# In[276]:


scaler = StandardScaler()
scaled_data = scaler.fit_transform(train_df_selected)
print(train_df_selected)


# In[277]:


train_df_selected.head()


# ## Data Classification

# In[281]:


X = train_df_selected.drop(['label'], axis = 1) #all x variables (INPUT).
y = train_df_selected['label'] #only label is y variable (OUTPUT).


# In[282]:


X_test_1 = dropped_duplicates_test_1_df.drop('label', axis=1)
y_test_1 = dropped_duplicates_test_1_df['label']


# In[283]:


X_test_2 = dropped_duplicates_test_2_df.drop('label', axis=1)
y_test_2 = dropped_duplicates_test_2_df['label']


# #### Class distribution

# In[284]:


label_count = y.value_counts()
label_count


# In[285]:


y.value_counts().plot.pie(autopct='%.2f')


# In[286]:


label_count.index


# In[287]:


df_encoded = pd.get_dummies(train_df_selected, columns=['proto'])
# Split into features and labels
X = df_encoded.drop('label', axis=1)
y = df_encoded['label']


# #### Data Splitting

# In[288]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[289]:


X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[290]:


y_train.value_counts().plot.pie(autopct='%.2f')


# In[291]:


y_train.value_counts()


# ### Random Undersampling 

# In[292]:


rus = RandomUnderSampler(sampling_strategy=1) # Numerical value
# rus = RandomUnderSampler(sampling_strategy="not minority") # String
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
ax = y_train_rus.value_counts().plot.pie(autopct='%.2f')
_ = ax.set_title("Under-sampling")


# In[293]:


# Class distribution
y_train_rus.value_counts()


# #### Random Oversampling 

# In[294]:


ros = RandomOverSampler(sampling_strategy="not majority") # String
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)

ax = y_train_ros.value_counts().plot.pie(autopct='%.2f')
_ = ax.set_title("Over-sampling")


# In[295]:


y_train_ros.value_counts()


# # Model Building without Class Balancing

# ### Decision Tree

# In[300]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train) #input data for imbalanced data sets

# Training Cross-validation Models
from sklearn.metrics import make_scorer, recall_score, matthews_corrcoef
from sklearn.model_selection import cross_validate

model_cv = DecisionTreeClassifier(random_state=42) #METHODOLOGY : instantiating decision tree classifier
cv_scoring = {'MCC': make_scorer(matthews_corrcoef)} #specifying custom scoring function
    #use make score function and as input argument use mathews correlation coefficient or mcc
    #the reason for using the mcc is because it is a great metric for imbalanced data sets
    #because it considers all of the true positive, true negative, false negative, false positive
    #so the 4 tp, tn, fp, fn whereas other equations from accuracy or specificity are looking at the 
    #positive or negative classes or the number of correct predictions versus the total however,
    #Mathews correlation coefficient considers all of the terms in the same equation and even if
    #there is imbalanced data set, this metric is robust.
cv = cross_validate(model_cv, X_train, y_train, cv=5, scoring=cv_scoring) #here we are specifying the cv scoring to
    #be scoring so are specifying the cv scoring variable for the option scoring and we are going to 
    #"cv=5" perform 5 fold cross validation 
    #For cross validation function already specify mcc using custom function so can calculate average from
    #cross validation results

# Apply model to make prediction
from sklearn.metrics import matthews_corrcoef

y_train_pred = model.predict(X_train) #making predictions -> put in x and get y
y_test_pred = model.predict(X_test) # put in x for test set and get y
  
mcc_train = matthews_corrcoef(y_train, y_train_pred) #calculating mcc
mcc_test = matthews_corrcoef(y_test, y_test_pred)
mcc_cv = cv['test_MCC'].mean() # bc have 5 fold and list of 5 values so wil calculate mean of that

# Display model performance results -> creates summary data frame
train_df1_labels = pd.Series(['MCC_train', 'MCC_CV', 'MCC_test'], name = 'Performance_metric_names')
train_df1_values = pd.Series([mcc_train, mcc_cv, mcc_test], name = 'Performance_metric_values')
train_df2 = pd.concat([train_df1_labels, train_df1_values], axis=1)
train_df2


# ### Naive Bayes 

# In[297]:


model = GaussianNB()
model.fit(X_train, y_train)

from sklearn.metrics import make_scorer, recall_score, matthews_corrcoef
from sklearn.model_selection import cross_validate

model_cv = GaussianNB()
cv_scoring = {'MCC': make_scorer(matthews_corrcoef)}
cv = cross_validate(model_cv, X_train, y_train, cv = 5, scoring=cv_scoring)

from sklearn.metrics import matthews_corrcoef

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

mcc_train = matthews_corrcoef(y_train, y_train_pred)
mcc_test = matthews_corrcoef(y_test, y_test_pred)
mcc_cv = cv['test_MCC'].mean()

train_df1_labels = pd.Series(['MCC_train','MCC_CV','MCC_test'], name = 'Performance_metric_names')
train_df1_values = pd.Series([mcc_train, mcc_cv, mcc_test], name = 'Performance_metric_values')
train_df6 = pd.concat([train_df1_labels, train_df1_values], axis=1)
train_df6


# ### KNN

# In[298]:


model = KNeighborsClassifier(n_neighbors=2) #1
model.fit(X_train, y_train)

# Training Cross-validation Models
from sklearn.metrics import make_scorer, recall_score, matthews_corrcoef
from sklearn.model_selection import cross_validate

model_cv = KNeighborsClassifier(n_neighbors=2) #1
cv_scoring = {'MCC': make_scorer(matthews_corrcoef)}
cv = cross_validate(model_cv, X_train, y_train, cv=5, scoring=cv_scoring)

# Apply model to make prediction
from sklearn.metrics import matthews_corrcoef

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

mcc_train = matthews_corrcoef(y_train, y_train_pred)
mcc_test = matthews_corrcoef(y_test, y_test_pred)
mcc_cv = cv['test_MCC'].mean()

# Display model performance results
train_df1_labels = pd.Series(['MCC_train', 'MCC_CV', 'MCC_test'], name = 'Performance_metric_names')
train_df1_values = pd.Series([mcc_train, mcc_cv, mcc_test], name = 'Performance_metric_values')
train_df10 = pd.concat([train_df1_labels, train_df1_values], axis=1)
train_df10


# # Model Building with Undersampled Balanced Class

# ### Decision Tree

# In[301]:


model = DecisionTreeClassifier(random_state=30) #random_state = 42
model.fit(X_train_rus, y_train_rus) #balanced data set from undersampling

# Training Cross-validation Models
from sklearn.metrics import make_scorer, recall_score, matthews_corrcoef
from sklearn.model_selection import cross_validate

model_cv = DecisionTreeClassifier(random_state=30) #random_state = 42
cv_scoring = {'MCC': make_scorer(matthews_corrcoef)}
cv = cross_validate(model_cv, X_train_rus, y_train_rus, cv=5, scoring=cv_scoring) #cv = 5

# Apply model to make prediction
from sklearn.metrics import matthews_corrcoef

y_train_pred = model.predict(X_train_rus)
y_test_pred = model.predict(X_test)
  
mcc_train = matthews_corrcoef(y_train_rus, y_train_pred)
mcc_test = matthews_corrcoef(y_test, y_test_pred)
mcc_cv = cv['test_MCC'].mean()

# Display model performance results
train_df1_labels = pd.Series(['MCC_train', 'MCC_CV', 'MCC_test'], name = 'Performance_metric_names')
train_df1_values = pd.Series([mcc_train, mcc_cv, mcc_test], name = 'Performance_metric_values')
train_df3 = pd.concat([train_df1_labels, train_df1_values], axis=1)
train_df3


# ### Naive Bayes

# In[302]:


model = GaussianNB() 
model.fit(X_train_rus, y_train_rus) 

# Training Cross-validation Models
from sklearn.metrics import make_scorer, recall_score, matthews_corrcoef
from sklearn.model_selection import cross_validate

model_cv = GaussianNB()
cv_scoring = {'MCC': make_scorer(matthews_corrcoef)}
cv = cross_validate(model_cv, X_train_rus, y_train_rus, cv=5, scoring=cv_scoring)

# Apply model to make prediction
from sklearn.metrics import matthews_corrcoef

y_train_pred = model.predict(X_train_rus)
y_test_pred = model.predict(X_test)
  
mcc_train = matthews_corrcoef(y_train_rus, y_train_pred)
mcc_test = matthews_corrcoef(y_test, y_test_pred)
mcc_cv = cv['test_MCC'].mean()

# Display model performance results
train_df1_labels = pd.Series(['MCC_train', 'MCC_CV', 'MCC_test'], name = 'Performance_metric_names')
train_df1_values = pd.Series([mcc_train, mcc_cv, mcc_test], name = 'Performance_metric_values')
train_df7 = pd.concat([train_df1_labels, train_df1_values], axis=1)
train_df7


# ### KNN 

# In[303]:


model = KNeighborsClassifier(n_neighbors=2) #1
model.fit(X_train_rus, y_train_rus)

# Training Cross-validation Models
from sklearn.metrics import make_scorer, recall_score, matthews_corrcoef
from sklearn.model_selection import cross_validate

model_cv = KNeighborsClassifier(n_neighbors=2) #1
cv_scoring = {'MCC': make_scorer(matthews_corrcoef)}
cv = cross_validate(model_cv, X_train_rus, y_train_rus, cv=5, scoring=cv_scoring)

# Apply model to make prediction
from sklearn.metrics import matthews_corrcoef

y_train_pred = model.predict(X_train_rus)
y_test_pred = model.predict(X_test)

mcc_train = matthews_corrcoef(y_train_rus, y_train_pred)
mcc_test = matthews_corrcoef(y_test, y_test_pred)
mcc_cv = cv['test_MCC'].mean()

# Display model performance results
train_df1_labels = pd.Series(['MCC_train', 'MCC_CV', 'MCC_test'], name = 'Performance_metric_names')
train_df1_values = pd.Series([mcc_train, mcc_cv, mcc_test], name = 'Performance_metric_values')
train_df11 = pd.concat([train_df1_labels, train_df1_values], axis=1)
train_df11


# # Model Building with Oversampled Balanced Class

# ### Decision Tree 

# In[304]:


model = DecisionTreeClassifier(random_state=42)
model.fit(X_train_ros, y_train_ros)

from sklearn.metrics import make_scorer, recall_score, matthews_corrcoef
from sklearn.model_selection import cross_validate

model_cv = DecisionTreeClassifier(random_state=42) #METHODOLOGY : instantiating decision tree classifier
cv_scoring = {'MCC': make_scorer(matthews_corrcoef)}

cv = cross_validate(model_cv, X_train_ros, y_train_ros, cv=5, scoring=cv_scoring)

# Apply model to make prediction
from sklearn.metrics import matthews_corrcoef

y_train_pred = model.predict(X_train_ros) #making predictions -> put in x and get y
y_test_pred = model.predict(X_test) # put in x for test set and get y
  
mcc_train = matthews_corrcoef(y_train_ros, y_train_pred) #calculating mcc
mcc_test = matthews_corrcoef(y_test, y_test_pred)
mcc_cv = cv['test_MCC'].mean() # bc have 5 fold and list of 5 values so wil calculate mean of that

# Display model performance results -> creates summary data frame
train_df_labels = pd.Series(['MCC_train', 'MCC_CV', 'MCC_test'], name = 'Performance_metric_names')
train_df_values = pd.Series([mcc_train, mcc_cv, mcc_test], name = 'Performance_metric_values')
train_df4 = pd.concat([train_df_labels, train_df_values], axis=1)
train_df4


# ### Naive Bayes 

# In[305]:


model = GaussianNB() 
model.fit(X_train_ros, y_train_ros) 

# Training Cross-validation Models
from sklearn.metrics import make_scorer, recall_score, matthews_corrcoef
from sklearn.model_selection import cross_validate

model_cv = GaussianNB()
cv_scoring = {'MCC': make_scorer(matthews_corrcoef)}
cv = cross_validate(model_cv, X_train_rus, y_train_rus, cv=5, scoring=cv_scoring)

# Apply model to make prediction
from sklearn.metrics import matthews_corrcoef

y_train_pred = model.predict(X_train_rus)
y_test_pred = model.predict(X_test)
  
mcc_train = matthews_corrcoef(y_train_rus, y_train_pred)
mcc_test = matthews_corrcoef(y_test, y_test_pred)
mcc_cv = cv['test_MCC'].mean()

# Display model performance results
train_df_labels = pd.Series(['MCC_train', 'MCC_CV', 'MCC_test'], name = 'Performance_metric_names')
train_df_values = pd.Series([mcc_train, mcc_cv, mcc_test], name = 'Performance_metric_values')
train_df8 = pd.concat([train_df_labels, train_df_values], axis=1)
train_df8


# ### KNN

# In[306]:


model = KNeighborsClassifier(n_neighbors=2) 
model.fit(X_train, y_train)

# Training Cross-validation Models
from sklearn.metrics import make_scorer, recall_score, matthews_corrcoef
from sklearn.model_selection import cross_validate

model_cv = KNeighborsClassifier(n_neighbors=2) 
cv_scoring = {'MCC': make_scorer(matthews_corrcoef)}
cv = cross_validate(model_cv, X_train, y_train, cv=5, scoring=cv_scoring)

# Apply model to make prediction
from sklearn.metrics import matthews_corrcoef

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
  
mcc_train = matthews_corrcoef(y_train, y_train_pred)
mcc_test = matthews_corrcoef(y_test, y_test_pred)
mcc_cv = cv['test_MCC'].mean()

# Display model performance results
train_df_labels = pd.Series(['MCC_train', 'MCC_CV', 'MCC_test'], name = 'Performance_metric_names')
train_df_values = pd.Series([mcc_train, mcc_cv, mcc_test], name = 'Performance_metric_values')
train_df12 = pd.concat([train_df_labels, train_df_values], axis=1)
train_df12


# # Model Building with Class Weight Balancing

# ### Decision Tree 

# In[307]:


model = DecisionTreeClassifier(random_state=42, class_weight='balanced') # Class weight balancing
model.fit(X_train, y_train)

# Training Cross-validation Models
from sklearn.metrics import make_scorer, recall_score, matthews_corrcoef
from sklearn.model_selection import cross_validate

model_cv = DecisionTreeClassifier(random_state=42, class_weight='balanced') # Class weight balancing
cv_scoring = {'MCC': make_scorer(matthews_corrcoef)}
cv = cross_validate(model_cv, X_train, y_train, cv=5, scoring=cv_scoring)

# Apply model to make prediction
from sklearn.metrics import matthews_corrcoef

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
  
mcc_train = matthews_corrcoef(y_train, y_train_pred)
mcc_test = matthews_corrcoef(y_test, y_test_pred)
mcc_cv = cv['test_MCC'].mean()

# Display model performance results
train_df1_labels = pd.Series(['MCC_train', 'MCC_CV', 'MCC_test'], name = 'Performance_metric_names')
train_df1_values = pd.Series([mcc_train, mcc_cv, mcc_test], name = 'Performance_metric_values')
train_df5 = pd.concat([train_df1_labels, train_df1_values], axis=1)
train_df5


# ### Naive Bayes

# In[308]:


model = GaussianNB() 
model.fit(X_train, y_train)

# Training Cross-validation Models
from sklearn.metrics import make_scorer, recall_score, matthews_corrcoef
from sklearn.model_selection import cross_validate

model_cv = GaussianNB()
cv_scoring = {'MCC': make_scorer(matthews_corrcoef)}
cv = cross_validate(model_cv, X_train, y_train, cv=5, scoring=cv_scoring)

# Apply model to make prediction
from sklearn.metrics import matthews_corrcoef

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
  
mcc_train = matthews_corrcoef(y_train, y_train_pred)
mcc_test = matthews_corrcoef(y_test, y_test_pred)
mcc_cv = cv['test_MCC'].mean()

# Display model performance results
train_df1_labels = pd.Series(['MCC_train', 'MCC_CV', 'MCC_test'], name = 'Performance_metric_names')
train_df1_values = pd.Series([mcc_train, mcc_cv, mcc_test], name = 'Performance_metric_values')
train_df9 = pd.concat([train_df1_labels, train_df1_values], axis=1)
train_df9


# ### KNN 

# In[309]:


model = KNeighborsClassifier(n_neighbors=2) 
model.fit(X_train, y_train)

# Training Cross-validation Models
from sklearn.metrics import make_scorer, recall_score, matthews_corrcoef
from sklearn.model_selection import cross_validate

model_cv = KNeighborsClassifier(n_neighbors=2) 
cv_scoring = {'MCC': make_scorer(matthews_corrcoef)}
cv = cross_validate(model_cv, X_train, y_train, cv=5, scoring=cv_scoring)

# Apply model to make prediction
from sklearn.metrics import matthews_corrcoef

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
  
mcc_train = matthews_corrcoef(y_train, y_train_pred)
mcc_test = matthews_corrcoef(y_test, y_test_pred)
mcc_cv = cv['test_MCC'].mean()

# Display model performance results
train_df1_labels = pd.Series(['MCC_train', 'MCC_CV', 'MCC_test'], name = 'Performance_metric_names')
train_df1_values = pd.Series([mcc_train, mcc_cv, mcc_test], name = 'Performance_metric_values')
train_df13 = pd.concat([train_df1_labels, train_df1_values], axis=1)
train_df13


# # Summary Table

# ### Decision Tree

# In[310]:


#import numpy as np
df = pd.DataFrame(pd.np.empty((0, 3))) 
train_df_1 = pd.concat([train_df2.Performance_metric_values,
                train_df3.Performance_metric_values,
                train_df4.Performance_metric_values,
                train_df5.Performance_metric_values], axis=1)
train_df_1.columns = ['No class balancing', 'Class balancing (undersampling)', 'Class balancing (oversampling)', 'Class balancing (class weights)']
train_df_1 = train_df_1.T
train_df_1.columns = ['Training', 'CV', 'Test']
train_df_1


# ### Naive Bayes 

# In[311]:


#Naive bayes (6 - w/o class balancing,
#7 - w undersampled class balancing,
#8 - with undersampled class balancing,
#9 - )
train_df_2 = pd.concat([train_df6.Performance_metric_values,
                train_df7.Performance_metric_values,
                train_df8.Performance_metric_values,
                train_df9.Performance_metric_values], axis=1)
train_df_2.columns = ['No class balancing', 'Class balancing (undersampling)', 'Class balancing (oversampling)', 'Class balancing (class weights)']
train_df_2 = train_df_2.T
train_df_2.columns = ['Training', 'CV', 'Test']
train_df_2


# ###  KNN

# In[312]:


train_df_3 = pd.concat([train_df10.Performance_metric_values,
                train_df11.Performance_metric_values,
                train_df12.Performance_metric_values,
                train_df13.Performance_metric_values], axis=1)
train_df_3.columns = ['No class balancing', 'Class balancing (undersampling)', 'Class balancing (oversampling)', 'Class balancing (class weights)']
train_df_3 = train_df_3.T
train_df_3.columns = ['Training', 'CV', 'Test']
train_df_3


# # Prediction

# ### Decision Tree Prediction 

# In[313]:


from sklearn.tree import DecisionTreeClassifier
DecisionTreeclassifier = DecisionTreeClassifier()
DecisionTreeclassifier.fit(X_train, y_train)


# In[314]:


X_train = X[221:1100]
X_test = X[0:220]
y_train = y[221:1100] #making predictions -> put in x and get y
y_test = y[0:220] 

from sklearn.tree import DecisionTreeClassifier
DecisionTreeClassifier = DecisionTreeClassifier()
DecisionTreeClassifier.fit(X_train.astype(int), y_train.astype(int))

from sklearn.metrics import accuracy_score

DecisionTreeClassifier_pred = DecisionTreeClassifier.predict(X_test.astype(int))
accuracy_score(y_test.astype(int), DecisionTreeClassifier_pred)


# In[315]:


y_pred = DecisionTreeclassifier.predict(X_test)


# In[316]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[317]:


from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)


# In[318]:


y_pred = regressor.predict(X_test)


# In[319]:


#compare predicted values with actual values to how accurate predictions were.
df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
df


# In[321]:


#Decision Tree PREDICTION
decisiontree_pred = DecisionTreeclassifier.predict(X[1000:1101])
print(decisiontree_pred)


# # KNN Prediction

# In[322]:


from sklearn.neighbors import KNeighborsClassifier
KNNclassifier = KNeighborsClassifier()
KNNclassifier.fit(X_train, y_train)


# In[323]:


y2_pred = KNNclassifier.predict(X_test)


# In[324]:


#PREDICTION missing class labels
knn_pred = KNNclassifier.predict(X[1000:1101])
print(knn_pred)


# In[325]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y2_pred))
print(classification_report(y_test, y2_pred))


# # Save Prediction to CSV File

# In[326]:


prediction = pd.DataFrame(knn_pred, columns=['predict'])
prediction = prediction.rename_axis(index='ID')
prediction.to_csv('D:/2025/DATA MINING/bikim_20591364/Predict_1.csv')


# In[327]:


decision_tree_prediction = pd.DataFrame(decisiontree_pred, columns=['predict'])
decision_tree_prediction = decision_tree_prediction.rename_axis(index='ID')
decision_tree_prediction.to_csv('D:/2025/DATA MINING/bikim_20591364/Predict_2.csv')


# # Prediction Accuracy

# # Decision Tree

# In[328]:


from sklearn.model_selection import train_test_split

X_train_pred = X[221:1100]
X_test_pred = X[0:220]
y_train_pred = y[221:1100] #making predictions -> put in x and get y
y_test_pred = y[0:220] 

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(X_train_pred.astype(int), y_train_pred.astype(int))


from sklearn.metrics import accuracy_score

y_predicted = tree.predict(X_test.astype(int))
accuracy_score(y_test_pred.astype(int), y_predicted)


# # KNN

# In[329]:


from sklearn.model_selection import train_test_split

X_train_pred = X[221:1100]
X_test_pred = X[0:220]
y_train_pred = y[221:1100] #making predictions -> put in x and get y
y_test_pred = y[0:220] 

from sklearn.tree import DecisionTreeClassifier
knn = KNeighborsClassifier()
knn.fit(X_train_pred.astype(int), y_train_pred.astype(int))


from sklearn.metrics import accuracy_score

y_predicted = knn.predict(X_test.astype(int))
accuracy_score(y_test_pred.astype(int), y_predicted)


# In[ ]:


names = ["Decision Tree", "K Nearest Neighbours"]

