**Network Intrusion Detection ML**   
A clean, end‑to‑end machine learning pipeline for detecting malicious network activity. This project applies supervised classification, structured data preprocessing, and cross‑validated model evaluation to build a reliable intrusion detection system. It demonstrates practical skills in data cleaning, feature engineering, model selection, and cybersecurity‑focused analytics.
    
     
**What This Project Does**
* Cleans and preprocesses real‑world network activity data
* Handles missing values, duplicates, mixed data types, and class imbalance
* Trains multiple ML classifiers (k‑NN, Naïve Bayes, Decision Tree, plus optional ensembles)
* Performs cross‑validation and hyperparameter tuning
* Selects the best model using accuracy and F1‑score
* Generates final predictions for two unseen test datasets
This project was developed in Python using Jupyter Notebook.


**How to Run**   
Install Dependencies   
All the required packages will need to be installed using "pip install <package_name>" for this program to work. Some of the packages required include pandas, numpy,imblearn seaborn, csv, StandardScaler, RandomUnderSampler, GridSearchCV, KNeighborsClassifier, GaussianNB, DecisionTreeClassifier, train_test_split, accuracy_score, train_test_split,scikit-learn and precision_score.   
  
Run with Python:    
Use "python run.py" in the command terminal.
    
Run with Jupyter Notebook:   
Open and execute all cells in the run.ipynb file.   
   
The pipeline will automatically preprocess the data, train and evaluate models, and generate both prediction files.
