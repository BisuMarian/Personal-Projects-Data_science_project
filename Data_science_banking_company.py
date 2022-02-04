from turtle import color
import matplotlib
import pandas as pd
import matplotlib.pyplot as mt
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
import sklearn
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.linear_model import LarsCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, MiniBatchKMeans, k_means
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier 
from sklearn.tree import plot_tree

df = pd.read_csv('BankChurners2.csv')

pd.set_option('display.max_columns', 500)

#Cleaning the data

df.pop('CLIENTNUM')
df.pop('Total_Trans_Ct')

df.rename(columns={'Attrition_Flag':'Attrition','Customer_Age':'Age','Dependent_count':'Dep_ct',
                    'Education_Level':'Education','Marital_Status':'Marital','Income_Category':'Income',
                    'Months_on_book':'Months','Credit_Limit':'Credit_L','Total_Revolving_Bal':
                    'Rev_balance','Total_Trans_Amt':'Trans_Amt'
                    }, inplace=True)

#Checking for missing values

print(df.head(),'\n')

print(df.isna().sum(),'\n')

#DESCRIPTIVE STATISTICS

print(df.describe())

#Histogram plots for every continuous variable

figure, axis = mt.subplots(2,3, figsize = (10,5))
axis[0,0].hist(data = df, x = 'Credit_L' ,color = 'g', ec = 'black', label = 'Credit_L')
axis[0,0].set_title("Credit Limit")
axis[0,0].set_xlabel('Max value of the credit')
axis[0,0].set_ylabel('Number of clients')

axis[0,1].hist(data = df, x = 'Dep_ct', color = 'y', ec = 'black')
axis[0,1].set_title("Dependency count")
axis[0,1].set_xlabel('Number of persons in dependency')
axis[0,1].set_ylabel('Number of clients')

axis[0,2].hist(data = df, x = 'Age', color = 'b', ec = 'black')
axis[0,2].set_title("Age")
axis[0,2].set_xlabel('Age')
axis[0,2].set_ylabel('Number of clients')

axis[1,0].hist(data = df, x = 'Rev_balance',color = 'r', ec = 'black')
axis[1,0].set_title("Revolving balance")
axis[1,0].set_xlabel('Final balance of depth')
axis[1,0].set_ylabel('Number of clients')

axis[1,1].hist(data = df, x = 'Trans_Amt', color = 'm', ec = 'black')
axis[1,1].set_title("Total transfers amount")
axis[1,1].set_xlabel('Total sume tranzactionate')
axis[1,1].set_ylabel('Number of clients')

axis[1,2].hist(data = df, x = 'Months', color = 'c', ec = 'black')
axis[1,2].set_title("Months on record")
axis[1,2].set_xlabel('Months since registered')
axis[1,2].set_ylabel('Number of clients')
mt.tight_layout(pad = 3.0)
mt.show()

# Pie chart plot for every categorial variable

figure , ax = mt.subplots(2,2, figsize = (10,5))
df['Gender'].value_counts().plot.pie( explode = [0,0.1], autopct = '%1.1f%%', shadow = True, ax= ax[0][0]).set_ylabel('')
ax[0][0].set_title("Gender")
df['Education'].value_counts().plot.pie( explode = [0,0.1,0.1,0.1,0.1,0.1,0.1], autopct = '%1.1f%%', shadow = True, ax= ax[0][1]).set_ylabel('')
ax[0][1].set_title("Education Level")
df['Attrition'].value_counts().plot.pie( explode = [0,0.1], autopct = '%1.1f%%', shadow = True, ax= ax[1][0]).set_ylabel('')
ax[1][0].set_title("Attrition Status")
df['Income'].value_counts().plot.pie( explode = [0,0.1,0.1,0.1,0.1,0.1], autopct = '%1.1f%%', shadow = True, ax= ax[1][1]).set_ylabel('')
ax[1][1].set_title("Income Category")

mt.tight_layout(pad = 3.0)
mt.show()

#PRINCIPAL COMPONENT ANALYSIS

features = ['Rev_balance', 'Trans_Amt', 'Months','Credit_L']
features = df.iloc[:,7:11]
scaled_data = preprocessing.scale(features.T)
pca = PCA()
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)

per_var = np.round(pca.explained_variance_ratio_*100, decimals = 1)

print(per_var)
#Scree Plot
labels = ['PC' + str(i) for i in range(1, len(per_var)+1)]
mt.bar(x = range(1, len(per_var)+1), height = per_var, tick_label = labels)
mt.ylabel('Percentage of explained variance')
mt.xlabel('Principal component')
mt.title('Scree Plot of PCA Analysis')
mt.show()

features = ['Rev_balance', 'Trans_Amt', 'Months','Credit_L']

x = df.loc[:, features].values
y = df.loc[:,['Attrition']].values 

# Standardizing the features
x = StandardScaler().fit_transform(x)

#Implementing PCA
pca = PCA(n_components= 2)

principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, df[['Attrition']]], axis = 1)

fig = mt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['Attrited Customer', 'Existing Customer']
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Attrition'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 5)
ax.legend(targets)
ax.grid()
mt.show()

# CLUSTER ANALYSIS - K-means method

#Elbow Plot
k_range = range(1, 10)
sse = []
for k in k_range:
    km = KMeans(n_clusters=k)
    km.fit(df[['Trans_Amt','Credit_L']])
    sse.append(km.inertia_)

mt.xlabel('Number of clusters')
mt.ylabel('Sum o squared errors')
mt.plot(k_range, sse)
mt.title('Elbow Plot')
mt.show()

km = KMeans(n_clusters= 3)
y_predicted = km.fit_predict(df[['Trans_Amt','Credit_L']])
df['Cluster'] = y_predicted
print(df.head())
kmeans = MiniBatchKMeans(n_clusters=3)
kmeans.fit(df[['Trans_Amt','Credit_L']])
df1 = df[df.Cluster == 0]
df2 = df[df.Cluster == 1]
df3 = df[df.Cluster == 2]
centroids = kmeans.cluster_centers_

mt.scatter(df1.Trans_Amt,df1.Credit_L, color = 'green', s = 5)
mt.scatter(df2.Trans_Amt,df2.Credit_L, color = 'blue', s = 5)
mt.scatter(df3.Trans_Amt,df3.Credit_L, color = 'red',s = 5)
mt.scatter(centroids[:, 0], centroids[:, 1],  marker = "*", s=10, 
    linewidths = 5, zorder = 10, c='purple')
mt.show()

df = df.drop('Cluster', axis = 'columns')

# DECISION TREE

features_cols = ['Education','Marital','Income','Months','Credit_L']

inputs = df[features_cols]
target = df.Attrition

#Coding of the categorial variables
le_education = LabelEncoder()
le_marital  = LabelEncoder()
le_income = LabelEncoder()

inputs['Education'] = le_education.fit_transform(inputs['Education'])
Education_map = dict(zip(le_education.classes_, range(len(le_education.classes_)))) 
inputs['Marital'] = le_marital.fit_transform(inputs['Marital'])
Marital_map = dict(zip(le_marital.classes_, range(len(le_marital.classes_))))
inputs['Income'] = le_income.fit_transform(inputs['Income'])
Income_map = dict(zip(le_income.classes_, range(len(le_income.classes_))))

print(Education_map, Marital_map, Income_map)

X = inputs
y = target

#Training the model by creating test and train samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state = 1)
clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy: ", metrics.accuracy_score(y_test,y_pred))

#Optimizing the Decision Tree
clf = DecisionTreeClassifier(criterion="entropy", max_depth=4)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy: ", metrics.accuracy_score(y_test,y_pred))
fig = mt.figure(figsize=(20,20))
_ = tree.plot_tree(clf, 
                   feature_names=features_cols, 
                   class_names=['Attrited','Existing'],
                   filled=True)
mt.show()

importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(clf.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False)
print(importances)

#This predicts if a customer is considered attrited or existing using the 
#created model by giving some random values for the implemented features
print(clf.predict([[2,1,5,49,1500]]))