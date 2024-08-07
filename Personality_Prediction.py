#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


md = pd.read_csv("C:\\Users\\bisoy\\Downloads\\data-final.csv", sep='\t')


# # md.shape()

# In[3]:


data = md.copy()


# In[4]:


#removing last columns which are of no use
data.drop(data.columns[50:107], axis=1, inplace=True)
data.drop(data.columns[51:], axis=1, inplace=True)


# In[5]:


data.head()


# In[6]:


data.shape


# In[7]:


data.describe()


# In[8]:


#checking the missing values
#print(data.isnull().values.sum())


# In[9]:


#dopping the missing values;
#data.dropna(inplace=True)


# # Pre-Processing Data

# In[10]:


data.info()


# In[11]:


print("The null values:")
print(data.isnull().sum())


# In[12]:


data_cleaned=data.dropna()


# In[13]:


data.head()


# In[14]:


print("The null values:")
print(data_cleaned.isnull().sum())


# In[15]:


seed_value=42
df_subset=data.sample(n=100000,random_state=seed_value)
df_subset.head()


# In[16]:


df_subset.shape


# In[17]:


# Participants' nationality distriution
countries = pd.DataFrame(df_subset['country'].value_counts())        #gets the total number of participants from countries in numbers
countries_5000 = countries[countries['country'] >= 5000]
plt.figure(figsize=(15,5))
sns.barplot(data=countries_5000, x=countries_5000.index, y='country')
plt.title('Countries With More Than 5000 Participants')
plt.ylabel('Participants');


# In[18]:


ext_questions = {'EXT1' : 'I am the life of the party',
                 'EXT2' : 'I dont talk a lot',
                 'EXT3' : 'I feel comfortable around people',
                 'EXT4' : 'I keep in the background',
                 'EXT5' : 'I start conversations',
                 'EXT6' : 'I have little to say',
                 'EXT7' : 'I talk to a lot of different people at parties',
                 'EXT8' : 'I dont like to draw attention to myself',
                 'EXT9' : 'I dont mind being the center of attention',
                 'EXT10': 'I am quiet around strangers'}

est_questions = {'EST1' : 'I get stressed out easily',
                 'EST2' : 'I am relaxed most of the time',
                 'EST3' : 'I worry about things',
                 'EST4' : 'I seldom feel blue',
                 'EST5' : 'I am easily disturbed',
                 'EST6' : 'I get upset easily',
                 'EST7' : 'I change my mood a lot',
                 'EST8' : 'I have frequent mood swings',
                 'EST9' : 'I get irritated easily',
                 'EST10': 'I often feel blue'}

agr_questions = {'AGR1' : 'I feel little concern for others',
                 'AGR2' : 'I am interested in people',
                 'AGR3' : 'I insult people',
                 'AGR4' : 'I sympathize with others feelings',
                 'AGR5' : 'I am not interested in other peoples problems',
                 'AGR6' : 'I have a soft heart',
                 'AGR7' : 'I am not really interested in others',
                 'AGR8' : 'I take time out for others',
                 'AGR9' : 'I feel others emotions',
                 'AGR10': 'I make people feel at ease'}

csn_questions = {'CSN1' : 'I am always prepared',
                 'CSN2' : 'I leave my belongings around',
                 'CSN3' : 'I pay attention to details',
                 'CSN4' : 'I make a mess of things',
                 'CSN5' : 'I get chores done right away',
                 'CSN6' : 'I often forget to put things back in their proper place',
                 'CSN7' : 'I like order',
                 'CSN8' : 'I shirk my duties',
                 'CSN9' : 'I follow a schedule',
                 'CSN10' : 'I am exacting in my work'}

opn_questions = {'OPN1' : 'I have a rich vocabulary',
                 'OPN2' : 'I have difficulty understanding abstract ideas',
                 'OPN3' : 'I have a vivid imagination',
                 'OPN4' : 'I am not interested in abstract ideas',
                 'OPN5' : 'I have excellent ideas',
                 'OPN6' : 'I do not have a good imagination',
                 'OPN7' : 'I am quick to understand things',
                 'OPN8' : 'I use difficult words',
                 'OPN9' : 'I spend time reflecting on things',
                 'OPN10': 'I am full of ideas'}

# Group Names and Columns
EXT = [column for column in data if column.startswith('EXT')]
EST = [column for column in data if column.startswith('EST')]
AGR = [column for column in data if column.startswith('AGR')]
CSN = [column for column in data if column.startswith('CSN')]
OPN = [column for column in data if column.startswith('OPN')]


# In[19]:


# Function to visualize the all questions and answers
def questionAnswers(groupName, questions, color):
    plt.figure(figsize=(40,60))
    for i in range(1, 11):
        plt.subplot(10,5,i)             
        plt.hist(data[groupName[i-1]], bins=14, color= color,alpha=.5) 
        plt.title(questions[groupName[i-1]], fontsize=18)


# In[20]:


print('Extroversion Personality')
questionAnswers(EXT, ext_questions, 'purple')


# In[21]:


print('Neuroticism Personality')
questionAnswers(EST, est_questions, 'black')


# In[22]:


print('Agreeable Personality')
questionAnswers(AGR, agr_questions, 'orange')


# In[23]:


print('Conscientious Personality')
questionAnswers(CSN, csn_questions, 'blue')


# In[24]:


print('Openness Personality')
questionAnswers(OPN, opn_questions, 'green')


# In[25]:


#**Clustering:**


# In[26]:


# New Section


# # Clustering

# In[27]:


#k-Means clustering
from sklearn.cluster import KMeans

new_data = df_subset.drop('country', axis=1);

kmeans = KMeans(n_clusters=5);
new_data.dropna(inplace=True)
k_fit = kmeans.fit(new_data);               #Data to be clustered


# In[28]:


pd.options.display.max_columns = 50;
predictions = k_fit.labels_
new_data['Clusters'] = predictions;
new_data.head()


# In[29]:



x = new_data.iloc[:, 0:50] 
print(x) 
y = new_data.iloc[:, 50:51]  
print(y)


# In[30]:


# Import train_test_split function
from sklearn.model_selection import train_test_split



# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,)


# In[ ]:





# In[31]:


#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)


# In[32]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[33]:


print(y_pred)


# In[34]:


new_data.Clusters.value_counts()


# In[35]:


pd.options.display.max_columns=50
new_data.groupby('Clusters').mean()


# In[36]:


#**Checking the pattern after grouping**


# In[37]:


list_of_col = list(new_data)
ext = list_of_col[0:10]
est = list_of_col[10:20]
agr = list_of_col[20:30]
csn = list_of_col[30:40]
opn = list_of_col[40:50]

sum_data = pd.DataFrame()
sum_data['extroversion'] = new_data[ext].sum(axis=1)/10
sum_data['neurotic'] = new_data[est].sum(axis=1)/10
sum_data['agreeable'] = new_data[agr].sum(axis=1)/10
sum_data['conscientious'] = new_data[csn].sum(axis=1)/10
sum_data['open'] = new_data[opn].sum(axis=1)/10
sum_data['clusters'] = predictions
sum_data.groupby('clusters').mean()


# In[38]:


#visualizing some clusters
dataclusters = sum_data.groupby('clusters').mean()
plt.figure(figsize=(22,6))
for i in range(0, 5):
    plt.subplot(1,5,i+1)
    plt.bar(dataclusters.columns, dataclusters.iloc[:, i], color='green', alpha=0.2)
    plt.plot(dataclusters.columns, dataclusters.iloc[:, i], color='red')
    plt.title('Cluster ' + str(i))
    plt.xticks(rotation=45)
    plt.ylim(0,4);


# In[39]:


# using PCA to visualize in 2D graph
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_fit = pca.fit_transform(new_data)

df_pca = pd.DataFrame(data=pca_fit, columns=['PCA1', 'PCA2'])
df_pca['Clusters'] = predictions
df_pca.head()


# In[40]:


plt.figure(figsize=(10,10))
sns.scatterplot(data=df_pca, x='PCA1', y='PCA2', hue='Clusters', palette='Set2', alpha=0.8)
plt.title('Personality Clusters after PCA');


# # Training models

# In[ ]:





# In[41]:


from sklearn.metrics import accuracy_score

accuracies = {}

#Random Forest
random_forest = RandomForestClassifier(n_estimators=100, random_state = 1)
random_forest.fit(X_train, y_train)

# make predictions for test data
Y_pred = random_forest.predict(X_test)
predictions = [round(value) for value in Y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
accuracies['Random Forest'] = accuracy* 100.0 
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[42]:


# from sklearn.metrics import plot_confusion_matrix
# print("Random Forest")
# plot_confusion_matrix(random_forest,X_test,y_test)


# In[43]:


# from sklearn.tree import DecisionTreeClassifier
# dt = DecisionTreeClassifier()
# dt.fit(X_train, y_train)
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score
# dt = DecisionTreeClassifier()
# dt.fit(X_train, y_train)

# Y_pred = dt.predict(X_test)
# predictions = [round(value) for value in Y_pred]

# # evaluate predictions
# accuracy = accuracy_score(y_test, predictions)
# accuracies['Decision Tree'] = accuracy* 100.0
# print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[44]:


# from sklearn.metrics import plot_confusion_matrix
# print("Decision Tree")
# plot_confusion_matrix(dt,X_test,y_test)


# In[45]:


from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

Y_pred = logreg.predict(X_test)
predictions = [round(value) for value in Y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
accuracies['Logistic Regression'] = accuracy* 100.0
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[46]:


# from sklearn.metrics import plot_confusion_matrix
# print("Logistic Regression")
# plot_confusion_matrix(logreg,X_test,y_test)


# In[47]:


#KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 2)  # n_neighbors means k
knn.fit(X_train, y_train)

Y_pred = knn.predict(X_test)
predictions = [round(value) for value in Y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
accuracies['KNN'] = accuracy* 100.0
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#try to find best k value
# scoreList = []
# for i in range(1,20):
#     knn2 = KNeighborsClassifier(n_neighbors = i)  # n_neighbors means k
#     knn2.fit(X_train, y_train)
#     scoreList.append(knn2.score(X_test, y_test))
    
# plt.plot(range(1,20), scoreList)
# plt.xticks(np.arange(1,20,1))
# plt.xlabel("K value")
# plt.ylabel("Score")
# plt.show()

# acc = max(scoreList)*100

# print("Maximum KNN Score is {:.2f}%".format(acc))


# In[48]:


# from sklearn.metrics import plot_confusion_matrix
# print("KNN")
# plot_confusion_matrix(knn,X_test,y_test)


# In[49]:


#XG boost Classifier
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score
# xgb = XGBClassifier()
# xgb.fit(X_train,y_train)

# Y_pred = xgb.predict(X_test)
# predictions = [round(value) for value in Y_pred]

# # evaluate predictions
# accuracy = accuracy_score(y_test, predictions)
# accuracies['XG Boost'] = accuracy* 100.0
# print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[50]:


# #Gradient Descent
# from xgboost import XGBClassifier
# from sklearn.linear_model import SGDClassifier
# from sklearn.metrics import accuracy_score

# sgd = SGDClassifier(max_iter=5, tol=None)
# sgd.fit(X_train, y_train)

# Y_pred = sgd.predict(X_test)
# predictions = [round(value) for value in Y_pred]

# # evaluate predictions
# accuracy = accuracy_score(y_test, predictions)
# accuracies['Gradient Descent'] = accuracy* 100.0
# print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[51]:


# from sklearn.svm import SVC
# svm = SVC(random_state = 1)
# svm.fit(X_train, y_train)

# Y_pred = svm.predict(X_test)

# predictions = [round(value) for value in Y_pred]
# # evaluate predictions
# accuracy = accuracy_score(y_test, predictions)
# accuracies['SVC'] = accuracy* 100.0
# print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[52]:


# from sklearn.metrics import plot_confusion_matrix
# print("SVC")
# plot_confusion_matrix(knn,X_test,y_test)


# In[53]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

Y_pred = gnb.predict(X_test)

predictions = [round(value) for value in Y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
accuracies['GNB'] = accuracy* 100.0
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[54]:


pd.DataFrame.from_dict(accuracies, orient='index', columns=['Accuracies(%)'])


# In[55]:


import numpy as np
colors = ["purple", "green", "orange", "magenta","#CFC60E","#0FBBAE"]

sns.set_style("whitegrid")
plt.figure(figsize=(16,5))
plt.yticks(np.arange(0,100,10))
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette=colors)
plt.show()


# # Testing with own data

# In[56]:


my_data = pd.read_excel("C:\\Users\\bisoy\\Downloads\\Untitled spreadsheet.xlsx")
my_data


# In[57]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
my_data_scaled = scaler.fit_transform(my_data)


# In[58]:


import pandas as pd

# Assuming my_data_scaled is a NumPy array
my_data_scaled_df = pd.DataFrame(my_data_scaled, columns=my_data.columns)

# Display information about the DataFrame
print(my_data_scaled_df.info())

# Display summary statistics
print(my_data_scaled_df.describe())


# In[59]:


my_data_scaled = np.nan_to_num(my_data_scaled)


# In[60]:


my_data_scaled = my_data_scaled.astype('float64')


# In[61]:


my_personality = k_fit.predict(my_data_scaled)
print('My Personality Cluster: ', my_personality)


# In[62]:


# Summing up the my question groups
col_list = list(my_data)
ext = col_list[0:10]
est = col_list[10:20]
agr = col_list[20:30]
csn = col_list[30:40]
opn = col_list[40:50]

my_sums = pd.DataFrame()
my_sums['extroversion'] = my_data[ext].sum(axis=1)/10
my_sums['neurotic'] = my_data[est].sum(axis=1)/10
my_sums['agreeable'] = my_data[agr].sum(axis=1)/10
my_sums['conscientious'] = my_data[csn].sum(axis=1)/10
my_sums['open'] = my_data[opn].sum(axis=1)/10
my_sums['cluster'] = my_personality
print('Sum of my question groups')
my_sums


# In[63]:


my_sum = my_sums.drop('cluster', axis=1)
plt.bar(my_sum.columns, my_sum.iloc[0,:], color='green', alpha=0.2)
plt.plot(my_sum.columns, my_sum.iloc[0,:], color='red')
plt.title('Cluster 2')
plt.xticks(rotation=45)
plt.ylim(0,4);


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




