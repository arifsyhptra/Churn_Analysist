#!/usr/bin/env python
# coding: utf-8

# ## Import package

# In[36]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ## Import data

# In[9]:


df=pd.read_csv('Churn.csv')
df.head()


# In[10]:


df.drop('customerID',axis='columns',inplace=True)


# In[11]:


df.info()


# In[12]:


df.describe()


# # Data Visualization

# In[25]:


df.nunique()


# In[26]:


def print_unique_col_values(df):
       for column in df:
            if df[column].dtypes=='object':
                print(f'{column}: {df[column].unique()}') 


# In[27]:


print_unique_col_values(df)


# In[28]:


def checkValues(df, name):
    print("For No : ")
    temp = df[df[name]=='No' ]
    print(temp.shape)
    print(temp["Churn"].value_counts())
    
    print("\nFor Yes : ")
    temp = df[df[name]=='Yes' ]
    print(temp.shape)
    print(temp["Churn"].value_counts())
    
    print("\nFor No Internet Service : ")
    temp = df[df[name]=='No internet service' ]
    print(temp.shape)
    print(temp["Churn"].value_counts())


# In[29]:


checkValues(df, "OnlineSecurity")


# In[30]:


checkValues(df, "OnlineSecurity")


# In[31]:


checkValues(df, "OnlineBackup")


# In[32]:


tenure_churn_no = df[df.Churn=='No'].tenure
tenure_churn_yes = df[df.Churn=='Yes'].tenure

plt.xlabel("tenure")
plt.ylabel("Number Of Customers")
plt.title("Customer Churn Prediction Visualiztion")

plt.hist([tenure_churn_yes, tenure_churn_no], rwidth=0.95, color=['blue','pink'],label=['Churn=Yes','Churn=No'])
plt.legend()


# In[33]:


mc_churn_no = df[df.Churn=='No'].MonthlyCharges      
mc_churn_yes = df[df.Churn=='Yes'].MonthlyCharges      

plt.xlabel("Monthly Charges")
plt.ylabel("Number Of Customers")
plt.title("Customer Churn Prediction Visualiztion")

plt.hist([mc_churn_yes, mc_churn_no], rwidth=0.95, color=['yellow','red'],label=['Churn=Yes','Churn=No'])
plt.legend()


# In[34]:


print_unique_col_values(df)


# In[37]:


fig, ax = plt.subplots(4, 3, figsize = (15, 20))
plt.suptitle('Count plot for various categorical features\n', fontsize = 30, color= 'teal')

ax1 = sns.countplot(x ='gender', data= df, hue= 'Churn', ax= ax[0, 0], palette= 'spring')
ax1.set(xlabel = 'Gender')

ax2 = sns.countplot(x ='Partner', data= df, hue= 'Churn', ax= ax[0, 1], palette= 'summer')
ax2.set(xlabel = 'Partner')

ax3 = sns.countplot(x ='Dependents', data= df, hue= 'Churn', ax= ax[0, 2], palette= 'viridis')
ax3.set(xlabel = 'Dependents')

ax4 = sns.countplot(x ='InternetService', data= df, hue= 'Churn', ax= ax[1, 0], palette= 'rocket')
ax4.set(xlabel = 'Internet Service')

ax5 = sns.countplot(x ='OnlineSecurity', data= df, hue= 'Churn', ax= ax[1, 1], palette= 'winter')
ax5.set(xlabel = 'Online Security')

ax6 = sns.countplot(x ='OnlineBackup', data= df, hue= 'Churn', ax= ax[1, 2], palette= 'mako')
ax6.set(xlabel = 'Online Backup')

ax7 = sns.countplot(x ='DeviceProtection', data= df, hue= 'Churn', ax= ax[2, 0], palette= 'flare')
ax7.set(xlabel = 'Device Protection')

ax8 = sns.countplot(x ='TechSupport', data= df, hue= 'Churn', ax= ax[2, 1], palette= 'crest')
ax8.set(xlabel = 'Tech Support')

ax9 = sns.countplot(x ='StreamingTV', data= df, hue= 'Churn', ax= ax[2, 2], palette= 'magma')
ax9.set(xlabel = 'Streaming TV')

ax10 = sns.countplot(x ='StreamingMovies', data= df, hue= 'Churn', ax= ax[3, 0], palette= 'viridis')
ax10.set(xlabel = 'Streaming Movies')

ax11 = sns.countplot(x ='Contract', data= df, hue= 'Churn', ax= ax[3, 1], palette= 'rocket_r')
ax11.set(xlabel = 'Contract')



plt.tight_layout()
plt.show()


# In[38]:


fig, ax = plt.subplots(figsize = (10, 5))

ax = sns.countplot(x ='PaymentMethod', data= df, hue= 'Churn', palette= 'spring')
ax.set(xlabel = 'Payment Method')

plt.show()


# In[39]:


# # Here Yes == No Internet Services 

# df["OnlineBackup"].replace('No internet service','Yes',inplace=True)
# df["OnlineSecurity"].replace('No internet service','Yes',inplace=True)
# df["DeviceProtection"].replace('No internet service','Yes',inplace=True)
# df["TechSupport"].replace('No internet service','Yes',inplace=True)


# In[40]:


# # Here No == No Internet Services 

# df["StreamingTV"].replace('No internet service','No',inplace=True)
# df["StreamingMovies"].replace('No internet service','No',inplace=True)
# df.replace('No phone service','No',inplace=True)


# In[41]:


df.replace('No internet service','No',inplace=True)
df.replace('No phone service','No',inplace=True)


# In[42]:


print_unique_col_values(df)


# In[43]:


df.sample()


# ## Convert Yes and No to 1 or 0

# In[45]:


for col in df:
    print(f'{col}: {df[col].unique()}') 


# In[46]:


df['gender'].replace({'Female':1,'Male':0},inplace=True)


# ## One hot encoding for categorical columns

# In[47]:


df1 = pd.get_dummies(data=df, columns=['InternetService','Contract','PaymentMethod'])
df1.columns


# In[48]:


df1.sample(2)


# In[49]:


df1.dtypes


# In[50]:


df1.TotalCharges.values


# In[51]:


df1.iloc[488].TotalCharges


# ### So there is some white space(" ") , we should remove that

# In[52]:


pd.to_numeric(df1.TotalCharges,errors='coerce').isnull()


# In[53]:


df1[pd.to_numeric(df1.TotalCharges,errors='coerce').isnull()]


# ### Remove rows with space in TotalCharges

# In[54]:


df1 = df1[df1.TotalCharges!=' ']
df1.shape


# In[55]:


df1.TotalCharges = pd.to_numeric(df1.TotalCharges)


# In[56]:


df1.info()


# In[57]:


df1.corr()


# ## Scaling

# In[58]:


cols_to_scale = ['tenure','MonthlyCharges','TotalCharges']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df1[cols_to_scale] = scaler.fit_transform(df1[cols_to_scale])


# ## Train Test Split

# In[59]:


X = df1.drop(["Churn", "gender", "Partner", "PhoneService", "Contract_Month-to-month", "Contract_One year", "Contract_Two year"], axis='columns')
y = df1["Churn"]


# In[60]:


y.value_counts()


# ## Handle Imabalance Dataset

# In[67]:


# from imblearn.over_sampling import SMOTE

# sm = SMOTE(random_state=28)
# X_res, y_res = sm.fit_resample(X, y)


# ## Build a model (ANN) in tensorflow/keras

# In[69]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0, stratify=y)


# In[70]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[72]:


import tensorflow as tf
from tensorflow import keras


# In[76]:


model = keras.Sequential([
    keras.layers.Dense(40, input_shape=(20,), activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1, activation='sigmoid')
])

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'model.h5', 
    monitor = 'val_loss'
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=5, verbose=0,
    mode='auto', baseline=None, restore_best_weights= True
)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[tf.keras.metrics.AUC(name='auc')]
)


BATCH_SIZE = 50
EPOCHS = 40

history = model.fit(
    X_train,
    y_train,
    validation_split=0.15,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=0,
    callbacks = [early_stopping, checkpoint])


# In[77]:


model.evaluate(X_test, y_test)


# In[79]:


y_pred = model.predict(X_test)


# In[80]:


y_pred[0]


# In[81]:


y_pred = y_pred.reshape(y_pred.shape[0])


# In[31]:


data[data['TotalCharges'].isna()]


# In[82]:


y_pred = y_pred.round()


# In[83]:


y_pred[0]


# In[ ]:





# In[84]:


history.history.keys()


# In[85]:


dict_keys(['loss', 'auc', 'val_loss', 'val_auc'])


# In[86]:


plt.figure(figsize=(10, 6))


train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_loss) + 1)

plt.plot(epochs, train_loss, label="Training Loss")
plt.plot(epochs, val_loss, label="Validation Loss")

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()


# In[87]:


fig, ax = plt.subplots()


ax.set_title("Accuracy")
ax.plot(history.history['auc'], label="Training accuracy")
ax.plot(history.history['val_auc'], label="Validation accuracy")
plt.legend()
plt.show()


# In[ ]:




