#!/usr/bin/env python
# coding: utf-8

# # IMPORTING ALL LIBRARIES

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# # READ DATA SETS

# In[2]:


train=pd.read_csv(r"C:\Users\Admin AM\Downloads\Property_Price_Train.csv")   #read train file
train


# In[3]:


train.info()


# In[4]:


test=pd.read_csv(r"C:\Users\Admin AM\Downloads\Property_Price_Test.csv")   #read test file
test


# In[5]:


test.info()


# In[6]:


train.head()    # target


# In[7]:


test.head()


# In[8]:


train.tail()


# In[9]:


test.tail()


# In[10]:


train.shape,test.shape


# # CHECK NULL VALUES OF TRAIN FILE AND REMOVE IT

# In[11]:


train.describe()


# In[12]:


pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)
train.drop_duplicates()
train.duplicated().value_counts()


# In[13]:


train.isnull().sum()  #null values


# In[14]:


100*train.isnull().sum()/len(train)          #PERCENTAGE OF VALUE COUNTS 


# In[15]:


train=train.drop(["Lane_Type","Pool_Quality","Fireplace_Quality","Fence_Quality","Miscellaneous_Feature","Garage_Built_Year","Garage_Finish_Year","Month_Sold","Year_Sold","Miscellaneous_Value","Brick_Veneer_Area","Brick_Veneer_Type"],axis=1)


# In[16]:


train.isnull().sum()


# In[17]:


#train.Lot_Extent.value_counts()
#train.Basement_Height.value_counts()
#train.Basement_Condition.value_counts()
#train.Exposure_Level.value_counts()
#train.BsmtFinType1.value_counts()
#train.BsmtFinType2.value_counts()
#train.Electrical_System.value_counts()
#train.Garage.value_counts()
#train.Garage_Quality.value_counts()
train.Garage_Condition.value_counts()


# In[18]:


train.Lot_Extent=train.Lot_Extent.fillna(train.Lot_Extent.mean()) 
train.Basement_Height=train.Basement_Height.fillna("Fa")
train.Basement_Condition=train.Basement_Condition.fillna("Po")
train.Exposure_Level=train.Exposure_Level.fillna("Mn")
train.BsmtFinType1=train.BsmtFinType1.fillna("LwQ")
train.BsmtFinType2=train.BsmtFinType2.fillna("GLQ")
train.Electrical_System=train.Electrical_System.fillna("Mix")
train.Garage=train.Garage.fillna("2Types")
train.Garage_Quality=train.Garage_Quality.fillna("Po")
train.Garage_Condition=train.Garage_Condition.fillna("Ex")


# In[19]:


train.isnull().sum()  #null values


# # CHECK DATA TYPES OF TRAIN FILE AND CONVERSION FROM CATEG. TO NUMER

# In[20]:


train.dtypes 


# In[21]:


#train.Lot_Extent.value_counts()
#train.Zoning_Class.value_counts()
#train.Road_Type.value_counts()
#train.Property_Shape.value_counts()
#train.Land_Outline.value_counts()
#train.Utility_Type.value_counts()
#train.Neighborhood.value_counts()
#train.Lot_Configuration.value_counts()
#train.Property_Slope.value_counts()
#train.Condition1.value_counts()
#train.Condition2.value_counts()
#train.House_Type.value_counts()
#train.House_Design.value_counts()
#train.Roof_Design.value_counts()
#train.Roof_Quality.value_counts()
#train.Exterior1st.value_counts()
#train.Exterior2nd.value_counts()
#train.Exterior_Material.value_counts()
#train.Exterior_Condition.value_counts()
#train.Foundation_Type.value_counts()
#train.Basement_Height.value_counts()
#train.Basement_Condition.value_counts()
#train.Exposure_Level.value_counts()
#train.BsmtFinType2.value_counts()
#train.Heating_Type.value_counts()
#train.Heating_Quality.value_counts()
#train.Air_Conditioning.value_counts()
#train.Kitchen_Quality.value_counts()
#train.Functional_Rate.value_counts()
#train.Garage.value_counts()
#train.Garage_Quality.value_counts()
#train.Garage_Condition.value_counts()
#train.Pavedd_Drive.value_counts()
#train.Sale_Type.value_counts()
#train.Sale_Condition.value_counts()
#train.BsmtFinType1.value_counts()
#train.Electrical_System.value_counts()


# In[22]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[23]:


train.Zoning_Class=le.fit_transform(train.Zoning_Class)
train.Road_Type=le.fit_transform(train.Road_Type)
train.Property_Shape=le.fit_transform(train.Property_Shape)
train.Land_Outline=le.fit_transform(train.Land_Outline)
train.Utility_Type=le.fit_transform(train.Utility_Type)
train.Neighborhood=le.fit_transform(train.Neighborhood)
train.Lot_Configuration=le.fit_transform(train.Lot_Configuration)
train.Property_Slope=le.fit_transform(train.Property_Slope)
train.Condition1=le.fit_transform(train.Condition1)
train.Condition2=le.fit_transform(train.Condition2)
train.House_Type=le.fit_transform(train.House_Type)
train.House_Design=le.fit_transform(train.House_Design)
train.Roof_Design=le.fit_transform(train.Roof_Design)
train.Roof_Quality=le.fit_transform(train.Roof_Quality)
train.Exterior1st=le.fit_transform(train.Exterior1st)
train.Exterior2nd=le.fit_transform(train.Exterior2nd)
train.Exterior_Material=le.fit_transform(train.Exterior_Material)
train.Exterior_Condition=le.fit_transform(train.Exterior_Condition)
train.Foundation_Type=le.fit_transform(train.Foundation_Type)
train.Basement_Height=le.fit_transform(train.Basement_Height)
train.Basement_Condition=le.fit_transform(train.Basement_Condition)
train.Exposure_Level=le.fit_transform(train.Exposure_Level)
train.BsmtFinType2=le.fit_transform(train.BsmtFinType2)
train.Heating_Type=le.fit_transform(train.Heating_Type)
train.Heating_Quality=le.fit_transform(train.Heating_Quality)
train.Air_Conditioning=le.fit_transform(train.Air_Conditioning)
train.Kitchen_Quality=le.fit_transform(train.Kitchen_Quality)
train.Functional_Rate=le.fit_transform(train.Functional_Rate)
train.Garage=le.fit_transform(train.Garage)
train.Garage_Quality=le.fit_transform(train.Garage_Quality)
train.Garage_Condition=le.fit_transform(train.Garage_Condition)
train.Pavedd_Drive=le.fit_transform(train.Pavedd_Drive)
train.Sale_Type=le.fit_transform(train.Sale_Type)
train.Sale_Condition=le.fit_transform(train.Sale_Condition)
train.BsmtFinType1=le.fit_transform(train.BsmtFinType1)
train.Electrical_System=le.fit_transform(train.Electrical_System)


# In[24]:


train['Lot_Extent']=pd.to_numeric(train['Lot_Extent'],errors='coerce')
train['Lot_Extent']=train['Lot_Extent'].astype('int64')
train['Garage_Area']=pd.to_numeric(train['Garage_Area'],errors='coerce')
train['Garage_Area']=train['Garage_Area'].astype('int64')
train['W_Deck_Area']=pd.to_numeric(train['W_Deck_Area'],errors='coerce')
train['W_Deck_Area']=train['W_Deck_Area'].astype('int64')
train['Open_Lobby_Area']=pd.to_numeric(train['Open_Lobby_Area'],errors='coerce')
train['Open_Lobby_Area']=train['Open_Lobby_Area'].astype('int64')
train['Enclosed_Lobby_Area']=pd.to_numeric(train['Enclosed_Lobby_Area'],errors='coerce')
train['Enclosed_Lobby_Area']=train['Enclosed_Lobby_Area'].astype('int64')


# # test data check null values and remove it

# In[25]:


test.describe()


# In[26]:


pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)
test.drop_duplicates()
test.duplicated().value_counts()


# In[27]:


test.isnull().sum()     #null values


# In[28]:


100*test.isnull().sum()/len(test)


# In[29]:


test=test.drop(["Lane_Type","Fireplace_Quality","Pool_Quality","Fence_Quality","Miscellaneous_Feature","Brick_Veneer_Area","Garage_Built_Year","Month_Sold","Year_Sold","Garage_Finish_Year","Miscellaneous_Value","Brick_Veneer_Type","Miscellaneous_Value"],axis=1)


# In[30]:


test.Zoning_Class.value_counts()
test.Lot_Extent.value_counts()
test.Utility_Type.value_counts()
test.Exterior1st.value_counts()
test.Basement_Height.value_counts()
test.Basement_Condition.value_counts()
test.Exposure_Level.value_counts()
test.BsmtFinType1.value_counts()
test.BsmtFinSF1.value_counts()
test.BsmtFinType2.value_counts()
test.BsmtFinSF2.value_counts()
test.BsmtUnfSF.value_counts()
test.Total_Basement_Area.value_counts()
test.Underground_Full_Bathroom.value_counts()
test.Underground_Half_Bathroom.value_counts()
test.Kitchen_Quality.value_counts()
test.Functional_Rate.value_counts()
test.Garage.value_counts()
test.Garage_Size.value_counts()
test.Garage_Area.value_counts()
test.Garage_Quality.value_counts()
test.Garage_Condition.value_counts()
test.Sale_Type.value_counts()


# In[31]:


test.Zoning_Class=test.Zoning_Class.fillna("RHD")
test.Lot_Extent=test.Lot_Extent.fillna(test.Lot_Extent.mean())
test.Utility_Type=test.Utility_Type.fillna("AllPub")
test.Exterior1st=test.Exterior1st.fillna("CB")
test.Exterior2nd=test.Exterior2nd.fillna("Stone")
test.Basement_Height=test.Basement_Height.fillna("Fa")
test.Basement_Condition=test.Basement_Condition.fillna("Po")
test.Exposure_Level=test.Exposure_Level.fillna("Mn")
test.BsmtFinType1=test.BsmtFinType1.fillna("LwQ")
test.BsmtFinSF1=test.BsmtFinSF1.fillna(test.BsmtFinSF1.mean())
test.BsmtFinType2=test.BsmtFinType2.fillna("GLQ")
test.BsmtFinSF2=test.BsmtFinSF2.fillna(test.BsmtFinSF2.mean())
test.BsmtUnfSF=test.BsmtUnfSF.fillna(test.BsmtUnfSF.mean())
test.Total_Basement_Area=test.Total_Basement_Area.fillna(test.Total_Basement_Area.mean())
test.Underground_Full_Bathroom=test.Underground_Full_Bathroom.fillna(3.0)
test.Underground_Half_Bathroom=test.Underground_Half_Bathroom.fillna(2.0)
test.Kitchen_Quality=test.Kitchen_Quality.fillna("Fa")
test.Functional_Rate=test.Functional_Rate.fillna("MS")
test.Garage=test.Garage.fillna("CarPort")
test.Garage_Size=test.Garage_Size.fillna(test.Garage_Size.mean())
test.Garage_Area=test.Garage_Area.fillna(test.Garage_Area.mean())
test.Garage_Quality=test.Garage_Quality.fillna("Po")
test.Garage_Condition=test.Garage_Condition.fillna("Ex")
test.Sale_Type=test.Sale_Type.fillna("Con")


# In[32]:


test.isnull().sum()


# # test data check dtypes and convert categ to numer

# In[33]:


test.dtypes          #check data types of test file


# In[34]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[35]:


test.Zoning_Class=le.fit_transform(test.Zoning_Class)
test.Road_Type=le.fit_transform(test.Road_Type)
test.Property_Shape=le.fit_transform(test.Property_Shape)
test.Land_Outline=le.fit_transform(test.Land_Outline)
test.Utility_Type=le.fit_transform(test.Utility_Type)
test.Lot_Configuration=le.fit_transform(test.Lot_Configuration)
test.Property_Slope=le.fit_transform(test.Property_Slope)
test.Neighborhood=le.fit_transform(test.Neighborhood)
test.Condition1=le.fit_transform(test.Condition1)
test.Condition2=le.fit_transform(test.Condition2)
test.House_Type=le.fit_transform(test.House_Type)
test.House_Design=le.fit_transform(test.House_Design)
test.Roof_Design=le.fit_transform(test.Roof_Design)
test.Roof_Quality=le.fit_transform(test.Roof_Quality)
test.Exterior1st=le.fit_transform(test.Exterior1st)
test.Exterior2nd=le.fit_transform(test.Exterior2nd)
test.Exterior_Material=le.fit_transform(test.Exterior_Material)
test.Exterior_Condition=le.fit_transform(test.Exterior_Condition)
test.Foundation_Type=le.fit_transform(test.Foundation_Type)
test.Basement_Height=le.fit_transform(test.Basement_Height)
test.Basement_Condition=le.fit_transform(test.Basement_Condition)
test.Exposure_Level=le.fit_transform(test.Exposure_Level)
test.BsmtFinType1=le.fit_transform(test.BsmtFinType1)
test.BsmtFinType2=le.fit_transform(test.BsmtFinType2)
test.Heating_Type=le.fit_transform(test.Heating_Type)
test.Heating_Quality=le.fit_transform(test.Heating_Quality)
test.Air_Conditioning=le.fit_transform(test.Air_Conditioning)
test.Electrical_System=le.fit_transform(test.Electrical_System)
test.Kitchen_Quality=le.fit_transform(test.Kitchen_Quality)
test.Functional_Rate=le.fit_transform(test.Functional_Rate)
test.Garage=le.fit_transform(test.Garage)
test.Garage_Quality=le.fit_transform(test.Garage_Quality)
test.Garage_Condition=le.fit_transform(test.Garage_Condition)
test.Pavedd_Drive=le.fit_transform(test.Pavedd_Drive)
test.Sale_Type=le.fit_transform(test.Sale_Type)
test.Sale_Condition=le.fit_transform(test.Sale_Condition)


# In[36]:


test['Lot_Extent']=pd.to_numeric(test['Lot_Extent'],errors='coerce')
test['Lot_Extent']=test['Lot_Extent'].astype('int64')
test['Lot_Size']=pd.to_numeric(test['Lot_Size'],errors='coerce')
test['Lot_Size']=test['Lot_Size'].astype('int64')
test['BsmtFinSF1']=pd.to_numeric(test['BsmtFinSF1'],errors='coerce')
test['BsmtFinSF1']=test['BsmtFinSF1'].astype('int64')
test['BsmtFinSF2']=pd.to_numeric(test['BsmtFinSF2'],errors='coerce')
test['BsmtFinSF2']=test['BsmtFinSF2'].astype('int64')
test['BsmtUnfSF']=pd.to_numeric(test['BsmtUnfSF'],errors='coerce')
test['BsmtUnfSF']=test['BsmtUnfSF'].astype('int64')
test['Total_Basement_Area']=pd.to_numeric(test['Total_Basement_Area'],errors='coerce')
test['Total_Basement_Area']=test['Total_Basement_Area'].astype('int64')
test['Underground_Full_Bathroom']=pd.to_numeric(test['Underground_Full_Bathroom'],errors='coerce')
test['Underground_Full_Bathroom']=test['Underground_Full_Bathroom'].astype('int64')
test['Underground_Half_Bathroom']=pd.to_numeric(test['Underground_Half_Bathroom'],errors='coerce')
test['Underground_Half_Bathroom']=test['Underground_Half_Bathroom'].astype('int64')
test['Garage_Size']=pd.to_numeric(test['Garage_Size'],errors='coerce')
test['Garage_Size']=test['Garage_Size'].astype('int64')
test['Garage_Area']=pd.to_numeric(test['Garage_Area'],errors='coerce')
test['Garage_Area']=test['Garage_Area'].astype('int64')


# # BASIC MODEL BUILDING

# In[37]:


tr1=train.copy()


# In[38]:


tr1_x=tr1.iloc[:,1:-1]
tr1_y=tr1.iloc[:,-1]


# In[39]:


from sklearn.model_selection import train_test_split


# In[40]:


tr1_xtrain,tr1_xtest,tr1_ytrain,tr1_ytest=train_test_split(tr1_x,tr1_y,test_size=.3,random_state=101)


# In[41]:


from sklearn.model_selection import train_test_split


# In[42]:


tr1_xtrain.shape,tr1_xtest.shape,tr1_ytrain.shape,tr1_ytest.shape


# In[43]:


from sklearn import linear_model


# In[44]:


ln=linear_model.LinearRegression()


# In[45]:


ln.fit(tr1_xtrain,tr1_ytrain)


# In[46]:


prediction=ln.predict(tr1_xtest)


# In[47]:


ln.intercept_


# In[48]:


ln.coef_


# In[49]:


rsq=ln.score(tr1_xtrain,tr1_ytrain)
rsq


# In[50]:


adjR2=1-(((1-rsq)*(1021-1))/(1021-66-1))
adjR2


# In[51]:


from sklearn import metrics


# In[52]:


mae=metrics.mean_absolute_error(tr1_ytest,prediction)
mae


# In[53]:


mse=metrics.mean_squared_error(tr1_ytest,prediction)
mse


# In[54]:


error=tr1_ytest-prediction
error


# In[55]:


absolute_error=np.abs(error)
absolute_error


# In[56]:


mape=np.mean(absolute_error/tr1_ytest)*100
mape


# # VISUALIZATION

# In[58]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[59]:


fig,ax=plt.subplots(21,4,figsize=(100,95))
sns.countplot("Id",data=train,ax=ax[0][0])
sns.countplot("Zoning_Class",data=train,ax=ax[0][1])
sns.countplot("Lot_Extent",data=train,ax=ax[0][2])
sns.countplot("Lot_Size",data=train,ax=ax[0][3])
sns.countplot("Road_Type",data=train,ax=ax[1][0])
sns.countplot("Property_Shape",data=train,ax=ax[1][1])
sns.countplot("Land_Outline",data=train,ax=ax[1][2])
sns.countplot("Utility_Type",data=train,ax=ax[1][3])
sns.countplot("Lot_Configuration",data=train,ax=ax[2][0])
sns.countplot("Property_Slope",data=train,ax=ax[2][1])
sns.countplot("Neighborhood",data=train,ax=ax[2][2])
sns.countplot("Condition2",data=train,ax=ax[2][3])
sns.countplot("House_Type",data=train,ax=ax[3][0])
sns.countplot("House_Design",data=train,ax=ax[3][1])
sns.countplot("Overall_Material",data=train,ax=ax[3][2])
sns.countplot("House_Condition",data=train,ax=ax[3][3])
sns.countplot("Construction_Year",data=train,ax=ax[4][0])
sns.countplot("Remodel_Year",data=train,ax=ax[4][1])
sns.countplot("Roof_Design",data=train,ax=ax[4][2])
sns.countplot("Roof_Quality",data=train,ax=ax[4][3])
sns.countplot("Exterior1st",data=train,ax=ax[5][0])
sns.countplot("Exterior2nd",data=train,ax=ax[5][1])
sns.countplot("Building_Class",data=train,ax=ax[5][2])
#sns.countplot("Brick_Veneer_Area",data=train,ax=ax[5][3])
#sns.countplot("Brick_Veneer_Area",data=train,ax=ax[6][0])
sns.countplot("Exterior_Material",data=train,ax=ax[6][1])
sns.countplot("Exterior_Condition",data=train,ax=ax[6][2])
sns.countplot("Basement_Height",data=train,ax=ax[6][3])
sns.countplot("Foundation_Type",data=train,ax=ax[7][0])
sns.countplot("Basement_Condition",data=train,ax=ax[7][1])
sns.countplot("Exposure_Level",data=train,ax=ax[7][2])
sns.countplot("BsmtFinType1",data=train,ax=ax[7][3])
sns.countplot("BsmtFinSF1",data=train,ax=ax[8][0])
sns.countplot("BsmtFinType2",data=train,ax=ax[8][1])
sns.countplot("BsmtFinSF2",data=train,ax=ax[8][2])
sns.countplot("BsmtUnfSF",data=train,ax=ax[8][3])
sns.countplot("Total_Basement_Area",data=train,ax=ax[9][0])
sns.countplot("Electrical_System",data=train,ax=ax[9][1])
sns.countplot("Heating_Quality",data=train,ax=ax[9][2])
sns.countplot("Air_Conditioning",data=train,ax=ax[9][3])
sns.countplot("Electrical_System",data=train,ax=ax[10][0])
sns.countplot("First_Floor_Area",data=train,ax=ax[10][1])
sns.countplot("Second_Floor_Area",data=train,ax=ax[10][2])
sns.countplot("LowQualFinSF",data=train,ax=ax[10][3])
sns.countplot("Grade_Living_Area",data=train,ax=ax[11][0])
sns.countplot("Underground_Full_Bathroom",data=train,ax=ax[11][1])
sns.countplot("Underground_Half_Bathroom",data=train,ax=ax[11][2])
sns.countplot("Full_Bathroom_Above_Grade",data=train,ax=ax[11][3])
sns.countplot("Half_Bathroom_Above_Grade",data=train,ax=ax[12][0])
sns.countplot("Bedroom_Above_Grade",data=train,ax=ax[12][1])
sns.countplot("Kitchen_Above_Grade",data=train,ax=ax[12][2])
sns.countplot("Kitchen_Quality",data=train,ax=ax[12][3])
sns.countplot("Rooms_Above_Grade",data=train,ax=ax[13][0])
sns.countplot("Functional_Rate",data=train,ax=ax[13][1])
sns.countplot("Fireplaces",data=train,ax=ax[13][2])
#sns.countplot("Fireplace_Quality",data=train,ax=ax[13][3])
sns.countplot("Garage",data=train,ax=ax[14][0])
#sns.countplot("Garage_Built_Year",data=train,ax=ax[14][1])
#sns.countplot("Garage_Finish_Year",data=train,ax=ax[14][2])
sns.countplot("Garage_Size",data=train,ax=ax[14][3])
sns.countplot("Garage_Area",data=train,ax=ax[15][0])
sns.countplot("Garage_Quality",data=train,ax=ax[15][1])
sns.countplot("Garage_Condition",data=train,ax=ax[15][2])
sns.countplot("Pavedd_Drive",data=train,ax=ax[15][3])
sns.countplot("W_Deck_Area",data=train,ax=ax[16][0])
sns.countplot("Open_Lobby_Area",data=train,ax=ax[16][1])
sns.countplot("Enclosed_Lobby_Area",data=train,ax=ax[16][2])
sns.countplot("Three_Season_Lobby_Area",data=train,ax=ax[16][3])
sns.countplot("Screen_Lobby_Area",data=train,ax=ax[17][0])
sns.countplot("Pool_Area",data=train,ax=ax[17][1])
#sns.countplot("Pool_Quality",data=train,ax=ax[18][3])
#sns.countplot("Fence_Quality",data=train,ax=ax[19][0])
#sns.countplot("Miscellaneous_Feature",data=train,ax=ax[19][1])
#sns.countplot("Miscellaneous_Value",data=train,ax=ax[17][2])
#sns.countplot("Month_Sold",data=train,ax=ax[17][3])
#sns.countplot("Year_Sold",data=train,ax=ax[18][0])
sns.countplot("Sale_Type",data=train,ax=ax[18][1])
sns.countplot("Sale_Condition",data=train,ax=ax[18][2])
sns.countplot("Sale_Price",data=train,ax=ax[18][3])


# In[60]:


fig,ax=plt.subplots(21,4,figsize=(50,45))
sns.countplot("Id",data=test,ax=ax[0][0])
sns.countplot("Zoning_Class",data=test,ax=ax[0][1])
sns.countplot("Lot_Extent",data=test,ax=ax[0][2])
sns.countplot("Lot_Size",data=test,ax=ax[0][3])
sns.countplot("Road_Type",data=test,ax=ax[1][0])
#sns.countplot("Lane_Type",data=test,ax=ax[1][1])
sns.countplot("Property_Shape",data=test,ax=ax[1][1])
sns.countplot("Land_Outline",data=test,ax=ax[1][2])
sns.countplot("Utility_Type",data=test,ax=ax[1][3])
sns.countplot("Lot_Configuration",data=test,ax=ax[2][0])
sns.countplot("Property_Slope",data=test,ax=ax[2][1])
sns.countplot("Neighborhood",data=test,ax=ax[2][2])
sns.countplot("Condition1",data=test,ax=ax[2][3])
sns.countplot("Condition2",data=test,ax=ax[3][0])
sns.countplot("House_Type",data=test,ax=ax[3][1])
sns.countplot("House_Design",data=test,ax=ax[3][2])
sns.countplot("Overall_Material",data=test,ax=ax[3][3])
sns.countplot("House_Condition",data=test,ax=ax[4][0])
sns.countplot("Construction_Year",data=test,ax=ax[4][1])
sns.countplot("Remodel_Year",data=test,ax=ax[4][2])
sns.countplot("Roof_Design",data=test,ax=ax[4][3])
sns.countplot("Roof_Quality",data=test,ax=ax[5][0])
sns.countplot("Exterior1st",data=test,ax=ax[5][1])
sns.countplot("Exterior2nd",data=test,ax=ax[5][2])
sns.countplot("Building_Class",data=test,ax=ax[5][3])
#sns.countplot("Brick_Veneer_Area",data=test,ax=ax[6][0])
#sns.countplot("Brick_Veneer_Area",data=test,ax=ax[6][1])
sns.countplot("Exterior_Material",data=test,ax=ax[6][2])
sns.countplot("Exterior_Condition",data=test,ax=ax[6][3])
sns.countplot("Basement_Height",data=test,ax=ax[7][0])
sns.countplot("Foundation_Type",data=test,ax=ax[7][1])
sns.countplot("Basement_Condition",data=test,ax=ax[7][2])
sns.countplot("Exposure_Level",data=test,ax=ax[7][3])
sns.countplot("BsmtFinType1",data=test,ax=ax[8][0])
sns.countplot("BsmtFinSF1",data=test,ax=ax[8][1])
sns.countplot("BsmtFinType2",data=test,ax=ax[8][2])
sns.countplot("BsmtFinSF2",data=test,ax=ax[8][3])
sns.countplot("BsmtUnfSF",data=test,ax=ax[9][0])
sns.countplot("Total_Basement_Area",data=test,ax=ax[9][1])
sns.countplot("Electrical_System",data=test,ax=ax[9][2])
sns.countplot("Heating_Quality",data=test,ax=ax[9][3])
sns.countplot("Air_Conditioning",data=test,ax=ax[10][0])
sns.countplot("Electrical_System",data=test,ax=ax[10][1])
sns.countplot("First_Floor_Area",data=test,ax=ax[10][2])
sns.countplot("Second_Floor_Area",data=test,ax=ax[10][3])
sns.countplot("LowQualFinSF",data=test,ax=ax[11][0])
sns.countplot("Grade_Living_Area",data=test,ax=ax[11][1])
sns.countplot("Underground_Full_Bathroom",data=test,ax=ax[11][2])
sns.countplot("Underground_Half_Bathroom",data=test,ax=ax[11][3])
sns.countplot("Full_Bathroom_Above_Grade",data=test,ax=ax[12][0])
sns.countplot("Half_Bathroom_Above_Grade",data=test,ax=ax[12][1])
sns.countplot("Bedroom_Above_Grade",data=test,ax=ax[12][2])
sns.countplot("Kitchen_Above_Grade",data=test,ax=ax[12][3])
sns.countplot("Kitchen_Quality",data=test,ax=ax[13][0])
sns.countplot("Rooms_Above_Grade",data=test,ax=ax[13][1])
sns.countplot("Functional_Rate",data=test,ax=ax[13][2])
sns.countplot("Fireplaces",data=test,ax=ax[13][3])
#sns.countplot("Fireplace_Quality",data=test,ax=ax[15][0])
sns.countplot("Garage",data=test,ax=ax[14][0])
#sns.countplot("Garage_Built_Year",data=test,ax=ax[14][1])
#sns.countplot("Garage_Finish_Year",data=test,ax=ax[14][2])
sns.countplot("Garage_Size",data=test,ax=ax[14][3])
sns.countplot("Garage_Area",data=test,ax=ax[15][0])
sns.countplot("Garage_Quality",data=test,ax=ax[15][1])
sns.countplot("Garage_Condition",data=test,ax=ax[15][2])
sns.countplot("Pavedd_Drive",data=test,ax=ax[15][3])
sns.countplot("W_Deck_Area",data=test,ax=ax[16][0])
sns.countplot("Open_Lobby_Area",data=test,ax=ax[16][1])
sns.countplot("Enclosed_Lobby_Area",data=test,ax=ax[16][2])
sns.countplot("Three_Season_Lobby_Area",data=test,ax=ax[16][3])
sns.countplot("Screen_Lobby_Area",data=test,ax=ax[17][0])
sns.countplot("Pool_Area",data=test,ax=ax[17][1])
#sns.countplot("Pool_Quality",data=test,ax=ax[18][3])
#sns.countplot("Fence_Quality",data=test,ax=ax[19][0])
#sns.countplot("Miscellaneous_Feature",data=test,ax=ax[19][1])
#sns.countplot("Miscellaneous_Value",data=test,ax=ax[17][2])
#sns.countplot("Month_Sold",data=test,ax=ax[17][3])
#sns.countplot("Year_Sold",data=test,ax=ax[18][0])
sns.countplot("Sale_Type",data=test,ax=ax[18][1])
sns.countplot("Sale_Condition",data=test,ax=ax[18][2])


# In[61]:


tr2=train.corr()
tr2


# In[62]:


plt.figure(figsize=(70,65))
heatmap=sns.heatmap(tr2,linewidth=1,annot=True,cmap=plt.cm.Blues)
plt.title("Heatmap using Seaborn Method")
plt.show()


# #  DETECT OUTLIERS AND TREAT THEM OF TRAIN FILE

# In[63]:


train.boxplot(figsize=(30,25))


# In[81]:


prout=train.copy()


# In[82]:


#train._get_numeric_data().columns


# In[83]:


#train.Sale_Price.value_counts()


# In[84]:


#train.boxplot(column="Id")
#train.boxplot(column="Lot_Size")
#train.boxplot(column="Lot_Size")
#train.boxplot(column="BsmtFinSF1")
#train.boxplot(column="Lot_Extent")
#train.boxplot(column="BsmtFinSF2")
#train.boxplot(column="BsmtUnfSF")
#train.boxplot(column="Total_Basement_Area")
#train.boxplot(column="First_Floor_Area")
#train.boxplot(column="Second_Floor_Area")
#train.boxplot(column="LowQualFinSF")
#train.boxplot(column="Grade_Living_Area")
#train.boxplot(column="Garage_Area")
#train.boxplot(column="W_Deck_Area")
#train.boxplot(column="Open_Lobby_Area")
#train.boxplot(column="Enclosed_Lobby_Area")
#train.boxplot(column="Three_Season_Lobby_Area")
#train.boxplot(column="Screen_Lobby_Area")
train.boxplot(column="Sale_Price")


# In[85]:


#train.hist(["Id"],bins=20)
#train.hist(["BsmtFinSF1"])
#train.hist(["Lot_Extent"])
#train.hist(["BsmtFinSF2"])
#train.hist(["BsmtFinSF1"])
#train.hist(["Total_Basement_Area"])
#train.hist(["First_Floor_Area"])
#train.hist(["Second_Floor_Area"])
#train.hist(["LowQualFinSF"])
#train.hist(["Grade_Living_Area"])
#train.hist(["Garage_Area"])
#train.hist(["W_Deck_Area"])
#train.hist(["Open_Lobby_Area"],bins=50)
#train.hist(["Enclosed_Lobby_Area"],bins=50)
#train.hist(["Three_Season_Lobby_Area"],bins=50)
#train.hist(["Screen_Lobby_Area"],bins=50)
train.hist(["Sale_Price"],bins=50)


# In[86]:


#sns.distplot(train["Id"])
#sns.distplot(train["Lot_Size"])
#sns.distplot(train["BsmtFinSF1"])
#sns.distplot(train["Lot_Extent"])
#sns.distplot(train["BsmtFinSF2"])
#sns.distplot(train["BsmtUnfSF"])
#sns.distplot(train["Total_Basement_Area"])
#sns.distplot(train["First_Floor_Area"])
#sns.distplot(train["Second_Floor_Area"])
#sns.distplot(train["LowQualFinSF"])
#sns.distplot(train["Grade_Living_Area"])
#sns.distplot(train["Garage_Area"])
#sns.distplot(train["W_Deck_Area"])
#sns.distplot(train["Open_Lobby_Area"])
#sns.distplot(train["Enclosed_Lobby_Area"])
#sns.distplot(train["Three_Season_Lobby_Area"])
#sns.distplot(train["Screen_Lobby_Area"])
sns.distplot(train["Sale_Price"])


# In[87]:


IQR=train["Lot_Size"].quantile(0.75)-train["Lot_Size"].quantile(0.25)
IQR


# In[88]:


up=train["Lot_Size"].quantile(0.75)+(3*IQR)
lw=train["Lot_Size"].quantile(0.25)-(3*IQR)
print(up,lw)


# In[89]:


train.Lot_Size.describe()


# In[90]:


prout.loc[prout["Lot_Size"]>23765.0,"Lot_Size"]=23765.0


# In[91]:


IQR2=train["BsmtFinSF1"].quantile(0.75)-train["BsmtFinSF1"].quantile(0.25)
IQR2


# In[92]:


up2=train["BsmtFinSF1"].quantile(0.75)+(3*IQR2)
lw2=train["BsmtFinSF1"].quantile(0.25)-(3*IQR2)
print(up2,lw2)


# In[93]:


train.BsmtFinSF1.describe()


# In[94]:


prout.loc[prout["BsmtFinSF1"]>2848.0,"BsmtFinSF1"]=2848.0


# In[95]:


up3=train["Lot_Extent"].mean()+3*train["Lot_Extent"].std()
lw3=train["Lot_Extent"].mean()-3*train["Lot_Extent"].std()
print(up3,lw3)


# In[96]:


train.Lot_Extent.describe()


# In[97]:


prout.loc[prout["Lot_Extent"]>136.13129541772906,"Lot_Extent"]=136.13129541772906


# In[98]:


up4=train["BsmtFinSF2"].mean()+3*train["BsmtFinSF2"].std()
lw4=train["BsmtFinSF2"].mean()-3*train["BsmtFinSF2"].std()
print(up4,lw4)


# In[99]:


train.BsmtFinSF2.describe()


# In[100]:


prout.loc[prout["BsmtFinSF2"]>530.0,"BsmtFinSF2"]=530.0


# In[101]:


IQR4=train["BsmtUnfSF"].quantile(0.75)-train["BsmtUnfSF"].quantile(0.25)
IQR4


# In[102]:


up4=train["BsmtUnfSF"].quantile(0.75)+(3*IQR4)
lw4=train["BsmtUnfSF"].quantile(0.25)-(3*IQR4)
print(up4,lw4)


# In[103]:


train.BsmtUnfSF.describe()


# In[104]:


prout.loc[prout["BsmtUnfSF"]>2561.5,"BsmtUnfSF"]=2561.5


# In[105]:


IQR5=train["Total_Basement_Area"].quantile(0.75)-train["Total_Basement_Area"].quantile(0.25)
IQR5


# In[106]:


up5=train["Total_Basement_Area"].quantile(0.75)+(3*IQR5)
lw5=train["Total_Basement_Area"].quantile(0.25)-(3*IQR5)
print(up5,lw5)


# In[107]:


train.Total_Basement_Area.describe()


# In[108]:


prout.loc[prout["Total_Basement_Area"]>2807.5,"Total_Basement_Area"]=2807.5


# In[109]:


up6=train["First_Floor_Area"].mean()+3*train["First_Floor_Area"].std()
lw6=train["First_Floor_Area"].mean()-3*train["First_Floor_Area"].std()
print(up6,lw6)


# In[110]:


train.First_Floor_Area.describe()


# In[111]:


prout.loc[prout["First_Floor_Area"]>2322.700373675161,"First_Floor_Area"]=2322.700373675161


# In[112]:


IQR7=train["Second_Floor_Area"].quantile(0.75)-train["Second_Floor_Area"].quantile(0.25)
IQR7


# In[113]:


up7=train["Second_Floor_Area"].quantile(0.75)+(1.5*IQR7)
lw7=train["Second_Floor_Area"].quantile(0.25)-(1.5*IQR7)
print(up7,lw7)


# In[114]:


train.Second_Floor_Area.describe()


# In[115]:


prout.loc[prout["First_Floor_Area"]>1820.0,"First_Floor_Area"]=1820.0


# In[116]:


up8=train["LowQualFinSF"].mean()+3*train["LowQualFinSF"].std()
lw8=train["LowQualFinSF"].mean()-3*train["LowQualFinSF"].std()
print(up8,lw8)


# In[117]:


train.LowQualFinSF.describe()


# In[118]:


prout.loc[prout["LowQualFinSF"]>151.76706285948669,"LowQualFinSF"]=151.76706285948669


# In[119]:


up9=train["Grade_Living_Area"].mean()+3*train["Grade_Living_Area"].std()
lw9=train["Grade_Living_Area"].mean()-3*train["Grade_Living_Area"].std()
print(up9,lw9)


# In[120]:


train.Grade_Living_Area.describe()


# In[121]:


prout.loc[prout["Grade_Living_Area"]>3092.4913553761457,"Grade_Living_Area"]=3092.4913553761457


# In[122]:


up10=train["Garage_Area"].mean()+3*train["Garage_Area"].std()
lw10=train["Garage_Area"].mean()-3*train["Garage_Area"].std()
print(up10,lw10)


# In[123]:


train.Garage_Area.describe()


# In[124]:


train.loc[train["Garage_Area"]>1102.9411493792454 ,"Garage_Area"]=1102.9411493792454 


# In[125]:


up11=train["W_Deck_Area"].mean()+3*train["W_Deck_Area"].std()
lw11=train["W_Deck_Area"].mean()-3*train["W_Deck_Area"].std()
print(up11,lw11)


# In[126]:


train.W_Deck_Area.describe()


# In[127]:


prout.loc[prout["W_Deck_Area"]>465.3790159660496 ,"W_Deck_Area"]=465.3790159660496 


# In[128]:


IQR12=train["Open_Lobby_Area"].quantile(0.75)-train["Open_Lobby_Area"].quantile(0.25)
IQR12


# In[129]:


up12=train["Open_Lobby_Area"].quantile(0.75)+(3*IQR12)
lw12=train["Open_Lobby_Area"].quantile(0.25)-(3*IQR12)
print(up12,lw12)


# In[130]:


train.Open_Lobby_Area.describe()


# In[131]:


prout.loc[prout["Open_Lobby_Area"]>379.5 ,"Open_Lobby_Area"]=379.5


# In[132]:


up13=train["Enclosed_Lobby_Area"].mean()+3*train["Enclosed_Lobby_Area"].std()
lw13=train["Enclosed_Lobby_Area"].mean()-3*train["Enclosed_Lobby_Area"].std()
print(up13,lw13)


# In[133]:


train.Enclosed_Lobby_Area.describe()


# In[134]:


train.loc[train["Enclosed_Lobby_Area"]>207.34045193428852 ,"Enclosed_Lobby_Area"]=207.34045193428852


# In[135]:


up14=train["Three_Season_Lobby_Area"].mean()+3*train["Three_Season_Lobby_Area"].std()
lw14=train["Three_Season_Lobby_Area"].mean()-3*train["Three_Season_Lobby_Area"].std()
print(up14,lw14)


# In[136]:


train.Three_Season_Lobby_Area.describe()


# In[137]:


prout.loc[prout["Three_Season_Lobby_Area"]>91.39366624305208,"Three_Season_Lobby_Area"]=91.39366624305208


# In[138]:


up15=train["Screen_Lobby_Area"].mean()+3*train["Screen_Lobby_Area"].std()
lw15=train["Screen_Lobby_Area"].mean()-3*train["Screen_Lobby_Area"].std()
print(up15,lw15)


# In[139]:


train.Screen_Lobby_Area.describe()


# In[140]:


prout.loc[prout["Screen_Lobby_Area"]>182.39669442924676 ,"Screen_Lobby_Area"]=182.39669442924676


# In[141]:


IQR16=train["Sale_Price"].quantile(0.75)-train["Sale_Price"].quantile(0.25)
IQR16


# In[142]:


up16=train["Sale_Price"].quantile(0.75)+(3*IQR16)
lw16=train["Sale_Price"].quantile(0.25)-(3*IQR16)
print(up16,lw16)


# In[143]:


train.Sale_Price.describe()


# In[144]:


prout.loc[prout["Sale_Price"]>466150.0 ,"Sale_Price"]=466150.0


# In[129]:


#Q1=train.quantile(0.25)
#Q3=train.quantile(0.75)
#IQR=Q3-Q1
#print(IQR)


# In[130]:


#pd.set_option("display.max_columns",None)
#train=train[((train<(Q1-1.5*IQR)) |(train>(Q3+1.5*IQR))).any(axis=1)]
#train


# #  DETECT OUTLIERS AND TREAT THEM OF TEST FILE

# In[145]:


#test._get_numeric_data().columns


# In[146]:


#test.Lot_Size.value_counts()


# In[147]:


#test.boxplot(column="Lot_Size")
#test.boxplot(column="BsmtFinSF1")
#test.boxplot(column="Lot_Extent")
#test.boxplot(column="BsmtFinSF2")
#test.boxplot(column="BsmtUnfSF")
#test.boxplot(column="Total_Basement_Area")
#test.boxplot(column="First_Floor_Area")
#test.boxplot(column="Second_Floor_Area")
#test.boxplot(column="LowQualFinSF")
#test.boxplot(column="Grade_Living_Area")
#test.boxplot(column="Garage_Area")
#test.boxplot(column="W_Deck_Area")
#test.boxplot(column="Open_Lobby_Area")
#test.boxplot(column="Enclosed_Lobby_Area")
#test.boxplot(column="Three_Season_Lobby_Area")
test.boxplot(column="Screen_Lobby_Area")


# In[148]:


#test.hist(["Lot_Size"],bins=20)
#test.hist(["BsmtFinSF1"])
#test.hist(["Lot_Extent"])
#test.hist(["BsmtFinSF2"])
#test.hist(["BsmtUnfSF"])
#test.hist(["Total_Basement_Area"])
#test.hist(["First_Floor_Area"])
#test.hist(["Second_Floor_Area"])
#test.hist(["LowQualFinSF"])
#test.hist(["Grade_Living_Area"])
#test.hist(["Garage_Area"])
#test.hist(["W_Deck_Area"])
#test.hist(["Open_Lobby_Area"],bins=50)
#test.hist(["Enclosed_Lobby_Area"],bins=50)
#test.hist(["Three_Season_Lobby_Area"],bins=50)
test.hist(["Screen_Lobby_Area"],bins=50)


# In[149]:


#sns.distplot(test["Lot_Size"])
#sns.distplot(test["BsmtFinSF1"])
#sns.distplot(test["Lot_Extent"])
#sns.distplot(test["BsmtFinSF2"])
#sns.distplot(test["BsmtUnfSF"])
#sns.distplot(test["Total_Basement_Area"])
#sns.distplot(test["First_Floor_Area"])
#sns.distplot(test["Second_Floor_Area"])
#sns.distplot(test["LowQualFinSF"])
#sns.distplot(test["Grade_Living_Area"])
#sns.distplot(test["Garage_Area"])
#sns.distplot(test["W_Deck_Area"])
#sns.distplot(test["Open_Lobby_Area"])
#sns.distplot(test["Enclosed_Lobby_Area"])
#sns.distplot(test["Three_Season_Lobby_Area"])
sns.distplot(test["Screen_Lobby_Area"])


# In[150]:


d2=test.copy()


# In[151]:


up_1=test["Lot_Size"].mean()+3*test["Lot_Size"].std()
lw_1=test["Lot_Size"].mean()-3*test["Lot_Size"].std()
print(up_1,lw_1)


# In[152]:


test.Lot_Size.describe()


# In[153]:


d2.loc[d2["Lot_Size"]>24752.74867073253,"Lot_Size"]=24752.74867073253


# In[154]:


IQR_2=test["BsmtFinSF1"].quantile(0.75)-test["BsmtFinSF1"].quantile(0.25)
IQR_2


# In[155]:


up_2=test["BsmtFinSF1"].quantile(0.75)+(3*IQR_2)
lw_2=test["BsmtFinSF1"].quantile(0.25)-(3*IQR_2)
print(up_2,lw_2)


# In[156]:


test.BsmtFinSF1.describe()


# In[157]:


d2.loc[d2["BsmtFinSF1"]>3008.0,"BsmtFinSF1"]=3008.0


# In[158]:


IQR_3=test["Lot_Extent"].quantile(0.75)-test["Lot_Extent"].quantile(0.25)
IQR_3


# In[159]:


up_3=test["Lot_Extent"].quantile(0.75)+(3*IQR_3)
lw_3=test["Lot_Extent"].quantile(0.25)-(3*IQR_3)
print(up_3,lw_3)


# In[160]:


test.Lot_Extent.describe()


# In[161]:


d2.loc[d2["Lot_Extent"]>132.0,"BsmtFinSF1"]=132.0


# In[162]:


up_4=train["BsmtFinSF2"].mean()+3*train["BsmtFinSF2"].std()
lw_4=train["BsmtFinSF2"].mean()-3*train["BsmtFinSF2"].std()
print(up_4,lw_4)


# In[163]:


test.BsmtFinSF2.describe()


# In[164]:


d2.loc[d2["BsmtFinSF2"]>430.0,"BsmtFinSF2"]=430.0


# In[165]:


IQR_5=test["BsmtUnfSF"].quantile(0.75)-test["BsmtUnfSF"].quantile(0.25)
IQR_5


# In[166]:


up_5=train["BsmtUnfSF"].quantile(0.75)+(2*IQR_5)
lw_5=train["BsmtUnfSF"].quantile(0.25)-(2*IQR_5)
print(up_5,lw_5)


# In[167]:


test.BsmtUnfSF.describe()


# In[168]:


d2.loc[d2["BsmtUnfSF"]>1958.0,"BsmtUnfSF"]=1958.0


# In[169]:


IQR_6=test["Total_Basement_Area"].quantile(0.75)-test["Total_Basement_Area"].quantile(0.25)
IQR_6


# In[170]:


up_6=train["Total_Basement_Area"].quantile(0.75)+(3*IQR_6)
lw_6=train["Total_Basement_Area"].quantile(0.25)-(3*IQR_6)
print(up_6,lw_6)


# In[171]:


test.Total_Basement_Area.describe()


# In[172]:


d2.loc[d2["Total_Basement_Area"]>2878.5,"Total_Basement_Area"]=2878.5


# In[173]:


IQR_7=test["First_Floor_Area"].quantile(0.75)-test["First_Floor_Area"].quantile(0.25)
IQR_7


# In[174]:


up_7=train["First_Floor_Area"].quantile(0.75)+(3*IQR_7)
lw_7=train["First_Floor_Area"].quantile(0.25)-(3*IQR_7)
print(up_7,lw_7)


# In[175]:


test.First_Floor_Area.describe()


# In[176]:


d2.loc[d2["First_Floor_Area"]>2949.0,"First_Floor_Area"]=2949.0


# In[177]:


IQR_8=test["Second_Floor_Area"].quantile(0.75)-test["Second_Floor_Area"].quantile(0.25)
IQR_8


# In[178]:


up_8=train["Second_Floor_Area"].quantile(0.75)+(1.5*IQR_8)
lw_8=train["Second_Floor_Area"].quantile(0.25)-(1.5*IQR_8)
print(up_8,lw_8)


# In[179]:


test.Second_Floor_Area.describe()


# In[180]:


d2.loc[d2["Second_Floor_Area"]>1703.0,"Second_Floor_Area"]=1703.0


# In[181]:


up_9=train["LowQualFinSF"].mean()+3*train["LowQualFinSF"].std()
lw_9=train["LowQualFinSF"].mean()-3*train["LowQualFinSF"].std()
print(up_9,lw_9)


# In[182]:


test.LowQualFinSF.describe()


# In[183]:


d2.loc[d2["LowQualFinSF"]>62.3047569372025,"LowQualFinSF"]=62.3047569372025


# In[184]:


IQR_10=test["Grade_Living_Area"].quantile(0.75)-test["Grade_Living_Area"].quantile(0.25)
IQR_10


# In[185]:


up_10=train["Grade_Living_Area"].quantile(0.75)+(3*IQR_10)
lw_10=train["Grade_Living_Area"].quantile(0.25)-(3*IQR_10)
print(up_10,lw_10)


# In[186]:


test.Grade_Living_Area.describe()


# In[187]:


d2.loc[d2["Grade_Living_Area"]>3588.0,"Grade_Living_Area"]=3588.0


# In[188]:


IQR_11=test["Garage_Area"].quantile(0.75)-test["Garage_Area"].quantile(0.25)
IQR_11


# In[189]:


up_11=train["Garage_Area"].quantile(0.75)+(3*IQR_11)
lw_11=train["Garage_Area"].quantile(0.25)-(3*IQR_11)
print(up_11,lw_11)


# In[190]:


test.Garage_Area.describe()


# In[191]:


d2.loc[d2["Garage_Area"]>1381.0,"Garage_Area"]=1381.0


# In[192]:


IQR_12=test["W_Deck_Area"].quantile(0.75)-test["W_Deck_Area"].quantile(0.25)
IQR_12


# In[193]:


up_12=train["W_Deck_Area"].quantile(0.75)+(3*IQR_12)
lw_12=train["W_Deck_Area"].quantile(0.25)-(3*IQR_12)
print(up_12,lw_12)


# In[194]:


test.W_Deck_Area.describe()


# In[195]:


d2.loc[d2["W_Deck_Area"]>684.0,"W_Deck_Area"]=684.0


# In[196]:


IQR_13=test["Open_Lobby_Area"].quantile(0.75)-test["Open_Lobby_Area"].quantile(0.25)
IQR_13


# In[197]:


up_13=train["Open_Lobby_Area"].quantile(0.75)+(3*IQR_13)
lw_13=train["Open_Lobby_Area"].quantile(0.25)-(3*IQR_13)
print(up_13,lw_13)


# In[198]:


test.Open_Lobby_Area.describe()


# In[199]:


d2.loc[d2["Open_Lobby_Area"]>312.0,"Open_Lobby_Area"]=312.0


# In[200]:


IQR_14=test["Enclosed_Lobby_Area"].quantile(0.75)-test["Enclosed_Lobby_Area"].quantile(0.25)
IQR_14


# In[201]:


up_14=train["Enclosed_Lobby_Area"].quantile(0.75)+(3*IQR_14)
lw_14=train["Enclosed_Lobby_Area"].quantile(0.25)-(3*IQR_14)
print(up_14,lw_14)


# In[202]:


test.Enclosed_Lobby_Area.describe()


# In[203]:


d2.loc[d2["Enclosed_Lobby_Area"]>66.0,"Enclosed_Lobby_Area"]=66.0


# In[204]:


up_15=train["Three_Season_Lobby_Area"].mean()+3*train["Three_Season_Lobby_Area"].std()
lw_15=train["Three_Season_Lobby_Area"].mean()-3*train["Three_Season_Lobby_Area"].std()
print(up_15,lw_15)


# In[205]:


test.Three_Season_Lobby_Area.describe()


# In[206]:


test.loc[test["Three_Season_Lobby_Area"]>37.97262395873686,"Three_Season_Lobby_Area"]=37.97262395873686


# In[207]:


up_16=train["Screen_Lobby_Area"].mean()+3*train["Screen_Lobby_Area"].std()
lw_16=train["Screen_Lobby_Area"].mean()-3*train["Screen_Lobby_Area"].std()
print(up_16,lw_16)


# In[208]:


test.Screen_Lobby_Area.describe()


# In[209]:


d2.loc[d2["Screen_Lobby_Area"]>153.45311264336374,"Screen_Lobby_Area"]=153.45311264336374


# In[210]:


#Q1=test.quantile(0.25)
#Q3=test.quantile(0.75)
#IQR_t=Q3-Q1
#print(IQR_t)


# In[195]:


#pd.set_option("display.max_columns",None)
#test=test[((test<(Q1-1.5*IQR_t)) |(test>(Q3+1.5*IQR_t))).any(axis=1)]
#test


# # SKEWNESS

# In[211]:


prout.skew()


# In[212]:


prout["Building_Class"]=np.sqrt(prout.Building_Class)
prout["Property_Slope"]=np.sqrt(prout.Property_Slope)
prout["Condition1"]=np.sqrt(prout.Condition1)
prout["Condition2"]=np.sqrt(prout.Condition2)
prout["House_Type"]=np.sqrt(prout.House_Type)
prout["Utility_Type"]=np.sqrt(prout.Utility_Type)
prout["Roof_Design"]=np.sqrt(prout.Roof_Design)  
prout["Roof_Quality"]=np.sqrt(prout.Roof_Quality)
prout["BsmtFinSF2"]=np.sqrt(prout.BsmtFinSF2)
prout["Heating_Type"]=np.sqrt(prout.Heating_Type)
prout["LowQualFinSF"]=np.sqrt(prout.LowQualFinSF)
prout["Underground_Half_Bathroom"]=np.sqrt(prout.Underground_Half_Bathroom)
prout["Kitchen_Above_Grade"]=np.sqrt(prout.Kitchen_Above_Grade)
prout["Three_Season_Lobby_Area"]=np.sqrt(prout.Three_Season_Lobby_Area)
prout["Screen_Lobby_Area"]=np.sqrt(prout.Screen_Lobby_Area)
prout["Pool_Area"]=np.sqrt(prout.Pool_Area)
prout["Sale_Price"]=np.sqrt(prout.Sale_Price)


# In[213]:


prout["Zoning_Class"]=np.square(prout.Zoning_Class)
prout["Road_Type"]=np.square(prout.Road_Type)
prout["Land_Outline"]=np.square(prout.Land_Outline)
prout["Lot_Configuration"]=np.square(prout.Lot_Configuration)
prout["Exterior_Material"]=np.square(prout.Exterior_Material)
prout["Exterior_Condition"]=np.square(prout.Exterior_Condition)
prout["Basement_Height"]=np.square(prout.Basement_Height)
prout["Basement_Condition"]=np.square(prout.Basement_Condition)
prout["BsmtFinType2"]=np.square(prout.BsmtFinType2)
prout["Air_Conditioning"]=np.square(prout.Air_Conditioning)
prout["Electrical_System"]=np.square(prout.Electrical_System)
prout["Kitchen_Quality"]=np.square(prout.Kitchen_Quality)
prout["Functional_Rate"]=np.square(prout.Functional_Rate)
prout["Garage_Quality"]=np.square(prout.Garage_Quality)
prout["Garage_Condition"]=np.square(prout.Garage_Condition)
prout["Pavedd_Drive"]=np.square(prout.Pavedd_Drive)
prout["Sale_Type"]=np.square(prout.Sale_Type)
prout["Sale_Condition"]=np.square(prout.Sale_Condition)


# In[214]:


d2.skew()


# In[216]:


d2["Property_Slope"]=np.sqrt(d2.Property_Slope)
d2["Condition1"]=np.sqrt(d2.Condition1)
d2["House_Type"]=np.sqrt(d2.House_Type)
d2["Roof_Design"]=np.sqrt(d2.Roof_Design)  
d2["Roof_Quality"]=np.sqrt(d2.Roof_Quality)
d2["BsmtFinSF1"]=np.sqrt(d2.BsmtFinSF1)
d2["BsmtFinSF2"]=np.sqrt(d2.BsmtFinSF2)
d2["Heating_Type"]=np.sqrt(d2.Heating_Type)
d2["Second_Floor_Area"]=np.sqrt(d2.Second_Floor_Area)
d2["LowQualFinSF"]=np.sqrt(d2.LowQualFinSF)
d2["Underground_Half_Bathroom"]=np.sqrt(d2.Underground_Half_Bathroom)
d2["Kitchen_Above_Grade"]=np.sqrt(d2.Kitchen_Above_Grade)
d2["W_Deck_Area"]=np.sqrt(d2.W_Deck_Area)
d2["Open_Lobby_Area"]=np.sqrt(d2.Open_Lobby_Area)
d2["Enclosed_Lobby_Area"]=np.sqrt(d2.Enclosed_Lobby_Area)
d2["Three_Season_Lobby_Area"]=np.sqrt(d2.Three_Season_Lobby_Area)
d2["Screen_Lobby_Area"]=np.sqrt(d2.Screen_Lobby_Area)
d2["Pool_Area"]=np.sqrt(d2.Pool_Area)


# In[217]:


d2["Zoning_Class"]=np.square(d2.Zoning_Class)
d2["Road_Type"]=np.square(d2.Road_Type)
d2["Land_Outline"]=np.square(d2.Land_Outline)
d2["Lot_Configuration"]=np.square(d2.Lot_Configuration)
d2["Condition2"]=np.square(d2.Condition2)
d2["Exterior_Material"]=np.square(d2.Exterior_Material)
d2["Exterior_Condition"]=np.square(d2.Exterior_Condition)  
d2["Basement_Height"]=np.square(d2.Basement_Height)
d2["Basement_Condition"]=np.square(d2.Basement_Condition)
d2["Exposure_Level"]=np.square(d2.Exposure_Level)
d2["BsmtFinType2"]=np.square(d2.LowQualFinSF)
d2["Air_Conditioning"]=np.square(d2.Air_Conditioning)
d2["Electrical_System"]=np.square(d2.Electrical_System)
d2["Kitchen_Quality"]=np.square(d2.Kitchen_Quality)
d2["Functional_Rate"]=np.square(d2.Functional_Rate)
d2["Garage_Quality"]=np.square(d2.Garage_Quality)
d2["Garage_Condition"]=np.square(d2.Garage_Condition)
d2["Pavedd_Drive"]=np.square(d2.Pavedd_Drive)
d2["Sale_Type"]=np.square(d2.Sale_Type)
d2["Sale_Condition"]=np.square(d2.Sale_Condition)


# In[218]:


d2.skew()


# # LINEAR REGRESSION 

# # LINEAR REGRESSOR

# In[219]:


tr_x=prout.iloc[:,1:-1]
tr_y=prout.iloc[:,-1]


# In[220]:


from sklearn.model_selection import train_test_split


# In[221]:


tr_xtrain,tr_xtest,tr_ytrain,tr_ytest=train_test_split(tr_x,tr_y,test_size=.3,random_state=101)


# In[222]:


tr_xtrain.shape,tr_xtest.shape,tr_ytrain.shape,tr_ytest.shape


# In[223]:


from sklearn import linear_model
lm=linear_model.LinearRegression()


# In[224]:


lm.fit(tr_xtrain,tr_ytrain)


# In[225]:


pred1=lm.predict(tr_xtest) 
pred1


# In[226]:


lm.coef_ 


# In[227]:


lm.intercept_


# In[228]:


r2_1=lm.score(tr_xtrain,tr_ytrain)
r2_1


# In[229]:


adj_r2_1=1-(((1-r2_1)*(580-1))/(580-5-1)) 
adj_r2_1   


# In[230]:


from sklearn import metrics


# In[231]:


error1=tr_ytest-pred1
error1


# In[232]:


aerror1=np.abs(error1)
aerror1


# In[233]:


mape_1=np.mean(aerror1/tr_ytest)*100
mape_1


# In[234]:


mae_1=metrics.mean_absolute_error(tr_ytest,pred1)
mae_1


# In[235]:


MSE_1=metrics.mean_squared_error(tr_ytest,pred1)
MSE_1


# # LASSO 

# In[237]:


from sklearn.linear_model import Lasso
lasso=Lasso()


# In[238]:


lasso.fit(tr_xtrain,tr_ytrain)


# In[239]:


pred2=lasso.predict(tr_xtest)
pred2


# In[240]:


r2_2=lasso.score(tr_xtrain,tr_ytrain)
r2_2


# In[241]:


adj_r2_2=1-(((1-r2_2)*(580-1))/(580-5-1)) 
adj_r2_2  


# In[242]:


error2=tr_ytest-pred2
error2


# In[243]:


aerror2=np.abs(error2)
aerror2


# In[244]:


mape_2=np.mean(aerror2/tr_ytest)*100
mape_2


# In[245]:


mae_2=metrics.mean_absolute_error(tr_ytest,pred2)
mae_2


# In[246]:


MSE_2=metrics.mean_squared_error(tr_ytest,pred2)
MSE_2


# # RIDGE 

# In[248]:


from sklearn.linear_model import Ridge
ridge=Ridge()


# In[249]:


ridge.fit(tr_xtrain,tr_ytrain)


# In[250]:


pred3=ridge.predict(tr_xtest)
pred3


# In[251]:


r2_3=ridge.score(tr_xtrain,tr_ytrain)
r2_3


# In[252]:


adj_r2_3=1-(((1-r2_3)*(580-1))/(580-5-1)) 
adj_r2_3  


# In[253]:


error3=tr_ytest-pred3
error3


# In[254]:


aerror3=np.abs(error3)
aerror3


# In[255]:


mape_3=np.mean(aerror3/tr_ytest)*100
mape_3


# In[256]:


mae_3=metrics.mean_absolute_error(tr_ytest,pred3)
mae_3


# In[257]:


MSE_3=metrics.mean_squared_error(tr_ytest,pred3)
MSE_3


# # ELASTIC-NET 

# In[259]:


from sklearn.linear_model import ElasticNet
enet=ElasticNet()


# In[260]:


enet.fit(tr_xtrain,tr_ytrain)


# In[261]:


pred4=enet.predict(tr_xtest)


# In[262]:


r2_4=enet.score(tr_xtrain,tr_ytrain)
r2_4


# In[263]:


adj_r2_4=1-(((1-r2_4)*(580-1))/(580-5-1)) 
adj_r2_4 


# In[264]:


error4=tr_ytest-pred4
error4


# In[265]:


aerror4=np.abs(error4)
aerror4


# In[266]:


mape_4=np.mean(aerror4/tr_ytest)*100
mape_4


# In[267]:


mae_4=metrics.mean_absolute_error(tr_ytest,pred4)
mae_4


# In[268]:


MSE_4=metrics.mean_squared_error(tr_ytest,pred4)
MSE_4


# # RANDOM FOREST REGRESSOR 

# In[269]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()


# In[270]:


rf.fit(tr_xtrain,tr_ytrain)


# In[271]:


pred5=rf.predict(tr_xtest)


# In[272]:


r2_5=rf.score(tr_xtrain,tr_ytrain)
r2_5


# In[273]:


adj_r2_5=1-(((1-r2_5)*(580-1))/(580-5-1)) 
adj_r2_5 


# In[274]:


error5=tr_ytest-pred5
error5


# In[275]:


aerror5=np.abs(error5)
aerror5


# In[276]:


mape_5=np.mean(aerror5/tr_ytest)*100
mape_5


# In[277]:


mae_5=metrics.mean_absolute_error(tr_ytest,pred5)
mae_5


# In[278]:


MSE_5=metrics.mean_squared_error(tr_ytest,pred5)
MSE_5

