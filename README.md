## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df = pd.read_csv("encoding.csv")
df
#Ordinal encoder
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm = ['Hot','Warm','Cold']
e1 = OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/f0c8d753-ebe6-4c16-8eff-f935db3eaacf)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/ad058bec-fa21-45ec-9d69-9a9086e618be)
```
#Label Encoder
import pandas as pd
df = pd.read_csv("encoding.csv")
df
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/4f04db29-6d63-4a61-949e-d5420e891757)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/e33088e2-5c15-4ed5-bfb3-1db537805480)
```
import pandas as pd
df = pd.read_csv("encoding.csv")
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/4372bd1c-24a9-429b-8857-5fe7a8d8758e)
```
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/2b577d05-7f94-4472-ba5f-9fcb9cfceeac)
```
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```
![image](https://github.com/user-attachments/assets/ebdf33bc-3f13-4b49-bf5e-da8596bfd249)
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![image](https://github.com/user-attachments/assets/d3f1f225-c9ac-4e7e-833b-df6d4a5432b0)
```
from category_encoders import TargetEncoder
import pandas as pd
df = pd.read_csv("data.csv")
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![image](https://github.com/user-attachments/assets/749ab808-95fb-40a5-a2a2-a86e4bc18957)
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("transform.csv")
df
```
![image](https://github.com/user-attachments/assets/8a2ade56-e5ed-4ba3-aa24-619f1408c864)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/fbcfe461-16cc-4446-b573-957c010e2b5a)
```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/d7ccabc3-c10c-4fa1-b9b6-1161635dc67e)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/6451ad99-27b4-4867-a37d-1728f82b53d0)
```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/4a6050d5-8d81-4927-8c83-925370e78d34)
```
df["Highly Positive Skew"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/505a8ce3-6fb0-4c41-b007-5a3f0ce0851f)
```
df["Moderate Positive Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Positive Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/1e34df74-e3f9-464b-a254-45c467b65608)
```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/030a7b48-7163-4710-94d7-8f1866a43022)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Positive Skew-1"]=qt.fit_transform(df[["Moderate Positive Skew"]])
df
```
![image](https://github.com/user-attachments/assets/82ae7c06-a63e-40b2-8eaa-ba28ee148993)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/4692e245-8e05-40ab-8bd1-bfdaf3a623ef)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/77c98e87-94f9-4ffe-946d-135e4a19498f)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
```
![image](https://github.com/user-attachments/assets/6d05f247-6c35-44f5-89f8-f7ac2f2ed1f5)
```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/3e87f63f-9c7d-45aa-9619-02f47e7f6d29)
```
sm.qqplot(df['Highly Negative Skew_1'],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/2e5d2a5f-e619-4242-9c38-dbc1f0677f03)
```
dt=pd.read_csv("titanic.csv")
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/0eb23d99-dea6-42b6-a4e2-d2caab7bfbdd)
```
sm.qqplot(dt['Age'],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/5ee9ed58-9786-41e7-953b-74da2f5aaaf1)

# RESULT:
       # INCLUDE YOUR RESULT HERE

       
