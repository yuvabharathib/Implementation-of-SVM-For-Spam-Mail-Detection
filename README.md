## Implementation-of-SVM-For-Spam-Mail-Detection
## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook
Algorithm
Import the required packages.
Import the dataset to operate on.
Split the dataset.
Predict the required output.
End the program.
## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: Yuvabharathi.B
RegisterNumber:  212222230181
import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result
import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
### Result Output
![image](https://github.com/yuvabharathib/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113497404/c62349da-83f1-45d4-aa18-a2139bf13ff4)
### data.head( )
![image](https://github.com/yuvabharathib/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113497404/3187045a-5903-48d9-8a34-7786ddb61c3b)
### data.info( )
![image](https://github.com/yuvabharathib/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113497404/5964295e-6e51-4746-8d8a-7ce6f65167b6)
### data.isnull().sum()
![image](https://github.com/yuvabharathib/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113497404/867170ed-6175-44e0-94fd-6248fe36e42b)
### Y_prediction
![image](https://github.com/yuvabharathib/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113497404/3aff344c-48e8-409b-9923-cd2dd4a6f360)
### Accuracy Value
![image](https://github.com/yuvabharathib/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113497404/58e33308-16fe-499b-b14b-2df7dce14685)
## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming
