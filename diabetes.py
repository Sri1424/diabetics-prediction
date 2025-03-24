import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

#ignore the harmless warnings
import warnings
warnings.filterwarnings('ignore')

#set to display all cols in dataset
pd.set_option('display.max_columns',None)
dataset=pd.read_csv("C:/Users/yashu/OneDrive/Desktop/ml/diabetes.csv",header=0)
dataset.head(56)
# Compute the correlation matrix
correlation_matrix = dataset.corr()

# Create a heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(correlation_matrix, annot=True, cmap="Blues", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()
dataset.info()

dataset.describe()
dataset[dataset.duplicated(keep='first')]
dataset.isnull().sum()

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
cols=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
for col in cols:
  dataset[col]=le.fit_transform(dataset[col])

dataset.head(58)
X=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]
from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test=tts(X,y,test_size=0.2,random_state=42)

x_train.shape,x_test.shape,y_train.shape,y_test.shape
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))

x_train=sc.fit_transform(x_train)

x_test=sc.fit_transform(x_test)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Create objects of classification algorithms with default hyper-parameters

ModelLR = LogisticRegression()
ModelDC = DecisionTreeClassifier()
ModelKNN = KNeighborsClassifier(n_neighbors=14)
ModelGNB = GaussianNB()

MM = [ModelLR, ModelDC, ModelKNN, ModelGNB]

for models in MM:

    # Train the model training dataset

    models.fit(x_train, y_train)

    # Prediction the model with test dataset

    y_pred = models.predict(x_test)

    # Print the model name

    print('Model Name: ', models)

    # confusion matrix in sklearn

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score

    # actual values

    actual = y_test

    # predicted values

    predicted = y_pred

    # confusion matrix

    matrix = confusion_matrix(actual,predicted, labels=[1,0],sample_weight=None, normalize=None)
    print('Confusion matrix : \n', matrix)



    # classification report for precision, recall f1-score and accuracy

    C_Report = classification_report(actual,predicted,labels=[1,0])

    print('Classification report : \n', C_Report)

    #accuracy score
    ac_score=accuracy_score(actual,predicted)
    print("Accuracy of the model: ",ac_score)
    print("<========================================================>")
