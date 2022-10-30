import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
data=pd.read_csv('brain_stroke.csv')
print(data.columns)
print(data.dtypes)
print(data.ever_married)
print(data.smoking_status.value_counts())
print(data.work_type.value_counts())
print(data.Residence_type.value_counts())
print(data.stroke)
print(data.shape)
from sklearn.preprocessing import LabelEncoder
labe=LabelEncoder()
data['Smoking_Status']=labe.fit_transform(data['smoking_status'])
X=data[['hypertension','heart_disease','avg_glucose_level','bmi','Smoking_Status']]
y=data['stroke']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y)
import keras.activations,keras.metrics,keras.losses
from keras.models import  Sequential
from keras.layers import Dense
model=Sequential()
model.add(Dense(units=X.shape[1],input_dim=x_train.shape[1],activation=keras.activations.sigmoid))
model.add(Dense(units=X.shape[1],activation=keras.activations.relu))
model.add(Dense(units=X.shape[1],activation=keras.activations.relu))
model.add(Dense(units=X.shape[1],activation=keras.activations.sigmoid))
model.add(Dense(units=1,activation=keras.activations.sigmoid))
model.compile(optimizer='adam',metrics='accuracy',loss=keras.losses.binary_crossentropy)
model.fit(x_train,y_train,batch_size=20,epochs=30)
pred=model.predict(x_test)
print(pred)
print('-----------------------------------------------')
print('---------------------------------------------------')
risk=pd.read_csv('Maternal_Risk.csv')
print(risk.columns)
print(risk.shape)
print(risk.isna().sum())
print(risk.dtypes)
sysbp=risk[['SystolicBP','DiastolicBP','BS','BodyTemp','HeartRate']]
from sklearn.preprocessing import LabelEncoder
lab=LabelEncoder()
risk['risklevel']=lab.fit_transform(risk['RiskLevel'])
from sklearn.model_selection import train_test_split
x_trin,x_tst,y_train,y_tst=train_test_split(risk[['SystolicBP','DiastolicBP','BS','BodyTemp','HeartRate']],risk['risklevel'])
import keras.activations,keras.losses
from keras.models import Sequential
from keras.layers import Dense
model=Sequential()
model.add(Dense(input_dim=risk[['SystolicBP','DiastolicBP','BS','BodyTemp','HeartRate']].shape[1],units=x_trin.shape[1],activation=keras.activations.sigmoid))
model.add(Dense(input_dim=risk[['SystolicBP','DiastolicBP','BS','BodyTemp','HeartRate']].shape[1],units=x_trin.shape[1],activation=keras.activations.sigmoid))
model.add(Dense(input_dim=risk[['SystolicBP','DiastolicBP','BS','BodyTemp','HeartRate']].shape[1],units=x_trin.shape[1],activation=keras.activations.sigmoid))
model.add(Dense(input_dim=risk[['SystolicBP','DiastolicBP','BS','BodyTemp','HeartRate']].shape[1],units=x_trin.shape[1],activation=keras.activations.sigmoid))
model.add(Dense(units=1,activation=keras.activations.sigmoid))
model.compile(optimizer='adam',loss=keras.losses.binary_crossentropy,metrics='accuracy')
model.fit(x_trin,y_train,batch_size=20,epochs=30)
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_trin,y_train)
pr=nb.predict(x_tst)
from sklearn.metrics import classification_report,confusion_matrix
cm=confusion_matrix(y_tst,pr)
print(cm)
crp=classification_report(y_tst,pr)
print(crp)
sn.pairplot(sysbp)
plt.show()
plt.plot(risk['SystolicBP'])
plt.plot(risk['DiastolicBP'])
plt.show()
sn.barplot(risk['BodyTemp'],risk['HeartRate'])
plt.show()
sn.barplot(risk['SystolicBP'],risk['DiastolicBP'])
plt.show()