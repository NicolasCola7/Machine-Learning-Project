import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn
from sklearn import linear_model, svm, metrics, model_selection
from scipy import stats
import statsmodels.api as sm

#1) CARICO E STAMPO I DATI
data = pd.read_csv("winequality-red.csv")
#print(data.head())
#print(data.info())

#2) PRE-PROCESSING

#controllo che non ci siano NaN, se ci sono li elimino 
#print(data.isnull().sum())

#Controllo che le variabili di tipo numerico non presentino dei valori fuori soglia
#print(data.describe())

#3) EDA

#visualizzo istogramma  del numero di  qualità
#sns.catplot(x='quality', data=data, kind='count')
#plt.show()

#matrice di correlazione
corr_matrix = data.corr()
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr_matrix, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True, fmt=".1f") 
plt.show()

#visualizzo la relazione di ogni feature con la qualità
#for col_name in data.columns[:-1]:
    #plot=plt.figure(figsize=(5,5))
   # sns.barplot(x='quality', y=col_name, data=data)
    
#OSS:
    #-All'aumentare della volatile-acidity la qualità diminuisce
        #-volatile acidity diminuisce all'aumentare del citric acid, dei solfati, e dell'alchool; aumenta all'aumentare del ph
    #-All'aumentare del citric-acid la qualità aumenta
        #citric-acid diminuisce all'aumentare del ph; aumenta all'aumentare di density, solfati e clorides
    #-All'aumentare dei chlorides la qualità diminuisce
        #-chlodides diminuiscono all'aumentare del alchool, pH; aumentano all'aumentare di density, e solfati
    #-All'aumentare del pH la qualità diminuisce pochissimo
    #_densità non influenza qualità
    #-All'aumentare dei solfati la qualità aumenta
    #-All'aumentare dell'achool ???
    #-
    
#4) SPLITTING

#separo dati 
data_final = data.copy()
data_final['quality'] = data['quality'].apply(lambda y_value: 1 if y_value >= 7 else 0)
#print(data_final.head())

data_train, data_test=model_selection.train_test_split(data_final, test_size=200, stratify=data_final[['quality']], random_state=3)
data_train, data_val  = model_selection.train_test_split(data_train, test_size = 200)
#print(data_train.shape, data_test.shape, data_val.shape)

X_val = data_val.drop('quality', axis=1)
Y_val = data_val[["quality"]]

X=data_train.drop('quality', axis=1)
y=data_train[['quality']]

#5) REGERESSIONE
#X = data_final['volatile acidity'].values.reshape(-1, 1)
#y = data_final['citric acid'].values.reshape(-1, 1)
#X = data_final['fixed acidity'].values.reshape(-1, 1)
#y = data_final['citric acid'].values.reshape(-1, 1)
#X = data_final['pH'].values.reshape(-1, 1)
#y = data_final['fixed acidity'].values.reshape(-1, 1)
#X = data_final['sulphates'].values.reshape(-1, 1)
#y = data_final['chlorides'].values.reshape(-1, 1)
X = data_final['density'].values.reshape(-1, 1)
y = data_final['fixed acidity'].values.reshape(-1, 1)

# Creazione del modello di regressione lineare
reg = linear_model.LinearRegression()

# Addestramento del modello
reg.fit(X, y)

# Predict the y-values using the trained model
y_pred = reg.predict(X)

# Stima dei coefficienti
coefficients = reg.coef_
intercept = reg.intercept_
print("Coefficiente angolare:", coefficients[0][0])
print("Intercetta:", intercept[0])

# Grafico dei punti e della retta
plt.scatter(X, y, color='blue', label='Dati')
plt.plot(X, reg.predict(X), color='red', label='Regressione Lineare')
plt.xlabel('Volatile Acidity')
plt.ylabel('Citric Acid')
plt.title('Regressione Lineare: Volatile Acidity vs Citric Acid')
plt.legend()
plt.show()

# Calcolo del coefficiente r^2
r_squared =metrics.r2_score(y, y_pred)
print("Coefficient R^2:", r_squared)

# Calcolo del valore di MSE
mse = metrics.mean_squared_error(y, y_pred)
print("Mean Squared Error:", mse)

# Analisi di normalità dei residui
residuals = y - y_pred
p_value = stats.shapiro(residuals)[1]
if p_value > 0.05:
    print("I residui seguono una distribuzione normale (p-value:", p_value, ")")
else:
    print("I residui non seguono una distribuzione normale (p-value:", p_value, ")")
fig = sm.qqplot(data, line='45')


#6)ADDESTRAMENTO DEL MODELLO

#definisco il modello
#model=linear_model.LinearRegression()
#model=svm.SVC(kernel="linear", C=1)

#addestro
#model.fit(X, y.values.ravel())

