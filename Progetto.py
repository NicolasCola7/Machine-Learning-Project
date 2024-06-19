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
print(data.head().T)
print()
print(data.info())
print()

#2) PRE-PROCESSING

#controllo che non ci siano NaN, se ci sono li elimino 
print(data.isnull().sum())
print()

#Controllo che le variabili di tipo numerico non presentino dei valori fuori soglia
print(data.describe().T)
print()
# boxplot
data.plot(kind='box', subplots=True, layout=(4,4), figsize=(20,15), sharex=False, sharey=False)
plt.show()

#3) EDA

#visualizzo istogramma  del numero di  qualità (Analisi Univariata)
sns.catplot(x='quality', data=data, kind='count')
plt.show()

#matrice di correlazione (Analisi Multivariata)
corr_matrix = data.corr()
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt=".1f")
plt.show()

#visualizzo la relazione di ogni feature con la qualità (Analisi Bivariata)
for col_name in data.columns[:-1]:
    plot=plt.figure(figsize=(5,5))
    sns.barplot(x='quality', y=col_name, data=data)
  
    #OSS:
        #-All'aumentare della volatile-acidity la qualità diminuisce
           
        #-All'aumentare del citric-acid la qualità aumenta
          
        #-All'aumentare dei chlorides la qualità diminuisce
           
        #-All'aumentare dei solfati la qualità aumenta
        
        #-All'aumentare dell'achool la qualità aumenta
        
    
#ridefinisco i valori nella colonna delle quality
data_final = data.copy()

data_final['quality'] = data['quality'].replace({
    8: 'buono',
    7: 'buono',
    6: 'medio buono',
    5: 'medio buono',
    4: 'non buono',
    3: 'non buono',
})


#trasformo le qualità in valori numerici
data_final['quality'] = data_final['quality'].replace({
    'buono': 0,
    'medio buono': 1,
    'non buono': 2
    })
print(data_final.head())
print()

#nuovo grafico delle qualità
sns.catplot(x='quality', data=data_final, kind='count')
plt.show()

#4) SPLITTING

np.random.seed(seed=42) 

data_train, data_test=model_selection.train_test_split(data_final, test_size= 200)
data_train, data_val  = model_selection.train_test_split(data_train, test_size = 200)
print(data_train.shape, data_test.shape, data_val.shape)
print()

X_val = data_val.drop('quality', axis=1)
Y_val = data_val[["quality"]]

X=data_train.drop('quality', axis=1)
y=data_train[['quality']]
'''
#5) REGERESSIONE

def linear_regression(X_reg, x_name, y_reg, y_name):
   
    # Creazione del modello di regressione lineare
    reg = linear_model.LinearRegression()
    
    if(X_reg.shape[1]>1):
       
        # Dividi i dati in set di addestramento e di test
        X_train_reg, X_test_reg, Y_train_reg, Y_test_reg = model_selection.train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
      
        reg.fit(X_train_reg, Y_train_reg)

        # Fai le previsioni sui dati di test
        y_pred_reg = reg.predict(X_test_reg)

        # Stima dei coefficienti
        coefficients = reg.coef_
        intercept = reg.intercept_
        print(f"Coefficiente angolare:{coefficients[0][0]:.4f}")
        print(f"Intercetta:{intercept[0]:.4f}")
         
        # Grafico dei punti e della retta
        plt.figure(figsize=(10, 6))
        plt.scatter(Y_test_reg, y_pred_reg, edgecolor='k', alpha=0.7)
        plt.plot([min(Y_test_reg), max(Y_test_reg)], [min(Y_test_reg), max(Y_test_reg)], color='red', linestyle='--')
        plt.xlabel('Valori Reali')
        plt.ylabel('Valori Predetti')
        plt.title('Regressione Lineare Multipla')
        plt.show()
         
        # Calcolo del coefficiente r^2
        r_squared =metrics.r2_score(Y_test_reg, y_pred_reg)
        print(f"Coefficiente R^2: {r_squared:.2f}")
         
         
         # Calcolo del valore di MSE
        mse = metrics.mean_squared_error(Y_test_reg, y_pred_reg)
        print(f"Mean Squared Error: {mse:.4f}")
     
        # Analisi di normalità dei residui
        residuals = Y_test_reg - y_pred_reg
        p_value = stats.shapiro(residuals)[1]
        if p_value > 0.05:
            print(f"I residui seguono una distribuzione normale (p-value: {p_value:.10f}", ")")
        else:
            print(f"I residui non seguono una distribuzione normale (p-value:{p_value:.10f}", ")")
             
        print()
         
        # Istogramma dei residui
        plt.figure(figsize=(10, 5))
        sns.histplot(residuals, kde=True)
        plt.xlabel('Residui')
        plt.ylabel('Frequenza')
        plt.title('Istogramma dei residui')
        plt.show()
     
        #QQ-Plot dei residui
        sm.qqplot(residuals, line='45')
        plt.title('QQ plot dei residui')
        plt.show()
    
    else:

        # Addestramento del modello
        reg.fit(X_reg, y_reg)
    
        # Predict the y-values using the trained model
        y_pred_reg = reg.predict(X_reg)
    
        # Stima dei coefficienti
        coefficients = reg.coef_
        intercept = reg.intercept_
        print(f"Coefficiente angolare:{coefficients[0][0]:.4f}")
        print(f"Intercetta:{intercept[0]:.4f}")
        
        # Grafico dei punti e della retta
        plt.scatter(X_reg, y_reg, color='blue', label='Dati')
        plt.plot(X_reg, y_pred_reg, color='red', label='Regressione Lineare')
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.title(f'Regressione Lineare: {x_name} vs {y_name}')
        plt.legend()
        plt.show()
        
        # Calcolo del coefficiente r^2
        r_squared =metrics.r2_score(y_reg, y_pred_reg)
        print(f"Coefficiente R^2: {r_squared:.2f}")
        
        
        # Calcolo del valore di MSE
        mse = metrics.mean_squared_error(y_reg, y_pred_reg)
        print(f"Mean Squared Error: {mse:.4f}")
        
        # Analisi di normalità dei residui
        residuals = y_reg - y_pred_reg
        p_value = stats.shapiro(residuals)[1]
        if p_value > 0.05:
            print(f"I residui seguono una distribuzione normale (p-value: {p_value:.10f}", ")")
        else:
            print(f"I residui non seguono una distribuzione normale (p-value:{p_value:.10f}", ")")
            
        print()
            
        # Istogramma dei residui
        plt.figure(figsize=(10, 5))
        sns.histplot(residuals, kde=True)
        plt.xlabel('Residui')
        plt.ylabel('Frequenza')
        plt.title('Istogramma dei residui')
        plt.show()
        
        #QQ-Plot dei residui
        sm.qqplot(residuals, line='45')
        plt.title('QQ plot dei residui')
        plt.show()

#prima coppia di variabili(Correlazione=0.7)
X_reg1 = data_final['fixed acidity'].values.reshape(-1, 1)
y_reg1 = data_final['citric acid'].values.reshape(-1, 1)
x_name1='fixed acidity'
y_name1='citric acid'

#seconda coppia di variabili (Correlazione=-0.7)
X_reg2 = data_final['fixed acidity'].values.reshape(-1, 1)
y_reg2 = data_final['pH'].values.reshape(-1, 1)
x_name2='fixed acidity'
y_name2='ph'


#BONUS:regressione lineare multipla
#X_reg3 = data_final[['fixed acidity', 'citric acid', 'chlorides', 'density', 'residual sugar', 'total sulfur dioxide', 'alcohol', 'sulphates', 'volatile acidity', 'free sulfur dioxide']]
#y_reg3 = data_final['pH'].values.reshape(-1, 1)
#x_name3='fixed acidity, citric acid, chlorides, density, residual sugar, total sulfur dioxide, alcohol, sulphates, volatile acidity, free sulfur dioxide'
#y_name3='pH'


linear_regression(X_reg1, x_name1, y_reg1, y_name1)
linear_regression(X_reg2, x_name2, y_reg2, y_name2)

#linear_regression(X_reg3, x_name3, y_reg3, y_name3)
'''

#6)ADDESTRAMENTO DEL MODELLO

#definisco il modello
#model=linear_model.LogisticRegression()
#model=svm.SVC(kernel="rbf", C=1, gamma=8)

#addestro
#model.fit(X, y.values.ravel())

#7)HYPERPARAMETER TUNING

kernels=["linear", "poly", "rbf", "sigmoid"]
for kernel in kernels:  
    
    print(f"KERNEL={kernel}")
    
    if (kernel=="linear"):
        
        for d in range(1,11):
            model=svm.SVC(kernel=kernel, C=d)
         
             #addestro il modello
            model.fit(X, y.values.ravel())
           
             #predizione
            y_pred = model.predict(X_val)
         
             #misclassificazione
            ME = np.sum(y_pred != Y_val["quality"])
            MR = ME/len(y_pred)
          
             #accuratezza
            Acc = 1 - MR
            print(f"Acc  C={d} : {Acc}.")
   
    else:
       
        for d in range(1,11):
            
           for c in range(1,11):
              
               if(kernel=="poly"):
                   
                     model=svm.SVC(kernel=kernel, degree=d, C=c)
                      
                      #addestro il modello
                     model.fit(X, y.values.ravel())
                        
                      #predizione
                     y_pred = model.predict(X_val)
                      
                      #misclassificazione
                     ME = np.sum(y_pred != Y_val["quality"])
                     MR = ME/len(y_pred)
                       
                      #accuratezza
                     Acc = 1 - MR
                     print(f"Acc degree= {d}, C={c} : {Acc}.")
                     
               else:
                   
                     model=svm.SVC(kernel=kernel, gamma=d, C=c)
                      
                      #addestro il modello
                     model.fit(X, y.values.ravel())
                        
                      #predizione
                     y_pred = model.predict(X_val)
                      
                      #misclassificazione
                     ME = np.sum(y_pred != Y_val["quality"])
                     MR = ME/len(y_pred)
                       
                      #accuratezza
                     Acc = 1 - MR
                     print(f"Acc gamma= {d}, C={c} : {Acc}.")


#OSS: che con rbf ho sempre accuracy maggiore.
#In particolare, in fase di HT, rbf con gamma>3 e C qualsiasi ottengo sempre 0,845
# lo stesso in fase di valutazione      
'''
#8) VALUTAZIONE DELLA PERFORMANCE

X_test = data_test.drop('quality', axis=1)
Y_test = data_test[["quality"]]

#Predizione sul test-set
y_pred_test = model.predict(X_test)
print(f"Predizione sul test set : {y_pred_test}")
print()
  
# Calcolo delle metriche:

#errore di misclassificazione
ME = np.sum(y_pred_test != Y_test["quality"])

#tasso di misclassificazione
MR = ME/len(y_pred_test)

#accuratezza
Acc = 1 - MR

# Stampa delle metriche
print(f"ME sul test: {ME}.")
print(f"MR sul test: {MR}.")
print(f"Accuratezza sul test: {Acc}.")
print()

#matrice di confusione
confusion_matrix =metrics.confusion_matrix(Y_test, y_pred_test)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Reds", cbar=False)
plt.title("Matrice di Confusione")
plt.xlabel("Predetto")
plt.ylabel("Vero")
plt.show()

#9)STUDIO STATISTICO SUI RISULTATI DELLA VALUTAZIONE

# Numero di iterazioni
k = 50

# Liste per memorizzare i risultati
accuracies = []

misclassification_rates = []

misclassification_errors= []

# Esecuzione del modello k volte
for i in range(k):
   
    # Split dei dati
    data_train, data_test = model_selection.train_test_split(data_final, test_size=200)
    data_train, data_val = model_selection.train_test_split(data_train, test_size=200)

    
    X = data_train.drop('quality', axis=1)
    y = data_train[['quality']]
    
    X_test = data_test.drop('quality', axis=1)
    Y_test = data_test[["quality"]]

    # Definizione e addestramento del modello
    model = svm.SVC(kernel="rbf", C=1, gamma=8)
    model.fit(X, y.values.ravel())

    # Predizione
    y_pred_test = model.predict(X_test)
    #print(f"Predizione sul test set {i} : {y_pred_test}")
    #print()
    
    #errore di misclassificazione
    ME = np.sum(y_pred_test != Y_test["quality"])

    #tasso di misclassificazione
    MR = ME/len(y_pred_test)

    #accuracy
    accuracy = 1 - MR
   
    # Memorizzazione dei risultati
    accuracies.append(accuracy)
    misclassification_rates.append(MR)
    misclassification_errors.append(ME)

  
    #Creazione di un DataFrame per le metriche
    metrics= pd.DataFrame({
          'Accuratezza': accuracies,
          'Misclassification Rate': misclassification_rates,
          'Errori di misclassificazione': misclassification_errors
          })
  

# Analisi statistica descrittiva
def descriptive_stats(data, metric_name):
    
    #definisco le statistiche descrittive 
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)  
    minimum = np.min(data)
    first_quartile = np.percentile(data, 25)
    median = np.median(data)
    third_quartile = np.percentile(data, 75)
    maximum= np.max(data)
    interquartile_range=third_quartile-first_quartile
    
    #intervallo di confidenza
    conf_interval = stats.norm.interval(0.95, loc=mean, scale=std_dev / np.sqrt(k))
   
    #stampo statistiche descrittive
    print(f"{metric_name}:")
    print(f"Media: {mean:.4f}")
    print(f"Deviazione Standard: {std_dev:.4f}")
    print(f"Min: {minimum:.4f}")
    print(f"Primo quartile: {first_quartile:.4f}")
    print(f"Mediana (Secondo Quartile):{median:.4f}")
    print(f"Terzo quartile: {third_quartile:.4f}")
    print(f"Max: {maximum:.4f}")
    print(f"Range Interquartile: {interquartile_range:.2f}")
    print(f"Intervallo di confidenza  della media al 95%: {conf_interval}")
    print()

    # Istogramma
    plt.figure(figsize=(5, 5))
    sns.histplot(data, kde=True)
    plt.xlabel(metric_name)
    plt.ylabel('Frequenza')
    plt.title(f'Istogramma  {metric_name}')
    plt.show()

    # Boxplot
    plt.figure(figsize=(5, 5))
    sns.boxplot(data)
    plt.axhline(mean, color='red', linestyle='--')
    plt.title(f'Boxplot {metric_name}')
    plt.show()

for column in metrics.columns:
    descriptive_stats(metrics[column], column)
    
print(metrics)
'''