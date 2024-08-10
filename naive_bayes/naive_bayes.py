import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from yellowbrick.classifier import ConfusionMatrix

with open('../Employees.pkl','rb') as f:
    X_treinamento,y_treinamento,X_teste,y_teste,df=pickle.load(f)

# ============= APLICAÇÃO DO ALGORITMO =============
naive_employees=GaussianNB()
naive_employees.fit(X_treinamento,y_treinamento)
previsao=naive_employees.predict(X_teste)
y_teste

accuracy_score(y_teste,previsao) #41%
cm=ConfusionMatrix(naive_employees)
cm.fit(X_treinamento,y_treinamento)
cm.score(X_teste,y_teste)