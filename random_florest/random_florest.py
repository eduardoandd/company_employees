import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

with open('../Employees.pkl','rb') as f:
    X_treinamento,y_treinamento,X_teste,y_teste,df=pickle.load(f)

random_florest= RandomForestClassifier(n_estimators=300,criterion='entropy',random_state=0)
random_florest.fit(X_treinamento,y_treinamento)
previsoes=random_florest.predict(X_teste)
y_teste
accuracy_score(y_teste,previsoes) #62