import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pickle


# ======== PRÉ PROCESSAMENTO ========
df= pd.read_excel('Employees.xlsx')
df.isnull().sum()
coluna_y=df['Country']
df.drop(['No',	'First Name',	'Last Name', 	'Monthly Salary','Sick Leaves',	'Unpaid Leaves',	'Overtime Hours','Country'],axis=1,inplace=True)
df['Start Date']=df['Start Date'].astype('object')
df['Country']=coluna_y
df.info()
df.describe()

# ======== VISUALIZAÇÃO ========
sns.countplot(x=df['Gender'],palette='dark')
sns.countplot(x=df['Job Rate'],palette='dark')

# ======== DIVISÃO ENTRE PREVISORES E CLASSE ========
X=df.iloc[:,0:-1].values
y=df.iloc[:,-1].values

# ======== TRATAMENTO DE ATRIBUTOS CATEGORICOS ========
label_encoder=LabelEncoder()
indices=[]

for i in range(X.shape[1]):

    #Comente esse código caso a acurácia seja menor.
    if df.dtypes[i] == 'object':
        X[:,i]=label_encoder.fit_transform(X[:,i])

    if df.dtypes[i] == 'object':
        indices.append(i)

one_hot_encoder=ColumnTransformer(transformers=[('OneHot',OneHotEncoder(),indices)],remainder='passthrough')
X=one_hot_encoder.fit_transform(X).toarray()

# ======== ESCALONAMENTO DE VALORES ========
X_treinamento,X_teste,y_treinamento,y_teste=train_test_split(X,y,test_size=0.3,random_state=0)

# ======== SALVANDO VARIÁVEIS EM DISCO ========
with open('Employees.pkl',mode='wb') as f:
    pickle.dump([X_treinamento,y_treinamento,X_teste,y_teste,df],f)
