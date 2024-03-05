<<<<<<< HEAD
# Importar las bibliotecas necesarias
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

# Cargar los datos
df = pd.read_csv('salaries.csv')

# Análisis de valores nulos
print("Valores nulos por columna:")
print(df.isnull().sum())

# Rellenar valores nulos, si los hay, por ejemplo, con la media o mediana
df.fillna(df.median(), inplace=True)

# Normalización de los datos (Ejemplo con la columna 'salary')
scaler = StandardScaler()
df['salary_normalized'] = scaler.fit_transform(df[['salary']])

# Análisis de correlación
correlation_matrix = df.corr()

# Visualización de la matriz de correlación en un heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f")
plt.title("Correlación entre características")
plt.show()

# Preparación de los datos para el modelo
# Asumiendo que queremos predecir 'salary' basado en otras características, excluyendo 'salary_normalized'
X = df.drop(['salary', 'salary_normalized'], axis=1)
# Convertir categorías a variables dummy
X = pd.get_dummies(X)
y = df['salary_normalized']

# División en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Comprobación de las dimensiones
print("Dimensiones de X_train:", X_train.shape)
print("Dimensiones de X_test:", X_test.shape)
print("Dimensiones de y_train:", y_train.shape)
print("Dimensiones de y_test:", y_test.shape)

# Preparación para un modelo de red neuronal
# Este es un ejemplo simple usando MLPRegressor de Scikit-learn
model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)

# Entrenamiento del modelo
model.fit(X_train, y_train)

# Evaluación del modelo (esto es solo un ejemplo, en la práctica deberías usar métricas de evaluación)
score = model.score(X_test, y_test)
print("Score del modelo:", score)
=======
# Importar las bibliotecas necesarias
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

# Cargar los datos
df = pd.read_csv('salaries.csv')

# Análisis de valores nulos
print("Valores nulos por columna:")
print(df.isnull().sum())

# Rellenar valores nulos, si los hay, por ejemplo, con la media o mediana
df.fillna(df.median(), inplace=True)

# Normalización de los datos (Ejemplo con la columna 'salary')
scaler = StandardScaler()
df['salary_normalized'] = scaler.fit_transform(df[['salary']])

# Análisis de correlación
correlation_matrix = df.corr()

# Visualización de la matriz de correlación en un heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f")
plt.title("Correlación entre características")
plt.show()

# Preparación de los datos para el modelo
# Asumiendo que queremos predecir 'salary' basado en otras características, excluyendo 'salary_normalized'
X = df.drop(['salary', 'salary_normalized'], axis=1)
# Convertir categorías a variables dummy
X = pd.get_dummies(X)
y = df['salary_normalized']

# División en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Comprobación de las dimensiones
print("Dimensiones de X_train:", X_train.shape)
print("Dimensiones de X_test:", X_test.shape)
print("Dimensiones de y_train:", y_train.shape)
print("Dimensiones de y_test:", y_test.shape)

# Preparación para un modelo de red neuronal
# Este es un ejemplo simple usando MLPRegressor de Scikit-learn
model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)

# Entrenamiento del modelo
model.fit(X_train, y_train)

# Evaluación del modelo (esto es solo un ejemplo, en la práctica deberías usar métricas de evaluación)
score = model.score(X_test, y_test)
print("Score del modelo:", score)
>>>>>>> bbf45d6a91f3e92eadcd36ce431b6d0e2687bdc2
