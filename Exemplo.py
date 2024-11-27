import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados
df = pd.read_excel(r'C:\Users\Lenovo\Documents\Esteira de Projetos - Amigão\Base Amigão_Aldo\amostra_v1.xlsx')

# Ver as primeiras linhas e tipos de dados
print("Primeiras linhas do dataset:")
print(df.head())
print("\nInformações do dataset:")
print(df.info())

# Já existe uma coluna de 'ticket_medio', então não precisamos criar novamente
# Criar uma variável de frequência média de compra
df['frequencia_compra'] = df['frequencia'] / df['quantidade']

# Criar uma variável de sensibilidade a promoções
df['percentual_compras_promocao'] = df.get('compras_em_promocao', df['quantidade'] * 0.1) / df['quantidade']

# Calcular o desvio padrão do valor das compras
df['desvio_valor_compras'] = df.groupby('idcliente')['valortotal'].transform(lambda x: x.std() if len(x) > 1 else 0)

# Verificar novamente se há valores faltantes após o cálculo
print("Verificação de valores faltantes após cálculo de desvio padrão:")
print(df.isna().sum())

# Preencher quaisquer valores faltantes remanescentes com 0
df.fillna(0, inplace=True)

# Adicionar uma coluna de renda estimada fictícia se ela não existir
if 'renda_estimada' not in df.columns:
    np.random.seed(42)  # Para resultados reprodutíveis
    df['renda_estimada'] = np.random.normal(loc=5000, scale=1500, size=len(df))

# Padronizar as variáveis para clustering
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['ticket_medio', 'frequencia_compra', 'percentual_compras_promocao', 'desvio_valor_compras']])

# Aplicar K-Means para agrupar os shoppers
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(df_scaled)

# Visualizar os clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_scaled[:, 0], y=df_scaled[:, 1], hue=df['cluster'], palette='Set2')
plt.title('Clusters de Shoppers Baseados no Comportamento')
plt.xlabel('Ticket Médio (padronizado)')
plt.ylabel('Frequência de Compra (padronizado)')
plt.show()

# Variáveis preditoras e alvo
X = df[['ticket_medio', 'frequencia_compra', 'percentual_compras_promocao', 'desvio_valor_compras']]
y = df['renda_estimada']  # Usando a coluna fictícia 'renda_estimada' criada

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinamento do modelo RandomForest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predição e avaliação do modelo
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'\nMean Squared Error: {mse}')

# Visualizar os resultados da predição
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('Renda Real')
plt.ylabel('Renda Predita')
plt.title('Renda Real vs. Predita')
plt.show()

