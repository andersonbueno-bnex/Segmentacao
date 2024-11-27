import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar a planilha
sheet_data = pd.read_excel(r'C:\Users\Lenovo\Documents\Esteira de Projetos - Amigão\Base Amigão_Aldo\amostra_v1.xlsx')  # Substitua pelo caminho da sua planilha

# Overview of basic statistics to understand distributions
summary_stats = sheet_data.describe()

# Função para plotar histogramas para variáveis contínuas
def plot_histograms(data, columns):
    for column in columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(data[column], kde=True)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

# Função para plotar boxplots para detecção de outliers
def plot_boxplots(data, columns):
    for column in columns:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=data[column])
        plt.title(f'Boxplot of {column} - Outliers Detection')
        plt.xlabel(column)
        plt.show()

# Colunas contínuas para análise de distribuições e outliers
continuous_columns = ['frequencia', 'valortotal', 'quantidade', 'ticket_medio', 'preco_medio_produto']

# Plotando os gráficos
plot_histograms(sheet_data, continuous_columns)
plot_boxplots(sheet_data, continuous_columns)

# Exibindo as estatísticas descritivas
import ace_tools as tools; tools.display_dataframe_to_user(name="Resumo Estatístico dos Dados", dataframe=summary_stats)


# Certifique-se de ter os pacotes instalados, caso contrário, instale usando:
# !pip install matplotlib seaborn pandas

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Vamos supor que `sheet_data` já seja seu DataFrame; caso contrário, importe os dados
# Exemplo de carregamento:
# sheet_data = pd.read_csv("seu_arquivo.csv")

# Overview of basic statistics to understand distributions
summary_stats = sheet_data.describe()

# Plot histograms for continuous variables
def plot_histograms(data, columns):
    for column in columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(data[column], kde=True)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

# Boxplots to check for outliers
def plot_boxplots(data, columns):
    for column in columns:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=data[column])
        plt.title(f'Boxplot of {column} - Outliers Detection')
        plt.xlabel(column)
        plt.show()

# Columns to analyze for distributions and outliers
continuous_columns = ['frequencia', 'valortotal', 'quantidade', 'ticket_medio', 'preco_medio_produto']

# Plotting
plot_histograms(sheet_data, continuous_columns)
plot_boxplots(sheet_data, continuous_columns)

# Return summary statistics for numerical insights
print("Resumo Estatístico dos Dados:")
print(summary_stats)




