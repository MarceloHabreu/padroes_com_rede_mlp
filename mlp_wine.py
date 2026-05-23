# =============================================================================
# TDE - FUNDAMENTOS DE INTELIGÊNCIA ARTIFICIAL
# UNIFACEMA - Análise e Desenvolvimento de Sistemas
# Disciplina: Fundamentos de Inteligência Artificial
# Atividade: Classificação de Vinhos com MLP (Multi-Layer Perceptron)
# Dataset: Wine Dataset - UCI Machine Learning Repository
# =============================================================================

# =============================================================================
# SEÇÃO 1 - IMPORTAÇÃO DAS BIBLIOTECAS
# =============================================================================

import numpy as np                          # Operações numéricas e arrays
import pandas as pd                         # Manipulação de dados em tabelas
import matplotlib.pyplot as plt             # Geração de gráficos
import seaborn as sns                       # Gráficos estatísticos (heatmap, etc.)
import warnings
warnings.filterwarnings('ignore')           # Suprime avisos desnecessários

from sklearn.neural_network import MLPClassifier        # A rede neural MLP
from sklearn.preprocessing import StandardScaler        # Normalização dos dados
from sklearn.model_selection import train_test_split    # Divisão treino/teste
from sklearn.metrics import (
    accuracy_score,         # Percentual de acerto
    confusion_matrix,       # Matriz de confusão
    classification_report   # Relatório detalhado por classe
)

# =============================================================================
# SEÇÃO 2 - CARREGAMENTO E EXPLORAÇÃO DO DATASET
# =============================================================================

# Nomes das 13 colunas de atributos + coluna de classe
# A primeira coluna do arquivo é a classe (1, 2 ou 3)
colunas = [
    'classe',
    'alcool',
    'acido_malico',
    'cinzas',
    'alcalinidade_cinzas',
    'magnesio',
    'fenois_totais',
    'flavanoides',
    'fenois_nao_flavanoides',
    'proantocianinas',
    'intensidade_cor',
    'matiz',
    'od280_od315',
    'prolina'
]

# Carrega o arquivo CSV sem cabeçalho (o dataset não possui linha de header)
df = pd.read_csv('wine.data', header=None, names=colunas)

print("=" * 60)
print("DATASET WINE - VISÃO GERAL")
print("=" * 60)
print(f"\nTotal de amostras: {len(df)}")
print(f"Total de atributos: {len(df.columns) - 1}")  # -1 pois a classe não é atributo de entrada
print(f"\nDistribuição das classes:")
print(df['classe'].value_counts().sort_index())
print(f"\nPrimeiras 5 linhas do dataset:")
print(df.head())
print(f"\nEstatísticas descritivas dos atributos:")
print(df.describe().round(2))

# =============================================================================
# SEÇÃO 3 - SEPARAÇÃO DE FEATURES (X) E RÓTULOS (y)
# =============================================================================

# X = entradas da rede (os 13 atributos químicos)
X = df.drop(columns=['classe']).values   # Retira a coluna classe, pega só os atributos

# y = saída esperada (a classe do vinho: 1, 2 ou 3)
y = df['classe'].values

print(f"\nFormato de X (entradas): {X.shape}")   # Deve ser (178, 13)
print(f"Formato de y (saídas):  {y.shape}")      # Deve ser (178,)

# =============================================================================
# SEÇÃO 4 - NORMALIZAÇÃO DOS DADOS (StandardScaler)
# =============================================================================
# Por que normalizar?
# Os atributos têm escalas muito diferentes:
#   - 'prolina' varia de ~278 a ~1680
#   - 'matiz'   varia de ~0.48 a ~1.71
# Sem normalização, atributos com valores maiores dominam o aprendizado.
#
# O StandardScaler transforma cada atributo para:
#   média = 0 e desvio padrão = 1
# Fórmula: x_norm = (x - média) / desvio_padrão
#
# IMPORTANTE: o scaler é ajustado APENAS nos dados de treino de cada rodada
# para evitar "data leakage" (vazamento de informação do teste para o treino).

# =============================================================================
# SEÇÃO 5 - DEFINIÇÃO DA ARQUITETURA DA MLP
# =============================================================================
# Topologia da rede:
#
#   Camada de Entrada:   13 neurônios  (um por atributo)
#         |
#   Camada Oculta 1:     64 neurônios  (ativação ReLU)
#         |
#   Camada Oculta 2:     32 neurônios  (ativação ReLU)
#         |
#   Camada de Saída:      3 neurônios  (softmax interno - uma por classe)
#
# Parâmetros do MLPClassifier:
#   hidden_layer_sizes = (64, 32)  -> define as duas camadas ocultas
#   activation = 'relu'            -> função de ativação das camadas ocultas
#                                     ReLU: f(x) = max(0, x)
#   solver = 'adam'                -> otimizador Adam (adaptativo, robusto)
#   max_iter = 500                 -> máximo de épocas de treinamento
#   random_state = None            -> seed diferente a cada rodada (aleatório)
#   early_stopping = True          -> para o treino se a val. loss parar de melhorar
#   validation_fraction = 0.1      -> 10% do treino usado para early stopping

def criar_modelo():
    """
    Cria e retorna uma instância nova do MLPClassifier.
    Chamada a cada rodada para garantir que o modelo começa do zero.
    """
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),  # Duas camadas ocultas: 64 e 32 neurônios
        activation='relu',            # ReLU nas camadas ocultas
        solver='adam',                # Otimizador Adam
        max_iter=500,                 # Máximo de iterações (épocas)
        early_stopping=True,          # Para antes se não houver melhoria
        validation_fraction=0.1,      # 10% do treino para validação interna
        n_iter_no_change=20,          # Nº de épocas sem melhoria para parar
        random_state=None             # Sem seed fixo: cada rodada é aleatória
    )

# =============================================================================
# SEÇÃO 6 - TREINAMENTO E TESTE (20 RODADAS)
# =============================================================================
# O modelo é treinado e testado 20 vezes.
# Em cada rodada:
#   1. Os dados são embaralhados e divididos aleatoriamente:
#      - 70% para treino
#      - 30% para teste
#   2. O scaler é ajustado SOMENTE nos dados de treino
#   3. Um modelo novo é criado e treinado
#   4. A acurácia no conjunto de teste é registrada
#
# Isso é importante para ter uma estimativa robusta do desempenho real,
# já que um único split pode ser favorável ou desfavorável por acaso.

N_RODADAS    = 20    # Número de rodadas de treino/teste
PERC_TREINO  = 0.70  # 70% dos dados para treino
PERC_TESTE   = 0.30  # 30% dos dados para teste

# Listas para armazenar os resultados de cada rodada
acuracias           = []   # Acurácia de cada rodada
melhor_acuracia     = 0    # Controla qual foi a melhor rodada
melhor_y_teste      = None # Rótulos reais da melhor rodada
melhor_y_pred       = None # Predições da melhor rodada

print("\n" + "=" * 60)
print("TREINAMENTO E TESTE - 20 RODADAS")
print("=" * 60)

for rodada in range(1, N_RODADAS + 1):

    # --- 6.1 Divisão aleatória treino/teste ---
    # O parâmetro stratify=y garante que as proporções das classes
    # sejam mantidas tanto no treino quanto no teste
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y,
        test_size=PERC_TESTE,   # 30% para teste
        stratify=y,              # Mantém proporção das classes
        random_state=None        # Aleatório a cada rodada
    )

    # --- 6.2 Normalização ---
    # O scaler aprende a média e desvio padrão SOMENTE do treino
    # e depois aplica a mesma transformação no teste
    scaler = StandardScaler()
    X_treino_norm = scaler.fit_transform(X_treino)  # Aprende e transforma treino
    X_teste_norm  = scaler.transform(X_teste)        # Só transforma o teste

    # --- 6.3 Criação e treinamento do modelo ---
    modelo = criar_modelo()
    modelo.fit(X_treino_norm, y_treino)  # Treina a rede neural

    # --- 6.4 Predição e avaliação ---
    y_pred   = modelo.predict(X_teste_norm)          # Faz predições no teste
    acuracia = accuracy_score(y_teste, y_pred) * 100 # Calcula acurácia em %

    acuracias.append(acuracia)
    print(f"  Rodada {rodada:02d}: Acurácia = {acuracia:.2f}%")

    # Guarda os dados da melhor rodada para a matriz de confusão
    if acuracia > melhor_acuracia:
        melhor_acuracia = acuracia
        melhor_y_teste  = y_teste
        melhor_y_pred   = y_pred

# =============================================================================
# SEÇÃO 7 - RESULTADOS CONSOLIDADOS
# =============================================================================

acuracias_np = np.array(acuracias)
media        = acuracias_np.mean()
desvio       = acuracias_np.std()
maximo       = acuracias_np.max()
minimo       = acuracias_np.min()

print("\n" + "=" * 60)
print("RESULTADOS CONSOLIDADOS (20 RODADAS)")
print("=" * 60)
print(f"  Acurácia Média:          {media:.2f}%")
print(f"  Desvio Padrão:           {desvio:.2f}%")
print(f"  Melhor Acurácia:         {maximo:.2f}%")
print(f"  Pior Acurácia:           {minimo:.2f}%")

# Relatório detalhado da melhor rodada
print("\nRelatório de Classificação (melhor rodada):")
print(classification_report(
    melhor_y_teste,
    melhor_y_pred,
    target_names=['Classe 1', 'Classe 2', 'Classe 3']
))

# =============================================================================
# SEÇÃO 8 - GERAÇÃO DOS GRÁFICOS
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('MLP - Classificação de Vinhos | Wine Dataset UCI', fontsize=14, fontweight='bold')

# --- Gráfico 1: Acurácia por Rodada ---
ax1 = axes[0]
cores = ['#2ecc71' if a >= media else '#e74c3c' for a in acuracias]
barras = ax1.bar(range(1, N_RODADAS + 1), acuracias, color=cores, edgecolor='black', linewidth=0.5)
ax1.axhline(y=media, color='navy', linestyle='--', linewidth=2, label=f'Média: {media:.2f}%')
ax1.set_xlabel('Rodada')
ax1.set_ylabel('Acurácia (%)')
ax1.set_title('Acurácia por Rodada')
ax1.set_xticks(range(1, N_RODADAS + 1))
ax1.set_ylim([max(0, minimo - 5), 105])
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
# Rótulo de valor em cada barra
for i, (barra, acc) in enumerate(zip(barras, acuracias)):
    ax1.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 0.3,
             f'{acc:.1f}', ha='center', va='bottom', fontsize=7, rotation=90)

# --- Gráfico 2: Distribuição das Acurácias (Boxplot + pontos) ---
ax2 = axes[1]
bp = ax2.boxplot(acuracias, patch_artist=True,
                 boxprops=dict(facecolor='#3498db', alpha=0.7),
                 medianprops=dict(color='red', linewidth=2))
ax2.scatter([1]*N_RODADAS, acuracias, color='navy', zorder=5, alpha=0.7, s=40)
ax2.set_ylabel('Acurácia (%)')
ax2.set_title('Distribuição das Acurácias')
ax2.set_xticklabels(['20 Rodadas'])
ax2.grid(axis='y', alpha=0.3)
# Anota média e desvio padrão
ax2.text(1.35, media, f'Média\n{media:.2f}%', va='center', color='navy', fontsize=9)
ax2.text(1.35, minimo, f'Mín\n{minimo:.2f}%', va='center', color='#e74c3c', fontsize=9)
ax2.text(1.35, maximo, f'Máx\n{maximo:.2f}%', va='center', color='#2ecc71', fontsize=9)

# --- Gráfico 3: Matriz de Confusão da Melhor Rodada ---
ax3 = axes[2]
cm = confusion_matrix(melhor_y_teste, melhor_y_pred)
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Classe 1', 'Classe 2', 'Classe 3'],
    yticklabels=['Classe 1', 'Classe 2', 'Classe 3'],
    ax=ax3, linewidths=0.5
)
ax3.set_xlabel('Classe Predita')
ax3.set_ylabel('Classe Real')
ax3.set_title(f'Matriz de Confusão\n(Melhor Rodada: {melhor_acuracia:.2f}%)')

plt.tight_layout()
plt.savefig('resultados_mlp_wine.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nGráfico salvo como: resultados_mlp_wine.png")

# =============================================================================
# SEÇÃO 9 - TABELA DE RESULTADOS DAS 20 RODADAS
# =============================================================================

print("\n" + "=" * 60)
print("TABELA DE ACURÁCIAS POR RODADA")
print("=" * 60)
tabela = pd.DataFrame({
    'Rodada': range(1, N_RODADAS + 1),
    'Acurácia (%)': [f"{a:.2f}" for a in acuracias],
    'Status': ['✓ Acima da média' if a >= media else '✗ Abaixo da média' for a in acuracias]
})
print(tabela.to_string(index=False))
print(f"\n{'─'*40}")
print(f"  Média:         {media:.2f}%")
print(f"  Desvio Padrão: {desvio:.2f}%")
print(f"  Máximo:        {maximo:.2f}%")
print(f"  Mínimo:        {minimo:.2f}%")

print("\n✅ Script concluído com sucesso!")
