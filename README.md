# 🍷 Classificação de Vinhos com MLP — TDE FIA 2026

**Disciplina:** Fundamentos de Inteligência Artificial  
**Curso:** Análise e Desenvolvimento de Sistemas — UNIFACEMA  
**Atividade:** Trabalho Discente Efetivo (TDE) — 5º Período

Aplicação de uma Rede Neural do tipo **MLP (Multi-Layer Perceptron)** para classificação de vinhos utilizando o [Wine Dataset da UCI](https://archive.ics.uci.edu/dataset/109/wine). O modelo é treinado e testado **20 vezes com splits aleatórios**, gerando métricas de desempenho, matriz de confusão e gráficos de análise.

---

## 📁 Estrutura do Projeto

```
projeto/
├── mlp_wine.py     # Script principal
├── wine.data       # Dataset (ver instruções abaixo)
├── README.md       # Este arquivo
└── venv/           # Ambiente virtual (criado por você, não sobe pro Git)
```

> **Nota:** o arquivo `wine.data` deve estar na mesma pasta que `mlp_wine.py`.  
> Se não estiver, o script baixa automaticamente da UCI.

---

## ⚙️ Pré-requisitos

- Python **3.10 ou superior**
- Git

---

## 🚀 Como rodar

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
```

### 2. Crie e ative o ambiente virtual

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

> Você saberá que o venv está ativo quando aparecer `(venv)` no início do terminal.

### 3. Instale as dependências

```bash
pip install numpy pandas matplotlib scikit-learn seaborn
```

### 4. Execute o script

```bash
python mlp_wine.py
```

---

## 📊 O que o script gera

- **No terminal:**
  - Visão geral do dataset (178 amostras, 3 classes, 13 atributos)
  - Acurácia de cada uma das 20 rodadas
  - Acurácia média, desvio padrão, melhor e pior resultado
  - Relatório de classificação detalhado (precision, recall, F1-score)
  - Tabela final com status de cada rodada

- **Arquivo gerado:**
  - `resultados_mlp_wine.png` — gráfico com acurácia por rodada, boxplot e matriz de confusão

---

## 🧠 Arquitetura da Rede Neural

```
Entrada:        13 neurônios  (atributos químicos do vinho)
Camada Oculta 1: 64 neurônios  (ativação ReLU)
Camada Oculta 2: 32 neurônios  (ativação ReLU)
Saída:           3 neurônios  (classes 1, 2 e 3)

Otimizador: Adam
Normalização: StandardScaler (média=0, desvio padrão=1)
Split: 70% treino / 30% teste (aleatório a cada rodada)
```

---

## 📦 Dependências

| Biblioteca | Uso |
|---|---|
| `numpy` | Operações numéricas |
| `pandas` | Leitura e manipulação do dataset |
| `matplotlib` | Geração de gráficos |
| `seaborn` | Heatmap da matriz de confusão |
| `scikit-learn` | MLP, normalização e métricas |

## 👥 Integrantes
 
* Antonio Gabriel Conceição Santos
* Adriano Rikelmy Aguiar da Silva
* Marcelo Henrique Abreu Silva
