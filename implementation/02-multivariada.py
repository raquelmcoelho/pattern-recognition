# =============================================================
# NAIVE BAYES - CLASSIFICADOR MULTIVARIADO
# =============================================================
# Aluna: Raquel Maciel Coelho de Sousa
# Dataset: Iris
# Descrição: Implementação do classificador Naive Bayes com
#            análise de acurácia utilizando todos os atributos
#            (multivariado).
#
# Estrutura:
#   1. Imports
#   2. Funções Matemáticas
#   3. Dados: Carregamento e Split
#   4. Treino
#   5. Classificação
#   6. Avaliação
#   7. Visualização
#   8. Execução Principal
# =============================================================


# ---- 1. IMPORTS ---------------------------------------------

import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---- 2. FUNÇÕES MATEMÁTICAS ---------------------------------


def media(x):
    return sum(x) / len(x)

def variancia(x):
    u = media(x)
    return sum((xi - u) ** 2 for xi in x) / len(x)


def covariancia(x1, x2):
    """Covariância entre dois atributos."""
    u1, u2 = media(x1), media(x2)
    return sum((a - u1) * (b - u2) for a, b in zip(x1, x2)) / len(x1)


def matriz_covariancia_dxd(atributos, dados):
    """
    Monta a matriz de covariância dxd para um par de atributos.
    Retorna np.array
    [
        [var11, cov12, . . . ,   cov1d],
        [cov21, var22, . . . ,   cov2d],
        [  .   ,  .          ,     .   ]
        [  .   ,      .      ,     .   ]
        [  .   ,          .  ,     .   ]
        [covd1, covd2, . . . ,   vardd]
    ]
    """
    tabela = np.zeros((len(atributos), len(atributos)))
    for i, a in enumerate(atributos):
        for j, b in enumerate(atributos):
            if(j >= i):
                tabela[i][j] = covariancia(dados[a].values, dados[b].values)
                tabela[j][i] = tabela[i][j]  # simetria

    return np.array(tabela)

def gaussiana_d(x, u, sigma):
    """
    Distribuição gaussiana multivariada.
      d     : dimensão (número de atributos)
      x     : vetor [x1, x2, ..., xd]
      u     : vetor de médias [u1, u2, ..., ud]
      sigma : matriz de covariância dxd
    """
    x, u = np.array(x), np.array(u)
    d = len(x)
    det = np.linalg.det(sigma)
    det = np.maximum(det, 1e-9)
    inv = np.linalg.inv(sigma)
    diff = x - u
    expoente = -0.5 * (diff @ inv @ diff)
    denominador = (2 * np.pi) ** (d / 2) * np.sqrt(det)
    return (1 / denominador) * np.exp(float(expoente))


# ---- 3. DADOS: CARREGAMENTO E SPLIT -------------------------


def carregar_dados(nome="iris"):
    return sns.load_dataset(nome)


def split_treino_teste(dados, proporcao_treino):
    """
    Embaralha e divide os dados em treino e teste.
    Retorna (dados_treino, dados_teste).
    """
    dados = dados.sample(frac=1).reset_index(drop=True)
    corte = int(proporcao_treino * len(dados))
    return dados.iloc[:corte], dados.iloc[corte:]


# ---- 4. TREINO ----------------------------------------------

# MULTIVARIADO
def treinar_multivariado(dados_treino, classes):
    """
    Calcula a prioris, médias e matrizes de covariância
    para cada classe.

    Retorna dicionário com estrutura:
    {
        'a_prioris':   { classe: probabilidade_a_priori },
        'medias':      { classe: [u1, u2, ..., ud] },
        'covariancias':{ classe: np.array dxd }
    }
    """
    modelo = {"a_prioris": {}, "medias": {}, "covariancias": {}}
    atributos = list(dados_treino.columns[:-1])

    for classe in classes:
        filtro = dados_treino[dados_treino["species"] == classe]
        modelo["a_prioris"][classe] = len(filtro) / len(dados_treino)
        modelo["medias"][classe] = [media(filtro[a].values) for a in atributos]
        modelo["covariancias"][classe] = matriz_covariancia_dxd(atributos, filtro)

    return modelo


# ---- 5. CLASSIFICAÇÃO ---------------------------------------


# MULTIVARIADO
def classificar_multivariado(linha_teste, modelo, classes):
    """
    Calcula a posteriori usando a gaussiana multivariada.

    Retorna classe predita
    """

    def verossimilhanca_d(x, classe):
        u = modelo["medias"][classe]
        sigma = modelo["covariancias"][classe]
        return gaussiana_d(x, u, sigma)

    def evidencia_d(x):
        return sum(
            verossimilhanca_d(x, c) * modelo["a_prioris"][c] for c in classes
        )

    def posteriori_d(x, classe):
        ev = max(evidencia_d(x), 1e-9)
        return (verossimilhanca_d(x, classe) * modelo["a_prioris"][classe]) / ev


    probs = {c: posteriori_d(linha_teste, c) for c in classes}
    resultado = max(probs, key=probs.get)

    return resultado


# ---- 6. AVALIAÇÃO -------------------------------------------

# MULTIVARIADO
def avaliar_multivariado(dados_teste, modelo, classes):
    """
    Roda o classificador multivariado sobre os dados de teste.
    Retorna a taxa de acerto (0–100).
    """
    acertos = 0

    for _, linha_teste in dados_teste.iterrows():
        resultado = classificar_multivariado(linha_teste[:-1], modelo, classes)
        if resultado == linha_teste["species"]:
            acertos += 1

    return (acertos / len(dados_teste)) * 100



def realizar_multivariado(dados, proporcao_treino):
    """
    Executa uma realização completa do ciclo multivariado:
    split → treino → avaliação.
    Retorna a taxa de acerto.
    """
    classes = dados["species"].unique()

    dados_treino, dados_teste = split_treino_teste(dados, proporcao_treino)
    modelo = treinar_multivariado(dados_treino, classes)
    return avaliar_multivariado(dados_teste, modelo, classes)


# ---- 7. VISUALIZAÇÃO ----------------------------------------
# Filosofia: os gráficos são gerados UMA VEZ ao final de todas
# as realizações, de forma agregada. Nunca dentro do loop.


# MULTIVARIADA
def plotar_gaussianas_multivariado(dados, proporcao_treino):
    """
    Plota as distribuição da gaussiana aprendida,
    treinando UMA VEZ (modelo representativo).
    Não é chamado dentro do loop de realizações.
    """
    pass

def plotar_acuracias_multivariado(historico_taxas_acerto):
    """
    historico_taxas_acerto: lista [taxa1, taxa2, ..., taxaN]
    """
    pass

# ---- 8. EXECUÇÃO PRINCIPAL ----------------------------------

if __name__ == "__main__":
    PROPORCAO_TREINO = 0.8
    N_REALIZACOES = 100

    dados = carregar_dados()

    print("=" * 50)
    print("MULTIVARIADO")
    print("=" * 50)

    historico = []

    for i in range(N_REALIZACOES):
        taxa_de_acerto = realizar_multivariado(dados, PROPORCAO_TREINO)
        historico.append(taxa_de_acerto)

    acuracia_media =  sum(historico) / N_REALIZACOES

    print(f"\nAcurácia média ({N_REALIZACOES} realizações): {acuracia_media:.2f}%")
    print(f"\nDesvio padrão: {math.sqrt(variancia(historico)):.2f}")

    # Gráficos
    plotar_acuracias_multivariado(historico)
    plotar_gaussianas_multivariado(dados, PROPORCAO_TREINO)
