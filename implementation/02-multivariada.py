# =============================================================
# CLASSIFICADOR BAYESIANO GAUSSIANO MULTIVARIADO
# =============================================================
# Aluna: Raquel Maciel Coelho de Sousa
# Dataset:
#   1. Flor Iris
#   2. Vertebral Column
#   3. Breast Cancer
#   4. Dermatology
#   5. Artificial I
#
# Descrição: Implementação do classificador Bayesiano Gaussiano com
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
from ucimlrepo import fetch_ucirepo

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
            if j >= i:
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

    # regularização: evita matriz singular somando epsilon na diagonal
    epsilon = 1e-6
    sigma_reg = sigma + epsilon * np.eye(d)

    det = np.linalg.det(sigma_reg)
    det = max(float(det), 1e-9)
    inv = np.linalg.inv(sigma_reg)

    diff = x - u
    expoente = -0.5 * (diff @ inv @ diff)
    expoente = float(np.clip(expoente, -500, 500))  # evita overflow no exp
    denominador = (2 * np.pi) ** (d / 2) * np.sqrt(det)
    return (1 / denominador) * np.exp(float(expoente))


# ---- 3. DADOS: CARREGAMENTO E SPLIT -------------------------


def limpar_dados(dados):
    """
    Limpeza inteligente por tipo de coluna (exceto a coluna de classe):

      - Numérico puro         → mantém, preenche NaN com a média da coluna

    """

    for col in dados.columns[:-1]:  # não mexe na coluna de classe
        serie = dados[col]

        # já é numérico: preenche NaN com a média
        if pd.api.types.is_numeric_dtype(serie):
            dados[col] = serie.fillna(serie.mean())
            continue

    return dados


def carregar_dados(nome="iris"):
    """
    Carrega e limpa um dataset pelo nome.
    Suportados: "iris", "vertebral_column", "breast_cancer", "dermatology", "artificial_I"
    A coluna de classe é sempre padronizada como "target".
    """
    if nome == "artificial_I":
        return gerar_artificial_I()

    ids = {
        "breast_cancer": 17,
        "dermatology": 33,
        "iris": 53,
        "vertebral_column": 212,
    }

    ds = fetch_ucirepo(id=ids[nome])
    X = ds.data.features
    y = ds.data.targets
    y.columns = ["target"]

    dados = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    dados = limpar_dados(dados)

    return dados


def gerar_artificial_I(n_por_classe=50, seed=42):
    """
    Gera dataset Artificial I com 3 classes gaussianas 2D.
    """
    rng = np.random.default_rng(seed)

    classes_config = {
        "circulo": (1.0, 4.0, 0.5),
        "triangulo": (4.0, 4.0, 0.5),
        "estrela": (2.0, 2.0, 0.5),
    }

    partes = []
    for classe, (mx, my, dp) in classes_config.items():
        x1 = rng.normal(mx, dp, n_por_classe)
        x2 = rng.normal(my, dp, n_por_classe)
        df = pd.DataFrame({"x1": x1, "x2": x2, "target": classe})
        partes.append(df)

    return pd.concat(partes).reset_index(drop=True)

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
def treinar(dados_treino, classes):
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
        filtro = dados_treino[dados_treino["target"] == classe]
        modelo["a_prioris"][classe] = len(filtro) / len(dados_treino)
        modelo["medias"][classe] = [media(filtro[a].values) for a in atributos]
        modelo["covariancias"][classe] = matriz_covariancia_dxd(atributos, filtro)

    return modelo


# ---- 5. CLASSIFICAÇÃO ---------------------------------------


# MULTIVARIADO
def classificar(linha_teste, modelo, classes):
    """
    Calcula a posteriori usando a gaussiana multivariada.

    Retorna classe predita
    """

    def verossimilhanca_d(x, classe):
        u = modelo["medias"][classe]
        sigma = modelo["covariancias"][classe]
        return gaussiana_d(x, u, sigma)

    def evidencia_d(x):
        return sum(verossimilhanca_d(x, c) * modelo["a_prioris"][c] for c in classes)

    def posteriori_d(x, classe):
        ev = np.maximum(evidencia_d(x), 1e-9)
        return (verossimilhanca_d(x, classe) * modelo["a_prioris"][classe]) / ev


    probs = {c: posteriori_d(linha_teste, c) for c in classes}
    resultado = max(probs, key=probs.get)

    return resultado


# ---- 6. AVALIAÇÃO -------------------------------------------


# MULTIVARIADO
def avaliar(dados_teste, modelo, classes):
    """
    Roda o classificador multivariado sobre os dados de teste.
    Retorna a taxa de acerto (0–100).
    """
    acertos = 0
    registros = []

    for _, linha_teste in dados_teste.iterrows():
        previsto = classificar(linha_teste[:-1], modelo, classes)
        real = linha_teste["target"]
        registros.append((real, previsto))
        if previsto == real:
            acertos += 1

    taxa = (acertos / len(dados_teste)) * 100
    return taxa, registros


def realizar(dados, proporcao_treino):
    """
    Executa uma realização completa do ciclo multivariado:
    split → treino → avaliação.
    Retorna a taxa de acerto.
    """
    classes = dados["target"].unique()

    dados_treino, dados_teste = split_treino_teste(dados, proporcao_treino)
    modelo = treinar(dados_treino, classes)
    return avaliar(dados_teste, modelo, classes)


# ---- 7. VISUALIZAÇÃO ----------------------------------------
# Filosofia: os gráficos são gerados UMA VEZ ao final de todas
# as realizações, de forma agregada. Nunca dentro do loop.


def plotar_matriz_confusao(nome_dataset, registros, classes):
    """
    registros: lista de tuplas (real, previsto)
    """
    n = len(classes)
    indice = {c: i for i, c in enumerate(classes)}
    matriz = np.zeros((n, n), dtype=int)

    for real, previsto in registros:
        i = indice[real]
        j = indice[previsto]
        matriz[i][j] += 1

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matriz, cmap="Blues")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticklabels(classes, fontsize=10)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de Confusão: {nome_dataset}")
    plt.colorbar(im, ax=ax)

    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                str(matriz[i][j]),
                ha="center",
                va="center",
                fontsize=12,
                color="white" if matriz[i][j] > matriz.max() * 0.5 else "black",
            )

    plt.tight_layout()
    plt.show()


# ---- 8. EXECUÇÃO PRINCIPAL ----------------------------------


def executar_dataset(nome_dataset):
    PROPORCAO_TREINO = 0.8
    N_REALIZACOES = 25

    print("=" * 50)
    print("EXECUTANDO DATASET:", nome_dataset)
    print("=" * 50)

    dados = carregar_dados(nome_dataset)
    historico = []

    for i in range(N_REALIZACOES):
        taxa_de_acerto, registros = realizar(dados, PROPORCAO_TREINO)
        historico.append((taxa_de_acerto, registros))

    acuracia_media = sum(h[0] for h in historico) / N_REALIZACOES

    print(f"\nAcurácia média ({N_REALIZACOES} realizações): {acuracia_media:.2f}%")
    print(f"\nDesvio padrão: {math.sqrt(variancia([h[0] for h in historico])):.2f}")

    # Plotar matriz de confusão da realização mais representativa (mais próxima da média)
    classes = dados["target"].unique()
    realizacao_representativa = min(historico, key=lambda h: abs(h[0] - acuracia_media))
    plotar_matriz_confusao(nome_dataset, realizacao_representativa[1], classes)


if __name__ == "__main__":
    executar_dataset("iris")
    executar_dataset("vertebral_column")
    executar_dataset("breast_cancer")
    executar_dataset("dermatology")
    executar_dataset("artificial_I")
