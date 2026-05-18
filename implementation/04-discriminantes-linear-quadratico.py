# =============================================================
# DISCRIMINANTES LINEAR (LDA) E QUADRÁTICO (QDA)
# =============================================================
# Aluna: Raquel Maciel Coelho de Sousa
# Dataset:
#   1. Flor Iris
#   2. Vertebral Column
#   3. Breast Cancer
#   4. Dermatology
#   5. Artificial I
#
# Descrição: Implementação e comparação de quatro classificadores:
#            LDA, QDA, KNN e DMC.
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
import itertools
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


def distancia_euclidiana(x, y):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))


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


def gaussiana_1d(x, u, v):
    """
    Distribuição gaussiana univariada.
      x : valor observado
      u : média
      v : variância
    """
    v = max(v, 1e-9)
    return (1 / math.sqrt(2 * math.pi * v)) * math.exp(-((x - u) ** 2) / (2 * v))


def limpar_dados(dados):
    """
    Preencher dados faltantes com média da coluna (apenas para colunas numéricas).
    """
    for col in dados.columns[:-1]:  # não mexe na coluna de classe
        serie = dados[col]
        if pd.api.types.is_numeric_dtype(serie):
            dados[col] = serie.fillna(media(serie.dropna().values))
            continue

    return dados

def carregar_dados(nome):
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
    rng = np.random.default_rng(seed)

    classes_config = {
        "circulo":   (1.0, 4.0, 0.5),
        "triangulo": (4.0, 4.0, 0.5),
        "estrela":   (2.0, 2.0, 0.5),
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
    corte = min(corte, len(dados) - 1)  # garante ao menos 1 amostra de teste
    return dados.iloc[:corte], dados.iloc[corte:]

# ---- 4. TREINO ----------------------------------------------

def treinar_bayes(dados_treino, classes):
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

def treinar_naive(dados_treino, classes):
    """
    Naive Bayes univariado: estima média e variância de cada atributo
    separadamente para cada classe (assume independência entre atributos).

    Retorna dicionário com estrutura:
    {
        'a_prioris': { classe: prob },
        'medias':    { classe: { atributo: valor } },
        'variancias':{ classe: { atributo: valor } }
    }
    """
    modelo = {"a_prioris": {}, "medias": {}, "variancias": {}}
    atributos = list(dados_treino.columns[:-1])

    for classe in classes:
        filtro = dados_treino[dados_treino["target"] == classe]
        modelo["a_prioris"][classe] = len(filtro) / len(dados_treino)
        modelo["medias"][classe]    = {a: media(filtro[a].values) for a in atributos}
        modelo["variancias"][classe] = {a: variancia(filtro[a].values) for a in atributos}

    return modelo

def treinar_dmc(dados_treino, classes):
    """
    DMC (Distância Mínima ao Centroide).
    Treino: calcula o centroide (vetor de médias) de cada classe.

    Retorna { classe: [media_a1, media_a2, ..., media_ad] }
    """
    atributos = list(dados_treino.columns[:-1])
    centroides = {}
    for classe in classes:
        filtro = dados_treino[dados_treino["target"] == classe]
        centroides[classe] = [media(filtro[a].values) for a in atributos]
    return centroides


# =============================================================
#  DISCRIMINANTES LINEAR (LDA) E QUADRÁTICO (QDA)
# =============================================================
#
# Situações de matriz de covariância:
#
#  MODO "qda_full"    →  Σᵢ individual por classe,
#                        forma completa não-diagonal.
#                        g_i(x) = -½(x-μᵢ)ᵀΣᵢ⁻¹(x-μᵢ) + ln p(Wᵢ) + cᵢ
#
#  MODO "qda_diag"    → mesma equação porém Σᵢ forçada diagonal
#                        (zera covariâncias cruzadas entre atributos).
#                        Assume independência condicional por classe.
#
#  MODO "lda"         → Σ compartilhada (pooled) → discriminante linear.
#                        g_i(x) = wᵢᵀx + wᵢ₀
#                        wᵢ = Σ⁻¹μᵢ,  wᵢ₀ = ln p(Wᵢ) - ½μᵢᵀΣ⁻¹μᵢ
#
#  MODO "lda_esferico" → Σ = σ²I (mesma variância em todos os eixos).
#                        g_i(x) = (1/σ²)μᵢᵀx + wᵢ₀
#                        É o caso mais simples; superfície de decisão é
#                        hiperplano ortogonal a μᵢ − μⱼ.
# =============================================================

EPSILON = 1e-6  # regularização para evitar matriz singular


def regularizar(sigma):
    """Adiciona pequena perturbação na diagonal para garantir invertibilidade."""
    d = sigma.shape[0]
    return sigma + EPSILON * np.eye(d)


def pooled_covariance(dados_treino, classes):
    """
    Calcula a matriz de covariância pooled (compartilhada entre classes).
    Fórmula: Σ = Σᵢ [ (nᵢ-1) * Σᵢ ] / (N - C)
    onde nᵢ = nº amostras da classe i, N = total, C = nº classes.
    """
    atributos = list(dados_treino.columns[:-1])
    N = len(dados_treino)
    C = len(classes)

    acumulador = np.zeros((len(atributos), len(atributos)))
    for classe in classes:
        filtro = dados_treino[dados_treino["target"] == classe]
        ni = len(filtro)
        sigma_i = matriz_covariancia_dxd(atributos, filtro)
        acumulador += (ni - 1) * sigma_i

    return acumulador / max(N - C, 1)


def treinar_qda(dados_treino, classes, modo="qda_full"):
    """
    Treina o classificador QDA.

    Parâmetros
    ----------
    modo : "qda_full"  → Σᵢ completa por classe
           "qda_diag"  → Σᵢ diagonal por classe (covariâncias cruzadas = 0)

    Retorna modelo com:
        'a_prioris':   { classe: probabilidade_a_priori },
        'medias':      { classe: [u1, u2, ..., ud] },
        'covariancias':{ classe: np.array dxd }
        'modo'        : str
    """
    modelo = {"a_prioris": {}, "medias": {}, "covariancias": {}, "modo": modo}
    atributos = list(dados_treino.columns[:-1])

    for classe in classes:
        filtro = dados_treino[dados_treino["target"] == classe]
        modelo["a_prioris"][classe] = len(filtro) / len(dados_treino)
        modelo["medias"][classe] = np.array([media(filtro[a].values) for a in atributos])

        sigma = matriz_covariancia_dxd(atributos, filtro)

        if modo == "qda_diag":
            # força diagonal: mantém só as variâncias, anula covariâncias
            sigma = np.diag(np.diag(sigma))

        modelo["covariancias"][classe] = sigma

    return modelo


def treinar_lda(dados_treino, classes, modo="lda"):
    """
    Treina o classificador LDA.

    Parâmetros
    ----------
    modo : "lda"        → pooled covariance Σ (compartilhada)
           "lda_esferico" → Σ = σ²I, onde σ² = média das variâncias pooled

    Retorna modelo com:
        'a_prioris': { classe: probabilidade_a_priori }
        'medias'   : { classe: [u1, u2, ..., ud] }
        'w'        : { classe: np.array }    ← vetor de pesos wᵢ = Σ⁻¹μᵢ
        'w0'       : { classe: float }       ← bias wᵢ₀
        'sigma'    : np.array dxd            ← Σ usada
        'modo'     : str
    """
    modelo = {"a_prioris": {}, "medias": {}, "w": {}, "w0": {}, "modo": modo}
    atributos = list(dados_treino.columns[:-1])

    sigma_pooled = pooled_covariance(dados_treino, classes)

    if modo == "lda_esferico":
        # Σ = σ²I  onde σ² é a variância média da diagonal pooled
        sigma2 = float(np.mean(np.diag(sigma_pooled)))
        sigma2 = max(sigma2, EPSILON)
        d = sigma_pooled.shape[0]
        sigma_pooled = sigma2 * np.eye(d)

    sigma_reg = regularizar(sigma_pooled)
    sigma_inv = np.linalg.inv(sigma_reg)
    modelo["sigma"] = sigma_pooled

    for classe in classes:
        filtro = dados_treino[dados_treino["target"] == classe]
        ni = len(filtro)
        N  = len(dados_treino)

        modelo["a_prioris"][classe] = ni / N
        mu = np.array([media(filtro[a].values) for a in atributos])
        modelo["medias"][classe] = mu

        # wᵢ = Σ⁻¹μᵢ  (Simplificação linear)
        w = sigma_inv @ mu
        # wᵢ₀ = ln p(Wᵢ) - ½ μᵢᵀ Σ⁻¹ μᵢ
        w0 = np.log(max(ni / N, EPSILON)) - 0.5 * float(mu @ sigma_inv @ mu)

        modelo["w"][classe] = w
        modelo["w0"][classe] = w0

    return modelo


# ---- 5. CLASSIFICAÇÃO ---------------------------------------

def classificar_bayes(linha_teste, modelo, classes):
    """
    Calcula a posteriori usando a gaussiana multivariada.
    Retorna classe predita.
    """

    def verossimilhanca_d(x, classe):
        u = modelo["medias"][classe]
        sigma = modelo["covariancias"][classe]
        return gaussiana_d(x, u, sigma)

    def evidencia_d(x):
        return sum(verossimilhanca_d(x, c) * modelo["a_prioris"][c] for c in classes)

    def posteriori_d(x, classe):
        ev = max(float(evidencia_d(x)), 1e-9)
        return (verossimilhanca_d(x, classe) * modelo["a_prioris"][classe]) / ev

    probs = {c: posteriori_d(linha_teste, c) for c in classes}
    return max(probs, key=probs.get)

def classificar_naive(linha_teste, modelo, classes, atributos):
    """
    Naive Bayes: multiplica as gaussianas 1D de cada atributo para obter
    a verossimilhança conjunta (pressuposto de independência).
    Retorna classe predita.
    """
    def verossimilhanca(x_dict, classe):
        prob = 1.0
        for a in atributos:
            u = modelo["medias"][classe][a]
            v = modelo["variancias"][classe][a]
            prob *= gaussiana_1d(x_dict[a], u, v)
        return prob

    def evidencia(x_dict):
        return sum(verossimilhanca(x_dict, c) * modelo["a_prioris"][c] for c in classes)

    def posteriori(x_dict, classe):
        ev = max(evidencia(x_dict), 1e-9)
        return (verossimilhanca(x_dict, classe) * modelo["a_prioris"][classe]) / ev

    x_dict = {a: linha_teste[i] for i, a in enumerate(atributos)}
    probs = {c: posteriori(x_dict, c) for c in classes}
    return max(probs, key=probs.get)

def classificar_dmc(linha_teste, centroides, classes):
    """
    DMC: classifica pelo centroide mais próximo (distância euclidiana).
    """
    distancias = {
        c: distancia_euclidiana(linha_teste, centroides[c])
        for c in classes
    }
    return min(distancias, key=distancias.get)


def classificar_qda(linha_teste, modelo, classes):
    """
    QDA — Discriminante Quadrático.

    Computa para cada classe a função discriminante:
        g_i(x) = -½(x-μᵢ)ᵀΣᵢ⁻¹(x-μᵢ) + ln p(Wᵢ) + cᵢ
    onde  cᵢ = -½ ln|Σᵢ| - (d/2) ln(2π)

    A classe com maior g_i(x) é escolhida.
    """
    x = np.array(linha_teste, dtype=float)
    d = len(x)

    scores = {}
    for classe in classes:
        mu    = modelo["medias"][classe]
        sigma = modelo["covariancias"][classe]
        priori = modelo["a_prioris"][classe]

        sigma_reg = regularizar(sigma)
        sigma_inv = np.linalg.inv(sigma_reg)
        det = max(float(np.linalg.det(sigma_reg)), EPSILON)

        diff = x - mu
        # termo quadrático
        g = -0.5 * float(diff @ sigma_inv @ diff)
        # log-priori
        g += np.log(max(priori, EPSILON))
        # constante cᵢ = -½ ln|Σᵢ| - (d/2) ln(2π)
        g += -0.5 * np.log(det) - (d / 2) * np.log(2 * np.pi)

        scores[classe] = g

    return max(scores, key=scores.get)


def classificar_lda(linha_teste, modelo, classes):
    """
    LDA — Discriminante Linear.

    Computa para cada classe (Eq. 9 dos slides):
        g_i(x) = wᵢᵀ x + wᵢ₀
    onde wᵢ e wᵢ₀ foram calculados no treino.

    A classe com maior g_i(x) é escolhida.
    """
    x = np.array(linha_teste, dtype=float)

    scores = {
        classe: float(modelo["w"][classe] @ x) + modelo["w0"][classe]
        for classe in classes
    }
    return max(scores, key=scores.get)

def classificar_knn(linha_teste, dados_treino, classes, k=5):
    """
    KNN (K Vizinhos Mais Próximos).
    Não há treino — os dados de treino são o próprio modelo.

    Calcula a distância euclidiana para todos os pontos de treino,
    seleciona os K mais próximos e faz votação majoritária.
    """
    atributos = list(dados_treino.columns[:-1])

    # distância de linha_teste para cada ponto de treino
    distancias = []
    for _, linha_treino in dados_treino.iterrows():
        d = distancia_euclidiana(linha_teste, linha_treino[atributos].values)
        distancias.append((d, linha_treino["target"]))

    # ordena por distância e pega os K menores
    distancias.sort(key=lambda x: x[0])
    k_vizinhos = [classe for _, classe in distancias[:k]]

    # votação: classe mais frequente entre os K vizinhos
    votos = {c: k_vizinhos.count(c) for c in classes}
    return max(votos, key=votos.get)


# ---- 6. AVALIAÇÃO -------------------------------------------

def realizar_bayes(dados, proporcao_treino):
    """
    Executa uma realização completa: split → treino → avaliação.
    Retorna (taxa, registros, modelo, dados_treino, dados_teste).
    """
    classes = dados["target"].unique()
    dados_treino, dados_teste = split_treino_teste(dados, proporcao_treino)
    modelo = treinar_bayes(dados_treino, classes)

    acertos = 0
    registros = []
    for _, linha_teste in dados_teste.iterrows():
        previsto = classificar_bayes(linha_teste[:-1].values, modelo, classes)
        real = linha_teste["target"]
        registros.append((real, previsto))
        if previsto == real:
            acertos += 1

    taxa = (acertos / len(dados_teste)) * 100

    return taxa, registros, modelo, dados_treino, dados_teste

def realizar_naive(dados, proporcao_treino):
    classes  = dados["target"].unique()
    atributos = list(dados.columns[:-1])
    dados_treino, dados_teste = split_treino_teste(dados, proporcao_treino)
    modelo = treinar_naive(dados_treino, classes)

    acertos = 0
    registros = []
    for _, linha in dados_teste.iterrows():
        previsto = classificar_naive(linha[:-1].values, modelo, classes, atributos)
        real = linha["target"]
        registros.append((real, previsto))
        if previsto == real:
            acertos += 1

    taxa = (acertos / len(dados_teste)) * 100
    return taxa, registros, modelo, dados_treino, dados_teste

def realizar_dmc(dados, proporcao_treino):
    classes = dados["target"].unique()
    dados_treino, dados_teste = split_treino_teste(dados, proporcao_treino)
    centroides = treinar_dmc(dados_treino, classes)

    acertos = 0
    registros = []
    for _, linha in dados_teste.iterrows():
        previsto = classificar_dmc(linha[:-1].values, centroides, classes)
        real = linha["target"]
        registros.append((real, previsto))
        if previsto == real:
            acertos += 1

    taxa = (acertos / len(dados_teste)) * 100
    return taxa, registros, centroides, dados_treino, dados_teste

def realizar_knn(dados, proporcao_treino, k=5):
    classes = dados["target"].unique()
    dados_treino, dados_teste = split_treino_teste(dados, proporcao_treino)

    acertos = 0
    registros = []
    for _, linha in dados_teste.iterrows():
        previsto = classificar_knn(linha[:-1].values, dados_treino, classes, k)
        real = linha["target"]
        registros.append((real, previsto))
        if previsto == real:
            acertos += 1

    taxa = (acertos / len(dados_teste)) * 100
    return taxa, registros, dados_treino, dados_teste


def realizar_qda(dados, proporcao_treino, modo="qda_full"):
    """
    Executa uma realização do QDA.
    modo: "qda_full" ou "qda_diag"
    """
    classes = dados["target"].unique()
    dados_treino, dados_teste = split_treino_teste(dados, proporcao_treino)
    modelo = treinar_qda(dados_treino, classes, modo=modo)

    acertos = 0
    registros = []
    for _, linha in dados_teste.iterrows():
        previsto = classificar_qda(linha[:-1].values, modelo, classes)
        real = linha["target"]
        registros.append((real, previsto))
        if previsto == real:
            acertos += 1

    taxa = (acertos / len(dados_teste)) * 100
    return taxa, registros, modelo, dados_treino, dados_teste


def realizar_lda(dados, proporcao_treino, modo="lda"):
    """
    Executa uma realização do LDA.
    modo: "lda" ou "lda_esferico"
    """
    classes = dados["target"].unique()
    dados_treino, dados_teste = split_treino_teste(dados, proporcao_treino)
    modelo = treinar_lda(dados_treino, classes, modo=modo)

    acertos = 0
    registros = []
    for _, linha in dados_teste.iterrows():
        previsto = classificar_lda(linha[:-1].values, modelo, classes)
        real = linha["target"]
        registros.append((real, previsto))
        if previsto == real:
            acertos += 1

    taxa = (acertos / len(dados_teste)) * 100
    return taxa, registros, modelo, dados_treino, dados_teste


# ---- 7. VISUALIZAÇÃO ----------------------------------------

def melhor_par_atributos(dados, classes):
    """
    Escolhe o par (a1, a2) que maximize a separação real das massas de dados.
    Critério: (fisher(a1) + fisher(a2))
    
    Fisher: variância_fora(ai)  
            ------------------   
            variância _dentro(ai)

    Isso prioriza eixos onde as classes estão longe umas das outras e são compactas.

    - variância fora: variância das médias entre classes, distancia dos centros
      (quanto mais as classes se afastam, melhor)
    - variância dentro: média das variâncias dentro de cada classe, espalhamento da nuvem
      (quanto mais compactas, melhor)
    """
    atributos = list(dados.columns[:-1])
    
    def fisher(atributo):
        filtro = [dados[dados["target"] == c][atributo].values for c in classes]
        medias_por_classe = [media(f) for f in filtro]
        variancia_fora = variancia(medias_por_classe)
        variancia_dentro = media([variancia(f) for f in filtro])

        return variancia_fora /(variancia_dentro + 1e-6)

    
    fishers = {a: fisher(a) for a in atributos}
    pares = list(itertools.combinations(atributos, 2))
    scores_pares = {}
    
    for a1, a2 in pares:
        scores_pares[(a1, a2)] = (fishers[a1] + fishers[a2])

    melhor = max(scores_pares, key=scores_pares.get)

    print(f"  Scores de separabilidade (Fisher-like):")
    for par, score in sorted(scores_pares.items(), key=lambda x: -x[1]):
        print(f"    {par[0]:20s} × {par[1]:20s} : {score:.4f}")

    return melhor

def plotar_superficie_decisao(nome_dataset, dados, modelo, classes, par_atributos,
                               dados_treino, dados_teste, classificador_fn=None,
                               titulo_extra=""):
    """
    Plota a superfície de decisão para um par de atributos.

    classificador_fn : função f(ponto, modelo, classes) → classe
                       Se None, usa classificar_bayes.
    titulo_extra     : string adicionada ao título (ex: "QDA full")
    """
    if classificador_fn is None:
        classificador_fn = classificar_bayes
    a1, a2 = par_atributos
    atributos = list(dados.columns[:-1])
    cores = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#FF9800", "#00BCD4"]
    cores_classe = {c: cor for c, cor in zip(classes, cores)}

    #  grade de pixels cobrindo o espaço dos dados 
    margem = 0.5
    x1_lin = np.linspace(dados[a1].min() - margem, dados[a1].max() + margem, 200)
    x2_lin = np.linspace(dados[a2].min() - margem, dados[a2].max() + margem, 200)
    X1, X2 = np.meshgrid(x1_lin, x2_lin)

    # os atributos fora do par são fixados na média global
    # para que o classificador (que usa todos os atributos) funcione
    medias_globais = dados[atributos].apply(media)

    #  classifica cada pixel da grade 
    grade_classes = np.empty(X1.shape, dtype=object)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            ponto = medias_globais.copy()
            ponto[a1] = X1[i, j]
            ponto[a2] = X2[i, j]
            grade_classes[i, j] = classificador_fn(ponto.values, modelo, classes)

    #  converte classes para inteiros para detectar mudanças entre pixels 
    indice_classe = {c: i for i, c in enumerate(classes)}
    grade_num = np.vectorize(indice_classe.get)(grade_classes)

    fig, ax = plt.subplots(figsize=(9, 7))

    #  pinta cada região com a cor da classe 
    for c in classes:
        mascara = (grade_classes == c).astype(float)
        ax.contourf(X1, X2, mascara, levels=[0.5, 1.5],
                    colors=[cores_classe[c]], alpha=0.25)

    #  fronteira de decisão: onde a classe muda entre pixels vizinhos 
    borda_h = np.diff(grade_num, axis=0) != 0  # mudança vertical
    borda_v = np.diff(grade_num, axis=1) != 0  # mudança horizontal

    borda_x, borda_y = [], []
    for i, j in zip(*np.where(borda_h)):
        borda_x.append(x1_lin[j])
        borda_y.append(x2_lin[i])
    for i, j in zip(*np.where(borda_v)):
        borda_x.append(x1_lin[j])
        borda_y.append(x2_lin[i])

    ax.scatter(borda_x, borda_y, c="black", s=0.3, zorder=2)

    #  curvas de nível das gaussianas (apenas para modelos com covariância por classe) 
    i1 = atributos.index(a1)
    i2 = atributos.index(a2)

    if "covariancias" in modelo:
        for c in classes:
            u_full = np.array(modelo["medias"][c])
            sigma_full = modelo["covariancias"][c]

            # extrai submatriz 2x2 do par escolhido
            u_2d = np.array([u_full[i1], u_full[i2]])
            sigma_2d = np.array([
                [sigma_full[i1, i1], sigma_full[i1, i2]],
                [sigma_full[i2, i1], sigma_full[i2, i2]]
            ])

            Z = np.array([
                [gaussiana_d(np.array([x1, x2]), u_2d, sigma_2d)
                 for x1 in x1_lin]
                for x2 in x2_lin
            ])

            ax.contour(X1, X2, Z, levels=4,
                       colors=[cores_classe[c]], linewidths=1.5, zorder=3)

    elif "sigma" in modelo:
        # LDA: uma única gaussiana por classe usando Σ compartilhada
        sigma_full = modelo["sigma"]
        sigma_2d = np.array([
            [sigma_full[i1, i1], sigma_full[i1, i2]],
            [sigma_full[i2, i1], sigma_full[i2, i2]]
        ])
        for c in classes:
            u_full = np.array(modelo["medias"][c])
            u_2d = np.array([u_full[i1], u_full[i2]])
            Z = np.array([
                [gaussiana_d(np.array([x1, x2]), u_2d, sigma_2d)
                 for x1 in x1_lin]
                for x2 in x2_lin
            ])
            ax.contour(X1, X2, Z, levels=4,
                       colors=[cores_classe[c]], linewidths=1.5, zorder=3)

    #  pontos de treino (círculo) e teste (X) por classe 
    for c in classes:
        cor = cores_classe[c]
        t = dados_treino[dados_treino["target"] == c]
        ts = dados_teste[dados_teste["target"] == c]
        ax.scatter(t[a1], t[a2], color=cor, marker="o", s=25,
                   edgecolors="white", linewidths=0.4,
                   label=f"{c} treino", zorder=4)
        ax.scatter(ts[a1], ts[a2], color=cor, marker="X", s=60,
                   edgecolors="black", linewidths=0.5,
                   label=f"{c} teste", zorder=5)

    ax.set_xlabel(a1)
    ax.set_ylabel(a2)
    titulo = f"Superfície de Decisão — {nome_dataset}"
    if titulo_extra:
        titulo += f" [{titulo_extra}]"
    titulo += f"\n({a1} × {a2})"
    ax.set_title(titulo)
    ax.legend(fontsize=8, loc="upper left", markerscale=1.2)
    plt.tight_layout()
    plt.show()

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
            ax.text(j, i, str(matriz[i][j]),
                    ha="center", va="center", fontsize=12,
                    color="white" if matriz[i][j] > matriz.max() * 0.5 else "black")

    plt.tight_layout()
    plt.show()



# ---- 8. EXECUÇÃO PRINCIPAL ----------------------------------

DATASETS_COM_SUPERFICIE_DECISAO = {"iris", "vertebral_column", "artificial_I"}

def executar_dataset(nome_dataset, k_knn=5):
    PROPORCAO_TREINO = 0.8
    N_REALIZACOES = 25

    print("=" * 60)
    print("EXECUTANDO DATASET:", nome_dataset)
    print("=" * 60)

    dados = carregar_dados(nome_dataset)
    classes = dados["target"].unique()

    # ---- realizações -------------------------------------------
        historico_bayes     = []
        historico_naive     = []
        historico_dmc       = []
        historico_knn       = []
    historico_qda_full  = []   # QDA: Σᵢ completa por classe
    historico_qda_diag  = []   # QDA: Σᵢ diagonal por classe
    historico_lda       = []   # LDA: Σ pooled compartilhada
    historico_lda_esf   = []   # LDA: Σ = σ²I (esférica)

    for _ in range(N_REALIZACOES):
        historico_bayes.append(realizar_bayes(dados, PROPORCAO_TREINO))
        historico_naive.append(realizar_naive(dados, PROPORCAO_TREINO))
        historico_dmc.append(realizar_dmc(dados, PROPORCAO_TREINO))
        historico_knn.append(realizar_knn(dados, PROPORCAO_TREINO, k=k_knn))
        historico_qda_full.append(realizar_qda(dados, PROPORCAO_TREINO, modo="qda_full"))
        historico_qda_diag.append(realizar_qda(dados, PROPORCAO_TREINO, modo="qda_diag"))
        historico_lda.append(realizar_lda(dados, PROPORCAO_TREINO, modo="lda"))
        historico_lda_esf.append(realizar_lda(dados, PROPORCAO_TREINO, modo="lda_esferico"))

    # ---- resumo de acurácia ------------------------------------
    def resumo(historico, nome):
        taxas = [h[0] for h in historico]
        acc   = media(taxas)
        dev   = math.sqrt(variancia(taxas))
        print(f"  {nome:22s}: {acc:6.2f}% ± {dev:.2f}%")
        return acc, dev

    print(f"\nAcurácia média ({N_REALIZACOES} realizações):")
    print(f"  {'--- Trabalhos Anteriores ---':22s}")
    acc_bayes,    dev_bayes    = resumo(historico_bayes,    "Bayesiano Gauss.")
    acc_naive,    dev_naive    = resumo(historico_naive,    "Naive Bayes")
    acc_dmc,      dev_dmc      = resumo(historico_dmc,      "DMC")
    acc_knn,      dev_knn      = resumo(historico_knn,      f"KNN (k={k_knn})")
    print(f"  {'--- Trabalho Atual (LDA/QDA) ---':22s}")
    acc_qda_full, dev_qda_full = resumo(historico_qda_full, "QDA (Σᵢ completa)")
    acc_qda_diag, dev_qda_diag = resumo(historico_qda_diag, "QDA (Σᵢ diagonal)")
    acc_lda,      dev_lda      = resumo(historico_lda,      "LDA (Σ pooled)")
    acc_lda_esf,  dev_lda_esf  = resumo(historico_lda_esf,  "LDA (Σ=σ²I esf.)")

    # ---- realização representativa: QDA full -------------------
    rep_qda  = min(historico_qda_full, key=lambda h: abs(h[0] - acc_qda_full))
    rep_lda  = min(historico_lda,      key=lambda h: abs(h[0] - acc_lda))
    rep_naive = min(historico_naive,   key=lambda h: abs(h[0] - acc_naive))

    # ---- matriz de confusão (realização representativa QDA) ----
    print(f"\nMatriz de confusão — QDA full (realização mais próxima da média):")
    plotar_matriz_confusao(nome_dataset + " [QDA full]", rep_qda[1], classes)

    # ---- superfície de decisão ---------------------------------
    if nome_dataset in DATASETS_COM_SUPERFICIE_DECISAO:
        par = melhor_par_atributos(dados, classes)
        print(f"\nPar escolhido para superfície de decisão: {par}")

        taxa_rep, registros_rep, modelo_rep, treino_rep, teste_rep = rep_qda
        plotar_superficie_decisao(
            nome_dataset, dados, modelo_rep, classes,
            par, treino_rep, teste_rep,
            classificador_fn=classificar_qda,
            titulo_extra="QDA Σᵢ completa"
        )

        taxa_rep, registros_rep, modelo_rep, treino_rep, teste_rep = rep_lda
        plotar_superficie_decisao(
            nome_dataset, dados, modelo_rep, classes,
            par, treino_rep, teste_rep,
            classificador_fn=classificar_lda,
            titulo_extra="LDA Σ pooled"
        )


if __name__ == "__main__":
    executar_dataset("iris")
    executar_dataset("artificial_I")
    executar_dataset("vertebral_column")
    executar_dataset("breast_cancer")
    executar_dataset("dermatology")