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
# Descrição: Implementação e comparação de três classificadores:
#            Bayesiano Gaussiano Multivariado, KNN e DMC.
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


def distancia_euclidiana(x, y):
    """
    Distância euclidiana entre dois vetores x e y.
      x, y : listas ou arrays de mesmo tamanho
    """
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


# ---- 3. DADOS: CARREGAMENTO E SPLIT -------------------------


def limpar_dados(dados):
    """
    Limpeza inteligente por tipo de coluna (exceto a coluna de classe):
      - Numérico puro → mantém, preenche NaN com a média da coluna
    """
    for col in dados.columns[:-1]:  # não mexe na coluna de classe
        serie = dados[col]
        if pd.api.types.is_numeric_dtype(serie):
            dados[col] = serie.fillna(serie.mean())
            continue

    return dados

def normalizar_dados(dados):
    """
    Normalização Min-Max por atributo: escala cada coluna para [0, 1].
    Importante para KNN e DMC, que usam distância euclidiana.
    O Bayesiano não precisa, mas não é prejudicado por ela.
    """
    dados_norm = dados.copy()
    for col in dados.columns[:-1]:  # não mexe na coluna target
        x = dados[col].values
        x_min = min(x)
        x_max = max(x)
        amplitude = x_max - x_min
        if amplitude == 0:
            dados_norm[col] = 0.0  # atributo constante → sem informação
        else:
            dados_norm[col] = [(xi - x_min) / amplitude for xi in x]
    return dados_norm

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
    dados = normalizar_dados(dados)

    return dados


def gerar_artificial_I(n_por_classe=50, seed=42):
    """
    Gera dataset Artificial I com 3 classes gaussianas 2D.
    """
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
    Garante ao menos 1 amostra de teste.
    Retorna (dados_treino, dados_teste).
    """
    dados = dados.sample(frac=1).reset_index(drop=True)
    corte = int(proporcao_treino * len(dados))
    corte = min(corte, len(dados) - 1)  # garante ao menos 1 amostra de teste
    return dados.iloc[:corte], dados.iloc[corte:]


# ---- 4. TREINO ----------------------------------------------


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


def classificar(linha_teste, modelo, classes):
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


# ---- 6. AVALIAÇÃO -------------------------------------------


def avaliar(dados_teste, modelo, classes):
    """
    Roda o classificador multivariado sobre os dados de teste.
    Retorna (taxa_de_acerto, registros).
    """
    acertos = 0
    registros = []

    for _, linha_teste in dados_teste.iterrows():
        previsto = classificar(linha_teste[:-1].values, modelo, classes)
        real = linha_teste["target"]
        registros.append((real, previsto))
        if previsto == real:
            acertos += 1

    taxa = (acertos / len(dados_teste)) * 100
    return taxa, registros


def realizar(dados, proporcao_treino):
    """
    Executa uma realização completa: split → treino → avaliação.
    Retorna (taxa, registros, modelo, dados_treino, dados_teste).
    """
    classes = dados["target"].unique()
    dados_treino, dados_teste = split_treino_teste(dados, proporcao_treino)
    modelo = treinar(dados_treino, classes)
    taxa, registros = avaliar(dados_teste, modelo, classes)
    return taxa, registros, modelo, dados_treino, dados_teste


# -- DMC ------------------------------------------------------

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


def classificar_dmc(linha_teste, centroides, classes):
    """
    DMC: classifica pelo centroide mais próximo (distância euclidiana).
    """
    distancias = {
        c: distancia_euclidiana(linha_teste, centroides[c])
        for c in classes
    }
    return min(distancias, key=distancias.get)


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


# -- KNN ------------------------------------------------------

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


# ---- 7. VISUALIZAÇÃO ----------------------------------------
# Filosofia: gráficos gerados UMA VEZ ao final de todas as realizações.


def melhor_par_atributos(dados, classes):
    """
    Escolhe o par (a1, a2) que maximize a separação real das massas de dados.
    Critério: (Variância entre médias de classes) / (Média das variâncias internas).
    Isso prioriza eixos onde as classes estão longe umas das outras e são compactas.
    """
    atributos = list(dados.columns[:-1])
    
    def score_separabilidade(atrib):
        # Médias de cada classe para este atributo
        medias_por_classe = [dados[dados["target"] == c][atrib].mean() for c in classes]
        # Variância das médias (o quanto os centros estão longe)
        var_entre = variancia(medias_por_classe)
        
        # Variância dentro de cada classe (o quanto a nuvem é 'espalhada')
        var_dentro = media([variancia(dados[dados["target"] == c][atrib].values) for c in classes])
        
        # Razão de Fisher simplificada: quanto maior, mais 'separado' e 'compacto'
        return var_entre / (var_dentro + 1e-6)

    # Calculamos o score individual de cada atributo
    scores_individuais = {a: score_separabilidade(a) for a in atributos}
    
    # O melhor par será a combinação dos dois atributos com maiores scores individuais
    # pois eles oferecem as melhores direções de separação independentes
    pares = list(itertools.combinations(atributos, 2))
    scores_pares = {}
    
    for a1, a2 in pares:
        # O score do par é a soma da capacidade de separação de cada eixo
        scores_pares[(a1, a2)] = scores_individuais[a1] + scores_individuais[a2]

    melhor = max(scores_pares, key=scores_pares.get)

    print(f"  Scores de separabilidade (Fisher-like):")
    for par, score in sorted(scores_pares.items(), key=lambda x: -x[1]):
        print(f"    {par[0]:20s} × {par[1]:20s} : {score:.4f}")

    return melhor





def plotar_superficie_decisao(nome_dataset, dados, modelo, classes, par_atributos, dados_treino, dados_teste):
    """
    Plota a superfície de decisão para um par de atributos.

    Técnica: classifica cada ponto de uma grade 2D e pinta pela classe
    vencedora. A fronteira de decisão é desenhada onde a classe muda
    entre pixels vizinhos (pintada de preto), conforme sugerido pelo professor.

    par_atributos : (nome_col_1, nome_col_2)
    dados_treino  : DataFrame da realização representativa
    dados_teste   : DataFrame da realização representativa
    """
    a1, a2 = par_atributos
    atributos = list(dados.columns[:-1])
    cores = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#FF9800", "#00BCD4"]
    cores_classe = {c: cor for c, cor in zip(classes, cores)}

    # -- grade de pixels cobrindo o espaço dos dados --
    margem = 0.5
    x1_lin = np.linspace(dados[a1].min() - margem, dados[a1].max() + margem, 200)
    x2_lin = np.linspace(dados[a2].min() - margem, dados[a2].max() + margem, 200)
    X1, X2 = np.meshgrid(x1_lin, x2_lin)

    # os atributos fora do par são fixados na média global
    # para que o classificador (que usa todos os atributos) funcione
    medias_globais = dados[atributos].mean()

    # -- classifica cada pixel da grade --
    grade_classes = np.empty(X1.shape, dtype=object)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            ponto = medias_globais.copy()
            ponto[a1] = X1[i, j]
            ponto[a2] = X2[i, j]
            grade_classes[i, j] = classificar(ponto.values, modelo, classes)

    # -- converte classes para inteiros para detectar mudanças entre pixels --
    indice_classe = {c: i for i, c in enumerate(classes)}
    grade_num = np.vectorize(indice_classe.get)(grade_classes)

    fig, ax = plt.subplots(figsize=(9, 7))

    # -- pinta cada região com a cor da classe --
    for c in classes:
        mascara = (grade_classes == c).astype(float)
        ax.contourf(X1, X2, mascara, levels=[0.5, 1.5],
                    colors=[cores_classe[c]], alpha=0.25)

    # -- fronteira de decisão: onde a classe muda entre pixels vizinhos --
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

    # -- curvas de nível das gaussianas (formato elipsoidal por classe) --
    i1 = atributos.index(a1)
    i2 = atributos.index(a2)

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

    # -- pontos de treino (círculo) e teste (X) por classe --
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
    ax.set_title(f"Superfície de Decisão — {nome_dataset}\n({a1} × {a2})")
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

# datasets para os quais será plotada a superfície de decisão
DATASETS_COM_SUPERFICIE = {"iris", "vertebral_column", "artificial_I"}


def executar_dataset(nome_dataset, k_knn=5):
    PROPORCAO_TREINO = 0.8
    N_REALIZACOES = 25

    print("=" * 50)
    print("EXECUTANDO DATASET:", nome_dataset)
    print("=" * 50)

    dados = carregar_dados(nome_dataset)
    classes = dados["target"].unique()

    historico_bayes = []
    historico_dmc   = []
    historico_knn   = []

    for _ in range(N_REALIZACOES):
        historico_bayes.append(realizar(dados, PROPORCAO_TREINO))
        historico_dmc.append(realizar_dmc(dados, PROPORCAO_TREINO))
        historico_knn.append(realizar_knn(dados, PROPORCAO_TREINO, k=k_knn))

    def resumo(historico, nome):
        taxas = [h[0] for h in historico]
        acc   = media(taxas)
        dev   = math.sqrt(variancia(taxas))
        print(f"  {nome:12s}: {acc:.2f}% ± {dev:.2f}%")
        return acc, dev

    print(f"\nAcurácia média ({N_REALIZACOES} realizações):")
    acc_bayes, dev_bayes = resumo(historico_bayes, "Bayesiano")
    acc_dmc,   dev_dmc   = resumo(historico_dmc,   "DMC")
    acc_knn,   dev_knn   = resumo(historico_knn,   f"KNN (k={k_knn})")

    # realização mais próxima da média de cada classificador
    rep_bayes = min(historico_bayes, key=lambda h: abs(h[0] - acc_bayes))
    rep_dmc   = min(historico_dmc,   key=lambda h: abs(h[0] - acc_dmc))

    # matriz de confusão do Bayesiano (realização representativa)
    plotar_matriz_confusao(nome_dataset, rep_bayes[1], classes)

    # superfície de decisão (só para os datasets pedidos pelo professor)
    if nome_dataset in DATASETS_COM_SUPERFICIE:
        par = melhor_par_atributos(dados, classes)
        print(f"\nPar escolhido para superfície de decisão: {par}")
        print(f"  (critério: maior variância das médias entre classes)")
        taxa_rep, registros_rep, modelo_rep, treino_rep, teste_rep = rep_bayes
        plotar_superficie_decisao(
            nome_dataset, dados, modelo_rep, classes,
            par, treino_rep, teste_rep
        )


if __name__ == "__main__":
    # executar_dataset("iris")
    executar_dataset("vertebral_column")
    # executar_dataset("breast_cancer")
    # executar_dataset("dermatology")
    # executar_dataset("artificial_I")