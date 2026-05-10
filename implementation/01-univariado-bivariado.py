# =============================================================
# NAIVE BAYES - CLASSIFICADOR UNIVARIADO E BIVARIADO
# =============================================================
# Aluna: Raquel Maciel Coelho de Sousa
# Dataset: Iris
# Descrição: Implementação do classificador Naive Bayes com
#            análise de acurácia por atributo individual
#            (univariado) e por par de atributos (bivariado).
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


# ---- 2. FUNÇÕES MATEMÁTICAS ---------------------------------

def media(x):
    return sum(x) / len(x)


def variancia(x):
    u = media(x)
    return sum((xi - u) ** 2 for xi in x) / len(x)


def gaussiana_1d(x, u, v):
    """
    Distribuição gaussiana univariada.
      x : valor observado
      u : média
      v : variância
    """
    v = max(v, 1e-9)  # evita divisão por zero
    return (1 / math.sqrt(2 * math.pi * v)) * math.exp(-((x - u) ** 2) / (2 * v))


# BIVARIADA
def covariancia(x1, x2):
    """Covariância entre dois atributos."""
    u1, u2 = media(x1), media(x2)
    return sum((a - u1) * (b - u2) for a, b in zip(x1, x2)) / len(x1)


def matriz_covariancia_2x2(x1, x2):
    """
    Monta a matriz de covariância 2x2 para um par de atributos.
    Retorna np.array [[var1, cov], [cov, var2]]
    """
    return np.array([
        [variancia(x1), covariancia(x1, x2)],
        [covariancia(x1, x2), variancia(x2)]
    ])


def gaussiana_2d(x, u, sigma):
    """
    Distribuição gaussiana bivariada (multivariada 2D).
      x     : vetor [x1, x2]
      u     : vetor de médias [u1, u2]
      sigma : matriz de covariância 2x2
    """
    x, u = np.array(x), np.array(u)
    d = len(x)
    det = np.linalg.det(sigma)
    det = max(det, 1e-9)
    inv = np.linalg.inv(sigma)
    diff = x - u
    expoente = -0.5 * diff @ inv @ diff
    denominador = (2 * math.pi) ** (d / 2) * math.sqrt(det)
    return (1 / denominador) * math.exp(expoente)


# ---- 3. DADOS: CARREGAMENTO E SPLIT -------------------------

def carregar_dados():
    return sns.load_dataset("iris")


def split_treino_teste(dados, proporcao_treino):
    """
    Embaralha e divide os dados em treino e teste.
    Retorna (dados_treino, dados_teste).
    """
    dados = dados.sample(frac=1).reset_index(drop=True)
    corte = int(proporcao_treino * len(dados))
    return dados.iloc[:corte], dados.iloc[corte:]


# ---- 4. TREINO ----------------------------------------------

def treinar_univariado(dados_treino, classes, atributos):
    """
    Calcula médias, variâncias e probabilidades a priori
    para cada classe, por atributo individual.

    Retorna dicionário com estrutura:
    {
        'a_prioris': { classe: probabilidade_a_priori },
        'medias':    { classe: { atributo: media } },
        'variancias':{ classe: { atributo: variancia } }
    }
    """
    modelo = {"a_prioris": {}, "medias": {}, "variancias": {}}

    for classe in classes:
        filtro = dados_treino[dados_treino["species"] == classe]
        modelo["a_prioris"][classe] = len(filtro) / len(dados_treino)
        modelo["medias"][classe] = {}
        modelo["variancias"][classe] = {}

        for atributo in atributos:
            x = filtro[atributo].values
            modelo["medias"][classe][atributo] = media(x)
            modelo["variancias"][classe][atributo] = variancia(x)

    return modelo


# BIVARIADA
def treinar_bivariado(dados_treino, classes, pares):
    """
    Calcula médias, matrizes de covariância e a prioris
    para cada classe, por par de atributos.

    Retorna dicionário com estrutura:
    {
        'a_prioris':   { classe: probabilidade_a_priori },
        'medias':      { classe: { par: [u1, u2] } },
        'covariancias':{ classe: { par: np.array 2x2 } }
    }
    """
    modelo = {"a_prioris": {}, "medias": {}, "covariancias": {}}

    for classe in classes:
        filtro = dados_treino[dados_treino["species"] == classe]
        modelo["a_prioris"][classe] = len(filtro) / len(dados_treino)
        modelo["medias"][classe] = {}
        modelo["covariancias"][classe] = {}

        for par in pares:
            a1, a2 = par
            x1 = filtro[a1].values
            x2 = filtro[a2].values
            modelo["medias"][classe][par] = [media(x1), media(x2)]
            modelo["covariancias"][classe][par] = matriz_covariancia_2x2(x1, x2)

    return modelo

# ---- 5. CLASSIFICAÇÃO ---------------------------------------

def classificar_univariado(linha, modelo, classes, atributos):
    """
    Para cada atributo individualmente, calcula a posteriori
    de cada classe e retorna a classe predita.

    Retorna { atributo: classe_predita }
    """
    def verossimilhanca(x, classe, atributo):
        u = modelo["medias"][classe][atributo]
        v = modelo["variancias"][classe][atributo]
        return gaussiana_1d(x, u, v)

    def evidencia(x, atributo):
        return sum(
            modelo["a_prioris"][c] * verossimilhanca(x, c, atributo)
            for c in classes
        )

    def posteriori(x, classe, atributo):
        ev = evidencia(x, atributo)
        ev = max(ev, 1e-9)
        return (verossimilhanca(x, classe, atributo) * modelo["a_prioris"][classe]) / ev

    resultados = {}
    for atributo in atributos:
        x = linha[atributo]
        probs = {c: posteriori(x, c, atributo) for c in classes}
        resultados[atributo] = max(probs, key=probs.get)

    return resultados


# BIVARIADA
def classificar_bivariado(linha, modelo, classes, pares):
    """
    Para cada par de atributos, calcula a posteriori
    usando a gaussiana 2D e retorna a classe predita.

    Retorna { par: classe_predita }
    """
    def verossimilhanca_2d(x, classe, par):
        u = modelo["medias"][classe][par]
        sigma = modelo["covariancias"][classe][par]
        return gaussiana_2d(x, u, sigma)

    def evidencia_2d(x, par):
        return sum(
            modelo["a_prioris"][c] * verossimilhanca_2d(x, c, par)
            for c in classes
        )

    def posteriori_2d(x, classe, par):
        ev = max(evidencia_2d(x, par), 1e-9)
        return (verossimilhanca_2d(x, classe, par) * modelo["a_prioris"][classe]) / ev

    resultados = {}
    for par in pares:
        x = [linha[par[0]], linha[par[1]]]
        probs = {c: posteriori_2d(x, c, par) for c in classes}
        resultados[par] = max(probs, key=probs.get)

    return resultados


# ---- 6. AVALIAÇÃO -------------------------------------------

def avaliar_univariado(dados_teste, modelo, classes, atributos):
    """
    Roda o classificador univariado sobre os dados de teste.
    Retorna a taxa de acerto (0–100) por atributo.
    """
    acertos = {atributo: 0 for atributo in atributos}

    for _, linha in dados_teste.iterrows():
        resultados = classificar_univariado(linha, modelo, classes, atributos)
        for atributo in atributos:
            if resultados[atributo] == linha["species"]:
                acertos[atributo] += 1

    return {k: (v / len(dados_teste)) * 100 for k, v in acertos.items()}


# BIVARIADA
def avaliar_bivariado(dados_teste, modelo, classes, pares):
    """
    Roda o classificador bivariado sobre os dados de teste.
    Retorna a taxa de acerto (0–100) por par de atributos.
    """
    acertos = {par: 0 for par in pares}

    for _, linha in dados_teste.iterrows():
        resultados = classificar_bivariado(linha, modelo, classes, pares)
        for par in pares:
            if resultados[par] == linha["species"]:
                acertos[par] += 1

    return {k: (v / len(dados_teste)) * 100 for k, v in acertos.items()}


def realizar_univariado(dados, proporcao_treino):
    """
    Executa uma realização completa do ciclo univariado:
    split → treino → avaliação.
    Retorna as taxas de acerto por atributo.
    """
    classes = dados["species"].unique()
    atributos = list(dados.columns[:-1])

    dados_treino, dados_teste = split_treino_teste(dados, proporcao_treino)
    modelo = treinar_univariado(dados_treino, classes, atributos)
    return avaliar_univariado(dados_teste, modelo, classes, atributos)

def realizar_bivariado(dados, proporcao_treino, pares):
    """
    Executa uma realização completa do ciclo bivariado:
    split → treino → avaliação.
    Retorna as taxas de acerto por par de atributos.
    """
    classes = dados["species"].unique()

    dados_treino, dados_teste = split_treino_teste(dados, proporcao_treino)
    modelo = treinar_bivariado(dados_treino, classes, pares)
    return avaliar_bivariado(dados_teste, modelo, classes, pares)


# ---- 7. VISUALIZAÇÃO ----------------------------------------
# Filosofia: os gráficos são gerados UMA VEZ ao final de todas
# as realizações, de forma agregada. Nunca dentro do loop.

def plotar_acuracias_univariado(historico_acuracias, atributos):
    """
    Boxplot das acurácias por atributo ao longo de N realizações.
    Mostra média, mediana e dispersão de forma limpa.

    historico_acuracias: lista de dicts { atributo: acuracia }
    """
    dados_plot = {a: [r[a] for r in historico_acuracias] for a in atributos}
    df_plot = pd.DataFrame(dados_plot)

    fig, ax = plt.subplots(figsize=(9, 5))
    df_plot.boxplot(ax=ax, grid=False, patch_artist=True,
                    boxprops=dict(facecolor="#cce5ff", color="#333"),
                    medianprops=dict(color="#d62728", linewidth=2))

    ax.set_title("Acurácia por Atributo — Univariado\n(distribuição sobre N realizações)",
                 fontsize=13, pad=12)
    ax.set_ylabel("Acurácia (%)")
    ax.set_ylim(0, 105)
    ax.axhline(y=df_plot.values.mean(), color="gray", linestyle="--",
               linewidth=1, label="Média global")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plotar_gaussianas_univariado(dados, proporcao_treino):
    """
    Plota as distribuições gaussianas aprendidas para cada atributo,
    treinando UMA VEZ (modelo representativo).
    Não é chamado dentro do loop de realizações.
    """
    classes = dados["species"].unique()
    atributos = list(dados.columns[:-1])

    dados_treino, _ = split_treino_teste(dados, proporcao_treino)
    modelo = treinar_univariado(dados_treino, classes, atributos)

    cores = {"setosa": "#2196F3", "versicolor": "#4CAF50", "virginica": "#FF5722"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()

    for i, atributo in enumerate(atributos):
        ax = axes[i]
        x = np.linspace(dados[atributo].min() - 1, dados[atributo].max() + 1, 300)

        for classe in classes:
            u = modelo["medias"][classe][atributo]
            v = modelo["variancias"][classe][atributo]
            y = [gaussiana_1d(xi, u, v) for xi in x]
            label = f"{classe}  (μ={u:.2f}, σ²={v:.2f})"
            cor = cores.get(classe, None)
            ax.plot(x, y, label=label, color=cor)
            ax.fill_between(x, y, alpha=0.12, color=cor)

        ax.set_title(f"Distribuição: {atributo}", fontsize=11)
        ax.set_xlabel(atributo)
        ax.set_ylabel("p(x | classe)")
        ax.legend(fontsize=8)

    fig.suptitle("Verossimilhanças Gaussianas — Univariado", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


# BIVARIADA
def plotar_gaussianas_bivariado(dados, proporcao_treino, pares):
    """
    Plota as distribuições gaussianas aprendidas para cada par de atributos,
    treinando UMA VEZ (modelo representativo).
    Não é chamado dentro do loop de realizações.
    """
    from mpl_toolkits.mplot3d import Axes3D

    classes = dados["species"].unique()

    dados_treino, _ = split_treino_teste(dados, proporcao_treino)
    modelo = treinar_bivariado(dados_treino, classes, pares)

    cores = {"setosa": "blue", "versicolor": "green", "virginica": "red"}

    for par in pares:
        a1, a2 = par
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")

        x1_range = np.linspace(dados[a1].min() - 1, dados[a1].max() + 1, 60)
        x2_range = np.linspace(dados[a2].min() - 1, dados[a2].max() + 1, 60)
        X1, X2 = np.meshgrid(x1_range, x2_range)

        for classe in classes:
            u = modelo["medias"][classe][par]
            sigma = modelo["covariancias"][classe][par]
            Z = np.array([
                [gaussiana_2d([x1, x2], u, sigma) for x1 in x1_range]
                for x2 in x2_range
            ])
            ax.plot_surface(X1, X2, Z, alpha=0.5,
                            color=cores.get(classe), label=classe)

        ax.set_xlabel(a1)
        ax.set_ylabel(a2)
        ax.set_zlabel("p(x | classe)")
        ax.set_title(f"Gaussiana 2D: {a1} × {a2}")
        plt.tight_layout()
        plt.show()


def plotar_acuracias_bivariado(historico_acuracias, atributos, pares):
    """
    Exibe dois subplots lado a lado:
      - Barplot: acurácia média ± desvio padrão por par
      - Heatmap: acurácia média numa matriz atributo × atributo

    historico_acuracias: lista de dicts { (a1, a2): acuracia }
    """
    # -- médias e desvios por par --
    medias_par = {par: np.mean([r[par] for r in historico_acuracias]) for par in pares}
    desvios_par = {par: np.std([r[par] for r in historico_acuracias]) for par in pares}

    # -- monta matriz para o heatmap (4x4, diagonal vazia) --
    n = len(atributos)
    matriz = np.full((n, n), np.nan)
    for (a1, a2), acc in medias_par.items():
        i, j = atributos.index(a1), atributos.index(a2)
        matriz[i][j] = acc
        matriz[j][i] = acc  # espelha para preencher os dois lados

    rotulos = [a.replace("_", "\n") for a in atributos]
    labels_pares = [f"{a1.split('_')[0]}_{a1.split('_')[1][0]}\n×\n{a2.split('_')[0]}_{a2.split('_')[1][0]}"
                    for (a1, a2) in pares]

    fig, (ax_bar, ax_heat) = plt.subplots(1, 2, figsize=(14, 5))

    # -- Barplot --
    x = np.arange(len(pares))
    cores_bar = plt.cm.Blues(np.linspace(0.4, 0.85, len(pares)))
    barras = ax_bar.bar(x, [medias_par[p] for p in pares],
                        yerr=[desvios_par[p] for p in pares],
                        color=cores_bar, capsize=5, edgecolor="white", linewidth=0.8)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels_pares, fontsize=9)
    ax_bar.set_ylabel("Acurácia (%)")
    ax_bar.set_ylim(0, 105)
    ax_bar.set_title("Acurácia por Par de Atributos\n(média ± desvio padrão)", fontsize=11)
    ax_bar.bar_label(barras, fmt="%.1f%%", padding=4, fontsize=8)
    ax_bar.grid(axis="y", linestyle="--", alpha=0.4)
    ax_bar.set_axisbelow(True)

    # -- Heatmap --
    im = ax_heat.imshow(matriz, cmap="Blues", vmin=0, vmax=100)
    ax_heat.set_xticks(range(n))
    ax_heat.set_yticks(range(n))
    ax_heat.set_xticklabels(rotulos, fontsize=9)
    ax_heat.set_yticklabels(rotulos, fontsize=9)
    ax_heat.set_title("Heatmap de Acurácia\n(atributo × atributo)", fontsize=11)
    plt.colorbar(im, ax=ax_heat, label="Acurácia (%)")

    for i in range(n):
        for j in range(n):
            if not np.isnan(matriz[i][j]):
                ax_heat.text(j, i, f"{matriz[i][j]:.1f}%",
                             ha="center", va="center", fontsize=9,
                             color="white" if matriz[i][j] > 70 else "black")

    plt.tight_layout()
    plt.show()



# ---- 8. EXECUÇÃO PRINCIPAL ----------------------------------

if __name__ == "__main__":

    PROPORCAO_TREINO = 0.8
    N_REALIZACOES = 100

    dados = carregar_dados()
    atributos = list(dados.columns[:-1])

    # -- Univariado ------------------------------------------

    print("=" * 50)
    print("UNIVARIADO")
    print("=" * 50)

    historico_univariado = []

    for i in range(N_REALIZACOES):
        taxas_de_acerto = realizar_univariado(dados, PROPORCAO_TREINO)
        historico_univariado.append(taxas_de_acerto)

    acuracia_media = {
        a: sum(r[a] for r in historico_univariado) / N_REALIZACOES
        for a in atributos
    }

    print(f"\nAcurácia média ({N_REALIZACOES} realizações):")
    for atributo, acc in acuracia_media.items():
        print(f"  {atributo:20s}: {acc:.2f}%")

    # Gráficos
    plotar_acuracias_univariado(historico_univariado, atributos)
    plotar_gaussianas_univariado(dados, PROPORCAO_TREINO)

    # -- Bivariado --------------------------------------------

    print("=" * 50)
    print("BIVARIADO")
    print("=" * 50)
    
    pares = list(itertools.combinations(atributos, 2))
    historico_bivariado = []
    
    for i in range(N_REALIZACOES):
        taxas_de_acerto = realizar_bivariado(dados, PROPORCAO_TREINO, pares)
        historico_bivariado.append(taxas_de_acerto)
    
    acuracia_media_biv = {
        par: sum(r[par] for r in historico_bivariado) / N_REALIZACOES
        for par in pares
    }
    
    print(f"\nAcurácia média por par ({N_REALIZACOES} realizações):")
    for par, acc in acuracia_media_biv.items():
        print(f"  {str(par):45s}: {acc:.2f}%")
    
    # Gráficos
    plotar_acuracias_bivariado(historico_bivariado, atributos, pares)
    plotar_gaussianas_bivariado(dados, PROPORCAO_TREINO, pares)