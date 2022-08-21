import numpy as np
from ACO import ACO


def distancia_euclidiana(x1: float, y1: float, x2: float, y2: float) -> float:
    """Função que calcula a distância euclidiana entre dois pontos

    Args:
        x1 (float): Coordenada x do ponto 1
        y1 (float): Coordenada y do ponto 1
        x2 (float): Coordenada x do ponto 2
        y2 (float): Coordenada y do ponto 2

    Returns:
        float: Distância euclidiana
    """
    return ((x1-x2)**2 + (y1-y2)**2)**(1/2)


def matriz_de_adjacencia(nome_arquivo: str) -> np.ndarray:
    """Retorna a matriz de adjacência a partir das cooredadas
    das cidades em um arquivo.

    Args:
        nome_arquivo (str): Nome do arquivo.

    Returns:
        np.array: Matriz de adjacência.
    """
    lines: list[str] = list()
    with open(nome_arquivo, 'r') as arq:
        lines = arq.readlines()

    linha_dimensao: str = lines[3]
    # Remove o \n do final da string e a divide em uma lista relacao ao espaco
    numero_cidades: int = int(linha_dimensao.strip().split()[1])
    # Obtendo as linhas que contém apenas as coordenadas
    coordenadas: list[str] = lines[6:6+numero_cidades]

    coordenada_x: list[float] = list()
    coordenada_y: list[float] = list()

    for i in coordenadas:
        coordenadas_split: list[str] = i.split()
        coordenada_x.append(float(coordenadas_split[1]))
        coordenada_y.append(float(coordenadas_split[2]))

    matrix_adjacencia: list[list[float]] = []

    for i in range(0, numero_cidades):
        matriz_adjacencia_aux: list = []
        for j in range(0, numero_cidades):
            matriz_adjacencia_aux.append(distancia_euclidiana(coordenada_x[i],
                                                              coordenada_y[i],
                                                              coordenada_x[j],
                                                              coordenada_y[j]))

        matrix_adjacencia.append(matriz_adjacencia_aux)

    return np.array(matrix_adjacencia)


if __name__ == "__main__":

    matriz_adjacencia = matriz_de_adjacencia('berlin52.tsp')
    colonia_formiga = ACO(1, 2, 0.5, 0.1, 100, 50)
    colonia_formiga.fit(matriz_adjacencia)
    caminho = colonia_formiga.get_caminho()
    distancia = colonia_formiga.get_distancia()
    print(f'Distância: {distancia}')
