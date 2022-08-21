import numpy as np


class ACO():
    def __init__(self, alpha: float, beta: float,
                 rho: float, tau_0: float, n_interacoes: int,
                 n_fromigas: int) -> None:
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.tau_0 = tau_0
        self.n_interacoes = n_interacoes
        self.n_formigas = n_fromigas
        self.melhor_caminho: list
        self.menor_distancia: float

    def _probabilidade(self, matriz_feromonios: np.ndarray,
                       matriz_eta: np.ndarray,
                       cidade_origem: int,
                       cidades_destino: list[int]) -> np.ndarray:
        """Calcula a probabilidade da formiga ir para cada cidade

        Args:
            matriz_feromonios (np.ndarray): Matriz de feromônios
            matriz_eta (np.ndarray): Matriz 1/matriz_adjacencia
            cidade_origem (int): Cidade atual da formiga
            cidades_destino (list): Cidades para onde a formiga pode ir

        Returns:
            np.ndarray: Vetor com a probabilidade relacionada a cada cidade na
            lista cidades_destino
        """

        num = (matriz_feromonios[cidade_origem,
                                 cidades_destino]**self.alpha) * \
            (matriz_eta[cidade_origem, cidades_destino]**self.beta)
        den = num.sum()

        return num/den

    def _escolhe_cidade(self, vetor_probabilidade: np.ndarray,
                        cidades_destino: list) -> int:
        """Escolhe a cidade com base o vetor_probabilidades e em um número aleatório

        Args:
            vetor_probabilidade (np.ndarray): Vetor com as probabilidades de
                                            cada cidade ser escolhida como destino.
            cidades_destino (list): Cidades de destino.

        Returns:
            int: Cidade de destino
        """
        n = np.random.rand(1, 1)
        i = 0
        for i in range(0, vetor_probabilidade.shape[0]):
            n = n-vetor_probabilidade[i]
            if n < 0:
                break
        return cidades_destino[i]

    def _insere_feromonio(self, matriz_adjacencia: np.ndarray,
                          matriz_feromonio: np.ndarray,
                          caminho: list,
                          distancia: float) -> np.ndarray:
        """Insere feromônio no caminho percorrido.

        Args:
            matriz_feromonio (np.ndarray): Matriz de feromônio.
            caminho (list): Caminho percorrido.
            distancia (float): Distância total do caminho percorrido.

        Returns:
            np.ndarray: Matriz de feromônio atualizada.
        """
        for i in range(0, len(caminho)-1):
            matriz_feromonio[caminho[i], caminho[i+1]] += 1 / \
                matriz_adjacencia[caminho[i], caminho[i+1]]
            matriz_feromonio[caminho[i+1], caminho[i]] += 1 / \
                matriz_adjacencia[caminho[i+1], caminho[i]]
        return matriz_feromonio

    def _distancia_caminho(self, matriz_adjacencia: np.ndarray,
                           caminho: list) -> float:
        """Calcula a distância percorrida.

        Args:
            matriz_adjacencia (np.ndarray): Matriz de adjacência
            caminho (list): Caminho percorrido.

        Returns:
            float: Distância percorrida.
        """
        distancia = 0
        for i in range(0, len(caminho)-1):
            distancia += matriz_adjacencia[caminho[i], caminho[i+1]]
        return distancia

    def _formiga(self, matriz_adjacencia: np.ndarray,
                 matriz_eta: np.ndarray,
                 matriz_feromonios: np.ndarray) -> tuple:
        """Função que define a rota da formiga.

        Args:
            matriz_adjacencia (np.ndarray): Matriz de adjacência.
            matriz_eta (np.ndarray): Matriz 1/matriz_adjacencia.
            matriz_feromonios (np.ndarray): Matriz de feromônio.

        Returns:
            tuple: Distância e o caminho percorrido.
        """
        n_cidades = matriz_adjacencia.shape[0]
        cidades_destinos = list(range(1, n_cidades))
        caminho = [0]

        while len(cidades_destinos) > 1:
            cidades_probabilidade = self._probabilidade(matriz_adjacencia,
                                                        matriz_feromonios,
                                                        matriz_eta,
                                                        caminho[-1],
                                                        cidades_destinos)
            cidade_destino = self._escolhe_cidade(cidades_probabilidade,
                                                  cidades_destinos)
            cidades_destinos.remove(cidade_destino)
            caminho.append(cidade_destino)

        caminho.append(cidades_destinos[0])
        caminho.append(0)

        distancia: float = self._distancia_caminho(matriz_adjacencia,
                                                   caminho)

        return distancia, caminho

    def fit(self, matriz_adjacencia: np.ndarray) -> None:
        """Encontra o caminho com o custo satisfatório.

        Args:
            matriz_adjacencia (np.ndarray): Matriz de adjacência.
        """
        matriz_feromonio = np.zeros(matriz_adjacencia.shape) + self.tau_0
        matriz_eta = 1/matriz_adjacencia
        self.menor_distancia = np.inf

        for i in range(0, self.n_interacoes):
            for j in range(0, self.n_formigas):
                distancia, caminho = self._formiga(matriz_adjacencia,
                                                   matriz_eta,
                                                   matriz_feromonio)

                self._insere_feromonio(matriz_adjacencia,
                                       matriz_feromonio,
                                       caminho,
                                       distancia)

                if(distancia < self.menor_distancia):
                    self.menor_distancia = distancia
                    self.melhor_caminho = caminho

            # Evapor feromonios
            matriz_feromonio = (1-self.rho)*matriz_feromonio

    def get_caminho(self) -> list:
        """Retorna o melhor caminho encontrado.

        Returns:
            list: Caminho encontrado.
        """
        return self.melhor_caminho

    def get_distancia(self) -> float:
        """Retorna a menor distância encontrada.

        Returns:
            float: Menor distância
        """
        return self.menor_distancia
