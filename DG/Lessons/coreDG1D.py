import numpy as np
import matplotlib.pyplot as plt
import math
from mathDG1D import JacobiP, jacobi_gauss_quad, DMatrix1D

def MeshGen1D(K: int, xmin: float = 0.0, xmax: float = 1.0): 
    """ 
    Gerador de uma malha simples equidistante com K elementos.

    ### Parâmetros:
        K (int): Número de elementos da malha.
        xmin (float): Limite inferior do domínio real.
        xmax (float): Limite superior do domínio real.

    ### Retorna:
        Nv (int): Número total de vértices (nós).
        VX (array): Vetor de coordenadas globais dos vértices.
        EToV (array): Matriz de conectividade (K x 2) contendo os índices 
                      dos vértices que compõem cada elemento k.
    """
    Nv = K + 1  # Número total de vértices da malha
    VX = np.linspace(xmin, xmax, Nv) # Criação do vetor de coordenadas

    # Matriz de conectividade (Element to Vertex)
    EToV = np.zeros((K, 2), dtype=int) 
    for k in range(K): 
        EToV[k, 0] = k 
        EToV[k, 1] = k + 1

    return Nv, VX, EToV

import numpy as np

def LegendreBasis(P: int, xi: np.ndarray) -> np.ndarray:
    """
    Constrói a matriz de avaliação da base ortogonal de Legendre.

    Parâmetros:
        P (int): Grau máximo do Polinômio Aproximador.
        xi (array): Vetor com os pontos (coordenadas) a serem avaliados.

    Retorna:
        Lj (ndarray): Matriz (npts, P+1) contendo as funções de base de 
                      grau 0 a P avaliadas nos pontos xi.
    """
    Q = P + 1        # Número total de funções de base
    npts = len(xi)   # Número de pontos de avaliação
    
    # Aloca a matriz para armazenar a base de Legendre
    Lj = np.zeros((npts, Q)) 
    
    # Avaliação da base de Legendre (alpha=0, beta=0)
    for j in range(Q):          # Itera sobre o grau do polinômio (0 até P)
        for i in range(npts):   # Itera sobre os pontos da quadratura
            Lj[i, j] = JacobiP(xi[i], j, 0, 0)
            
    return Lj


def Jacobian(xi: np.ndarray, VX: np.ndarray) -> np.ndarray:
    """
    Computes the Jacobian and maps quadrature points from the
    reference element [-1,1] to every physical element.

    Parameters
    ----------
    xi : ndarray (Nq,)
        Quadrature points in the reference element.
    VX : ndarray (K+1,)
        Coordinates of the mesh vertices.

    Returns
    -------
    xcoord : ndarray (Nq, K)
        Physical coordinates of every quadrature point in every element.
    J : ndarray (K,)
        Jacobian of each element.
    """
    J = 0.5 * (VX[1:] - VX[:-1])
    B = 0.5 * (VX[1:] + VX[:-1])
    xcoord = xi[:, None] * J[None, :] + B[None, :]

    return xcoord, J

def MassMatrix(Nldof: int, 
               method: str = 'analytic', wi: np.ndarray = None, psi: np.ndarray = None) -> np.ndarray:
    """
    Constrói a diagonal da Matriz de Massa e sua respectiva Inversa.
    
    Parâmetros:
        Nldof (int): Número de graus de liberdade locais (Grau P + 1). Obrigatório.
        method (str): 'analytic' (padrão) ou 'quadrature'.
        wi (array): Pesos da quadratura. Obrigatório se method='quadrature'.
        psi (ndarray): Base de Legendre avaliada. Obrigatório se method='quadrature'.

    Retorna:
        diagM (array): Vetor contendo a diagonal principal da Matriz de Massa.
        invDiagM (array): Vetor contendo a inversa da diagonal principal.
    """
    diagM = np.zeros(Nldof)
    invDiagM = np.zeros(Nldof)
    
    if method == 'analytic':
        for n in range(Nldof):
            # Solução exata da integral do Polinômio de Legendre ao quadrado
            diagM[n] = 2.0 / (2.0 * n + 1.0)
            invDiagM[n] = 1.0 / diagM[n]
            
    elif method == 'quadrature':
        # Trava de segurança: verifica se os dados foram passados
        if wi is None or psi is None:
            raise ValueError("Erro: Para o método 'quadrature', você deve fornecer os argumentos 'wi' e 'psi'!")
            
        for i in range(Nldof):
            # Solução via integração numérica
            diagM[i] = np.sum(wi * (psi[:, i]**2))
            invDiagM[i] = 1.0 / diagM[i]
            
    else:
        raise ValueError("Erro: Método desconhecido. Escolha 'analytic' ou 'quadrature'.")
        
    return diagM, invDiagM

def StiffMatrix(phi: np.ndarray, Dphi: np.ndarray, wi: np.ndarray) -> np.ndarray:
    """
    Matriz de Rigidez (Advecção)

    ### Args:
    wi (array): Vetor de pesos da quadratura escolhida
    phi (array): Matriz do Polinômio Base avaliada em xi
    Dphi (array): Matriz de Derivadas do Polinomio Base avaliado em xi

    ### Return
    Retorna a matriz de Rigidez (advecção) do sistema

    """
    #Computes local stiffness matrix
    Sij = np.dot(wi*phi.T,Dphi)

    return Sij

def LiftMatrix(Nldof: int):
    """
    Constrói as Matrizes de Elevação (Lift Matrices) avaliando os Polinômios
    de Legendre nas fronteiras do elemento de referência (xi = -1 e xi = 1).
    
    Parâmetros:
        Nldof (int): Número de graus de liberdade locais (Grau P + 1).
        
    Retorna:
        Flk (ndarray): Matriz da face esquerda do elemento k.
        Frk (ndarray): Matriz da face direita do elemento k.
        Flkp1 (ndarray): Matriz de interação com o vizinho da direita.
        Frkm1 (ndarray): Matriz de interação com o vizinho da esquerda.
    """
    # Aloca as matrizes inicialmente com o valor 1 (assumindo P_n(1) = 1)
    Frk = np.ones((Nldof, Nldof))
    Flk = np.ones((Nldof, Nldof))
    Flkp1 = np.ones((Nldof, Nldof))
    Frkm1 = np.ones((Nldof, Nldof))
    
    # Aplica a propriedade P_n(-1) = (-1)^n
    # O slicing [1:Nldof:2] acessa exatamente as colunas/linhas de grau ímpar (1, 3, 5...)
    Flkp1[:, 1:Nldof:2] = -1.0 * Flkp1[:, 1:Nldof:2]
    Frkm1[1:Nldof:2, :] = -1.0 * Frkm1[1:Nldof:2, :]
    
    # Para a matriz Flk, a alternância ocorre tanto nas linhas quanto nas colunas
    Flk[:Nldof:2, 1:Nldof:2] = -1.0 * Flk[:Nldof:2, 1:Nldof:2]
    Flk[1:Nldof:2, :Nldof:2] = -1.0 * Flk[1:Nldof:2, :Nldof:2]
    
    return Flk, Frk, Flkp1, Frkm1

def ModifStiffMatrix(psi: np.ndarray, Dpsi: np.ndarray, wi: np.ndarray, epst: np.ndarray,
    type_form: str = 'Warburton',
    Frk: np.ndarray = None,
    Flk: np.ndarray = None,
    epsb: np.ndarray = None
) -> tuple:
    """
    Calcula a Matriz de Rigidez Modificada para o termo difusivo.

    Parâmetros:
    ----------
    psi : np.ndarray  
        Matriz com as funções de base avaliadas nos pontos de quadratura.
    Dpsi : np.ndarray
        Matriz com as derivadas das funções de base avaliadas nos pontos de quadratura.
    wi : np.ndarray
        Vetor contendo os pesos da quadratura escolhida.
    epst : np.ndarray
        Vetor contendo a viscosidade avaliada nos pontos internos do volume do elemento.
    type_form : str, opcional
        Define a formulação matemática a ser utilizada:
        - 'Warburton' (padrão): Abordagem Simétrica (Hesthaven-Warburton). Retorna Ssq1 e Ssq2.
        - 'Persson': Abordagem Assimétrica (Persson-Peraire). Retorna Ssq1 e None.
    Frk : np.ndarray, opcional
        Lift Matrix da face direita (Face Right k). Obrigatório se type_form='Warburton'.
    Flk : np.ndarray, opcional
        Lift Matrix da face esquerda (Face Left k). Obrigatório se type_form='Warburton'.
    epsb : np.ndarray, opcional
        Vetor contendo a viscosidade avaliada estritamente nas fronteiras do elemento. 
        Obrigatório se type_form='Warburton'.

    Retorna:
    -------
    tuple
        Uma tupla contendo as matrizes de rigidez modificadas transpostas:
        (Ssq1.T, Ssq2.T) para 'Warburton' ou (Ssq1.T, None) para 'Persson'.
    """
    
    if type_form == 'Warburton':
        # Trava de segurança para garantir que os operadores de fronteira foram fornecidos
        if Frk is None or Flk is None or epsb is None:
            raise ValueError(
                "Erro: Para a abordagem 'Warburton', os argumentos 'Frk', 'Flk' e 'epsb' "
                "não podem ser vazios."
            )

        # Abordagem Simétrica (Hesthaven-Warburton)
        Ssq1 = np.dot(np.sqrt(epst) * wi * psi.T, Dpsi)

        # Termo de fronteira acoplado aos Liftings para construção de Ssq2
        Boundterm = (np.sqrt(epsb[-1]) * Frk) - (np.sqrt(epsb[0]) * Flk)
        Ssq2 = Boundterm - Ssq1.T

        return Ssq1.T, Ssq2.T

    elif type_form == 'Persson':
        # Abordagem Assimétrica (Persson-Peraire)
        # O modelo de Persson absorve toda a viscosidade sem quebra simétrica
        Ssq1 = np.dot(epst * wi * psi.T, Dpsi)

        # Retorna Ssq1 e dispensa a necessidade de uma segunda matriz
        return Ssq1.T, None

    else:
        raise ValueError("Modelo de matriz modificada não reconhecido! Escolha 'Warburton' ou 'Persson'.")



def FluxProjection(ut_list: list, FluxFunc: callable, nlpsi: np.ndarray, nlwi: np.ndarray, InvM: np.ndarray,):
    """
    Projeta o fluxo físico no espaço modal através de uma projeção L2.

    Esta implementação é genérica e suporta um número arbitrário de
    equações (Burgers, Euler, Navier-Stokes, etc.).

    Parâmetros
    ----------
    ut_list : list of ndarray
        Lista contendo as variáveis conservativas. Cada array deve possuir
        formato (Nldof, K+2) ou (Nldof, K+2, nstage).
    FluxFunc : callable
        Função responsável por calcular os fluxos físicos.
    nlpsi : ndarray
        Base modal avaliada nos pontos da quadratura de superintegração.
    nlwi : ndarray
        Pesos da quadratura de superintegração.
    InvM : ndarray
        Diagonal inversa da matriz de massa.

    Retorna
    -------
    ndarray ou list(ndarray)

        Burgers:
            fhat

        Sistemas:
            [fhat1, fhat2, ...]
    """

    # Remove células fantasmas (ghost cells)
    uhat_list = []

    for ut in ut_list:

        if ut.ndim == 3:
            uhat = ut[:, 1:-1, 0]
        elif ut.ndim == 2:
            uhat = ut[:, 1:-1]
        else:
            raise ValueError(
                "Cada variável deve possuir dimensão 2 ou 3."
            )

        uhat_list.append(uhat)

    uh_list = [nlpsi @ uhat for uhat in uhat_list]   # Reconstrução da solução física
    fh_list = FluxFunc(*uh_list)   # Avaliação dos fluxos físicos

    # Burgers retorna apenas um array
    if not isinstance(fh_list, (list, tuple)):
        fh_list = [fh_list]

    # Verificação de consistência
    if len(fh_list) != len(uh_list):
        raise ValueError(
            f"FluxFunc retornou {len(fh_list)} fluxo(s), "
            f"mas existem {len(uh_list)} variável(is)."
        )

    # Projeção L2
    ProjectionMatrix = nlwi * nlpsi.T
    fhat_list = []

    for fh in fh_list:
        b = ProjectionMatrix @ fh
        fhat = InvM[:, None] * b
        fhat_list.append(fhat)

    # Compatibilidade com problemas escalares
    if len(fhat_list) == 1:
        return fhat_list[0]

    return fhat_list