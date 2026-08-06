import numpy as np

def lax_friedrichs(u_minus, u_plus, f_minus, f_plus, max_speed, **kwargs):
    """
    Fluxo Numérico de Lax-Friedrichs (Global ou Local).
    
    Parâmetros:
    ----------
    u_minus, u_plus : ndarray
        Estados da variável conservativa nas interfaces (esquerda/interna e direita/externa).
    f_minus, f_plus : ndarray
        Fluxos físicos avaliados em u_minus e u_plus.
    max_speed : float ou ndarray
        Velocidade máxima de propagação da onda (constante C).
    
    Retorna:
    -------
    ndarray
        O fluxo numérico avaliado na interface.
    """
    # Média dos fluxos físicos: {{f(u)}}
    f_avg = 0.5 * (f_minus + f_plus)
    
    # Termo dissipativo de salto (upwinding / estabilização): C/2 * [[u]]
    # Nota: O vetor normal (n) apontando para fora é embutido na construção 
    # das matrizes de Lift (Flk, Frk, etc) dentro da classe da Física.
    # Por convenção do método DG, o salto aqui é (u_minus - u_plus).
    dissipation = 0.5 * max_speed * (u_minus - u_plus) 
    
    return f_avg - dissipation


def roe(u_minus, u_plus, f_minus, f_plus, max_speed=None, **kwargs):
    """
    Esqueleto para o Fluxo Numérico de Roe.
    """
    raise NotImplementedError("O fluxo numérico de Roe será implementado aqui!")


def ldg_flux(u_minus, u_plus, q_minus, q_plus, beta=0.5, **kwargs):
    """
    Esqueleto para o fluxo numérico difusivo (Local Discontinuous Galerkin).
    """
    raise NotImplementedError("O fluxo numérico LDG será implementado aqui!")


def br1_flux(u_minus, u_plus, q_minus, q_plus, **kwargs):
    """
    Esqueleto para o fluxo numérico difusivo de Bassi-Rebay 1 (BR1).
    """
    raise NotImplementedError("O fluxo numérico BR1 será implementado aqui!")