import numpy as np

def lax_friedrichs(space, ut, ft, max_speed=None):
    """
    Calcula os fluxos numéricos centrais (Lax-Friedrich) nas interfaces.
    """
    if max_speed == None:
        max_speed = 1

    flux_num = (
            - 0.5 * np.dot(space.Flkp1, (ft[:, 2:] - max_speed * ut[:, 2:]))
            - 0.5 * np.dot(space.Frk,   (ft[:, 1:-1] + max_speed * ut[:, 1:-1]))
            + 0.5 * np.dot(space.Frkm1, (ft[:, :-2] + max_speed * ut[:, :-2]))
            + 0.5 * np.dot(space.Flk,   (ft[:, 1:-1] - max_speed * ut[:, 1:-1]))
        )

    return flux_num

def br1_flux(space, field, epst):
    """
    Calcula os fluxos numéricos centrais (BR1) nas interfaces.
    'field' pode ser a matriz 'ut' (para calcular q) 
    ou a matriz 'qt' (para calcular a difusão final).
    """
    # Extraímos a viscosidade das células correspondentes (Esquerda, Centro, Direita)
    e_C = np.sqrt(epst[0, 1:-1])
    e_R = np.sqrt(epst[0, 2:])
    e_L = np.sqrt(epst[0, :-2])

    flux_num = (
        + 0.5 * (e_R * np.dot(space.Flkp1, field[:, 2:]) + e_C * np.dot(space.Frk, field[:, 1:-1]))
        - 0.5 * (e_L * np.dot(space.Frkm1, field[:, :-2]) + e_C * np.dot(space.Flk, field[:, 1:-1]))
    )
    
    return flux_num