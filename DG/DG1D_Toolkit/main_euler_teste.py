import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as lg
from scipy.optimize import fsolve
import math
# -----------------------------------------------------------------------------
# IMPORTAÇÕES DO SEU TOOLKIT
# -----------------------------------------------------------------------------
from dg1d_core import DGSpace1D
from dg1d_toolkit.dg1d_integrators import RKSSP54_Step
from dg1d_toolkit.dg1d_limiters import HierarchicalLimiter

# -----------------------------------------------------------------------------
# CONFIGURAÇÕES DO PROBLEMA (Tubo de Choque de Sod)
# -----------------------------------------------------------------------------
K = 180         
N = 3           
gamma = 1.4
xmin, xmax = 0.0, 1.0
tmin, tmax = 0.0, 0.2
dt = 0.000025
Nsteps = 8000

# -----------------------------------------------------------------------------
# FUNÇÕES FÍSICAS (EULER)
# -----------------------------------------------------------------------------
def get_pressure(rho, rhou, Ener):
    return (gamma - 1.0) * (Ener - 0.5 * (rhou**2) / rho)

def FluxProjection_Euler(space, rhot, rhout, Enert):
    """ Projeção de Fluxo específica para o sistema de Euler """
    rho_h = np.dot(space.nlpsi, rhot[:, 1:-1])
    rhou_h = np.dot(space.nlpsi, rhout[:, 1:-1])
    Ener_h = np.dot(space.nlpsi, Enert[:, 1:-1])
    
    pres = get_pressure(rho_h, rhou_h, Ener_h)
    
    rhof = rhou_h
    rhouf = (rhou_h**2) / rho_h + pres
    Enerf = (Ener_h + pres) * rhou_h / rho_h
    
    rhoft = np.zeros((space.Nldof, space.K + 2))
    rhouft = np.zeros((space.Nldof, space.K + 2))
    Enerft = np.zeros((space.Nldof, space.K + 2))
    
    ProjMat = space.nlwi * space.nlpsi.T
    rhoft[:, 1:-1] = space.InvM[:, None] * np.dot(ProjMat, rhof)
    rhouft[:, 1:-1] = space.InvM[:, None] * np.dot(ProjMat, rhouf)
    Enerft[:, 1:-1] = space.InvM[:, None] * np.dot(ProjMat, Enerf)
    
    return rhoft, rhouft, Enerft

# -----------------------------------------------------------------------------
# O NOVO Lh LIMPO (Padrão Toolkit)
# -----------------------------------------------------------------------------
def Lh_euler(U_list, t, space):
    rhohat, rhouhat, Enerhat = U_list[0], U_list[1], U_list[2]
    
    # Recupera espaço nodal
    rhoh = np.dot(space.psi, rhohat)
    rhouh = np.dot(space.psi, rhouhat)
    Enerh = np.dot(space.psi, Enerhat)
    
    # Velocidade da Onda Local (Rusanov / Lax-Friedrichs local)
    pres = get_pressure(rhoh, rhouh, Enerh)
    cvel = np.sqrt(gamma * pres / rhoh)
    lm = np.abs(rhouh / rhoh) + cvel
    
    C = np.zeros(space.K + 1)
    C[1:-1] = np.maximum(lm[-1, :-1], lm[0, 1:])
    C[0], C[-1] = lm[0, 0], lm[-1, -1]
    
    # Aloca matrizes acolchoadas
    rhot = np.zeros((space.Nldof, space.K + 2))
    rhout = np.zeros((space.Nldof, space.K + 2))
    Enert = np.zeros((space.Nldof, space.K + 2))
    
    rhot[:, 1:-1] = rhohat
    rhout[:, 1:-1] = rhouhat
    Enert[:, 1:-1] = Enerhat
    
    # Condições de Contorno de Dirichlet
    rhohL, rhohR = np.dot(space.Flkp1[0, :], rhot[:, 1]), np.dot(space.Frk[0, :], rhot[:, -2])
    rhouhL, rhouhR = np.dot(space.Flkp1[0, :], rhout[:, 1]), np.dot(space.Frk[0, :], rhout[:, -2])
    EnerhL, EnerhR = np.dot(space.Flkp1[0, :], Enert[:, 1]), np.dot(space.Frk[0, :], Enert[:, -2])
    
    pin, pout = 1.0, 0.1
    pint = -pres[0, 0] + 2.0 * pin
    poutt = -pres[-1, -1] + 2.0 * pout
    
    rhot[0, 0] = -rhohL + 2.0 * 1.0
    rhot[0, -1] = -rhohR + 2.0 * 0.125
    rhout[0, 0] = -rhouhL + 2.0 * 0.0
    rhout[0, -1] = -rhouhR + 2.0 * 0.0
    Enert[0, 0] = -EnerhL + 2.0 * (pin / (gamma - 1.0))
    Enert[0, -1] = -EnerhR + 2.0 * (pout / (gamma - 1.0))
    
    # Projeção de Fluxos Físicos
    rhoft, rhouft, Enerft = FluxProjection_Euler(space, rhot, rhout, Enert)
    
    # Contorno nos fluxos
    rhoft[0, 0] = rhout[0, 0]
    rhoft[0, -1] = rhout[0, -1]
    rhouft[0, 0] = (rhout[0, 0]**2) / rhot[0, 0] + pint
    rhouft[0, -1] = (rhout[0, -1]**2) / rhot[0, -1] + poutt
    Enerft[0, 0] = (pint / (gamma - 1.0) + 0.5 * (rhout[0, 0]**2) / rhot[0, 0] + pint) * rhout[0, 0] / rhot[0, 0]
    Enerft[0, -1] = (poutt / (gamma - 1.0) + 0.5 * (rhout[0, -1]**2) / rhot[0, -1] + poutt) * rhout[0, -1] / rhot[0, -1]
    
    # Montagem Numérica Limpa (Aplicando LF para cada equação do sistema)
    def apply_LF(ut, ft):
        return space.InvM[:, None] * space.J[:]**(-1) * (
            np.dot(space.S.T, ft[:, 1:-1])
            - 0.5 * (np.dot(space.Flkp1, ft[:, 2:]) - C[1:] * np.dot(space.Flkp1, ut[:, 2:]))
            - 0.5 * (np.dot(space.Frk, ft[:, 1:-1]) + C[1:] * np.dot(space.Frk, ut[:, 1:-1]))
            + 0.5 * (np.dot(space.Frkm1, ft[:, :-2]) + C[:-1] * np.dot(space.Frkm1, ut[:, :-2]))
            + 0.5 * (np.dot(space.Flk, ft[:, 1:-1]) - C[:-1] * np.dot(space.Flk, ut[:, 1:-1]))
        )
        
    rhs_rho = apply_LF(rhot, rhoft)
    rhs_rhou = apply_LF(rhout, rhouft)
    rhs_ener = apply_LF(Enert, Enerft)
    
    return [rhs_rho, rhs_rhou, rhs_ener]

# -----------------------------------------------------------------------------
# LOOP PRINCIPAL
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    space = DGSpace1D(K=K, N=N, xmin=xmin, xmax=xmax, quad_type='GL')
    
    # CI
    idx = int(math.floor(K / 2))
    rho_x = np.ones((space.nip, K))
    rho_x[:, idx:] = 0.125
    rhou_x = np.zeros((space.nip, K))
    ener_x = (1.0 / (gamma - 1.0)) * np.ones((space.nip, K))
    ener_x[:, idx:] = 0.1 * ener_x[:, idx:]
    
    def project_IC(field):
        invdiag = 1.0 / (np.sum(space.wi * space.psi.T**2, axis=1))
        b = np.dot(space.wi * space.psi.T, field)
        return (invdiag * b.T).T

    U_modal = [project_IC(rho_x), project_IC(rhou_x), project_IC(ener_x)]
    limiter = HierarchicalLimiter(space)
    
    t = tmin
    print("Iniciando Sod Shock Tube (Euler)...")
    for n in range(1, Nsteps + 1):
        RHS_func = lambda U_list, time: Lh_euler(U_list, time, space)
        U_modal = RKSSP54_Step(U_modal, t, dt, RHS_func, limiter)
        t += dt

    # -------------------------------------------------------------------------
    # SOLUÇÃO ANALÍTICA E PLOTS (Do seu professor)
    # -------------------------------------------------------------------------
    PrL, PrR, rhoL, rhoR = 1.0, 0.1, 1.0, 0.125
    uL, uR = 0.0, 0.0
    gammaA, gammaC = gamma - 1.0, gamma + 1.0
    cR, cL = np.sqrt(gamma * PrR / rhoR), np.sqrt(gamma * PrL / rhoL)
    
    def func(p34):
        wortel = np.sqrt(2. * gamma * (gammaA + gammaC * p34))
        yy = (gammaA * (cR / cL) * (p34 - 1.)) / wortel
        return (yy / p34 - (PrR / PrL)) + ((1. - yy)**(2. * gamma / gammaA)) / p34 - (PrR / PrL) # Simplificação da equação
    
    p34 = fsolve(func, 3.0)[0]
    p3 = p34 * PrR
    alpha = gammaC / gammaA
    rho3 = rhoR * (1. + alpha * p34) / (alpha + p34)
    rho2 = rhoL * (p34 * PrR / PrL)**(1 / gamma)
    u2 = uL - uR + (2. / gammaA) * cL * (1. - (p34 * PrR / PrL)**(gammaA / (2. * gamma)))
    c2 = np.sqrt(gamma * p3 / rho2)
    
    spos = 0.5 + tmax * cR * np.sqrt(gammaA / (2. * gamma) + gammaC / (2. * gamma) * p34) + tmax * uR
    conpos = 0.5 + u2 * tmax + tmax * uR
    pos1 = 0.5 + (uL - cL) * tmax
    pos2 = 0.5 + (u2 + uR - c2) * tmax
    
    xgrid = space.xc.flatten('F')
    rhoE, uE, PrE = np.zeros_like(xgrid), np.zeros_like(xgrid), np.zeros_like(xgrid)
    
    for i, x in enumerate(xgrid):
        if x <= pos1:
            PrE[i], rhoE[i], uE[i] = PrL, rhoL, uL
        elif x <= pos2:
            PrE[i] = PrL * (1. + (pos1 - x) / (cL * alpha * tmax))**(2. * gamma / gammaA)
            rhoE[i] = rhoL * (1. + (pos1 - x) / (cL * alpha * tmax))**(2. / gammaA)
            uE[i] = uL + (2. / gammaC) * (x - pos1) / tmax
        elif x <= conpos:
            PrE[i], rhoE[i], uE[i] = p3, rho2, u2 + uR
        elif x <= spos:
            PrE[i], rhoE[i], uE[i] = p3, rho3, u2 + uR
        else:
            PrE[i], rhoE[i], uE[i] = PrR, rhoR, uR

    # Extrai físicas finais numéricas
    rho_final = np.dot(space.psi, U_modal[0]).flatten('F')
    rhou_final = np.dot(space.psi, U_modal[1]).flatten('F')
    ener_final = np.dot(space.psi, U_modal[2]).flatten('F')
    u_final = rhou_final / rho_final
    pres_final = get_pressure(rho_final, rhou_final, ener_final)
    
    # Plota Densidade
    plt.figure(1, figsize=(8, 6), dpi=100)
    plt.plot(xgrid, rhoE, 'r-', label='Analítica')
    plt.plot(xgrid, rho_final, 'b--', label='DG1D Toolkit')
    plt.title("Densidade (Sod Shock Tube)")
    plt.legend()
    
    # Plota Velocidade
    plt.figure(2, figsize=(8, 6), dpi=100)
    plt.plot(xgrid, uE, 'r-', label='Analítica')
    plt.plot(xgrid, u_final, 'b--', label='DG1D Toolkit')
    plt.title("Velocidade")
    
    # Plota Pressão
    plt.figure(3, figsize=(8, 6), dpi=100)
    plt.plot(xgrid, PrE, 'r-', label='Analítica')
    plt.plot(xgrid, pres_final, 'b--', label='DG1D Toolkit')
    plt.title("Pressão")
    plt.show()