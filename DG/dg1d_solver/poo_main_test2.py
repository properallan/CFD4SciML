import numpy as np
import matplotlib.pyplot as plt
import math

# Importando a biblioteca
from dg1d_core import DGSpace1D
from dg1d_physics import ConservationLaw
from dg1d_limiters import HierarchicalLimiter, SlopeLimiterN
import dg1d_integrators
from dg1d_postprocessing import create_gif

class LinearAdvection(ConservationLaw):
    def __init__(self, space, a=1.0):
        super().__init__(space, num_flux_name='lax_friedrichs')
        self.a = a  # Velocidade de advecção (ex: a = 1.0)

    def physical_flux(self, u):
        """ f(u) = a * u """
        return self.a * u

    def max_wave_speed(self, u):
        """ A velocidade máxima da onda é simplesmente o próprio 'a' """
        return np.abs(self.a)

    def apply_bc(self, ut_list, uL_list, uR_list, t):
        """ Condição de contorno periódica no intervalo [-1, 1] """
        ut = ut_list[0]
        # O valor que sai pela direita (-1) entra na esquerda (1) e vice-versa
        ut[0, 0] = ut[0, -2]   # Borda esquerda recebe o valor antes da borda direita real
        ut[0, -1] = ut[0, 1]   # Borda direita recebe o valor após a borda esquerda real

K = 50
N = 1  # P = 1 (ou N = 1, dependendo de como definiu o grau do polinômio)
xmin, xmax = -1.0, 1.0
tmin, tmax = 0.0, 10
Nsteps = 200
dt = 0.027

# Instancia o Palco (Matemática e Geometria)
space = DGSpace1D(K=K, N=N, xmin=xmin, xmax=xmax, quad_type='GL')

# Instancia o Ator (A Física)
problema = LinearAdvection(space)

# O limitador nasce recebendo o espaço!
# meu_limitador = HierarchicalLimiter(space)
meu_limitador = SlopeLimiterN(space)

# =================================================================
# 4. CONDIÇÃO INICIAL E LOOP NO TEMPO
# =================================================================
uic = np.sin(np.pi * space.xc)
b = np.dot(space.wi * space.psi.T, uic)
uhat_initial = space.InvM[:, None] * b

U_modal = [uhat_initial.copy()]
u_history = np.zeros((space.nip * space.K, Nsteps + 1))
u_history[:, 0] = np.dot(space.psi, uhat_initial).T.flatten()

t = tmin
print("Iniciando simulação...")
for n in range(1, Nsteps + 1):
    # Passamos o método compute_rhs do NOSSO OBJETO 'problema' para o integrador
    #U_modal = dg1d_integrators.RKSSP104_Step(U_modal, t, dt, problema.compute_rhs, meu_limitador)
    #U_modal = dg1d_integrators.RKSSP54_Step(U_modal, t, dt, problema.compute_rhs)
    U_modal = dg1d_integrators.RKSSP54_Step(U_modal, t, dt, problema.compute_rhs, meu_limitador) 
    t += dt
    
    # Salva histórico
    uh = np.dot(space.psi, U_modal[0]).T.flatten()
    u_history[:, n] = uh

# =================================================================
# 5. PLOT DOS RESULTADOS
# =================================================================
# plt.figure(figsize=(8,6))
# x = space.xc.T.flatten()
# plt.plot(x, u_history[:, 1], label='tstep = 1')
# plt.plot(x, u_history[:, 31], label='tstep = 31')
# plt.plot(x, u_history[:, 62], label='tstep = 62')
# plt.ylim(-0.8, 1.8)
# plt.xlim(0.0, 1.0)
# plt.legend()
# plt.title("POO Solver DG1D: Inviscid Burgers")
# plt.grid(True)
# plt.show()


x = space.xc.T.flatten()
create_gif(
    x=x,
    u_history=u_history,
    dt=dt,
    folder="advection_animation_wSL",
    filename="simulation.gif",
    title="Advection",
    xlabel="x",
    ylabel="u",
    xlim=(-1.0, 1.0),
    ylim=(-1.5, 1.5),
    duration=100,
    plot_kwargs={
        "color": "tab:blue",
        "lw": 2,
    },
)