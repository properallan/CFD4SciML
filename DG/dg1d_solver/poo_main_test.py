import numpy as np
import matplotlib.pyplot as plt
import math

# Importando a biblioteca
from dg1d_core import DGSpace1D
from dg1d_physics import ConservationLaw
from dg1d_limiters import HierarchicalLimiter
import dg1d_integrators
from dg1d_postprocessing import create_gif

# =================================================================
# 1. DEFINIÇÃO DA FÍSICA ESPECÍFICA (O CÓDIGO DO USUÁRIO)
# =================================================================
class InviscidBurgers(ConservationLaw):
    def __init__(self, space):
        # Chama o construtor da base passando o espaço
        super().__init__(space, num_flux_name='lax_friedrichs')

    def physical_flux(self, u):
        """ f(u) = u^2 / 2 """
        return (u**2) / 2.0

    def max_wave_speed(self, u):
        """ C = max|u| """
        return np.max(np.abs(u))

    def apply_bc(self, ut_list, uL_list, uR_list, t):
        """ Aplica condições de contorno preenchendo as Ghost Cells """
        ut = ut_list[0]
        uL = uL_list[0]
        
        # Dirichlet na entrada (x = 0)
        ut[0, 0] = -uL + 2.0 * (0.5 + np.sin(-2.0 * np.pi * uL * t))
        # Saída (cópia da entrada para este problema de teste)
        ut[0, -1] = ut[0, 0]

# =================================================================
# 2. CONFIGURAÇÃO DA SIMULAÇÃO
# =================================================================
K = 250
N = 4
tmin, tmax = 0.0, 0.5
Nsteps = 62
dt = 0.008

# Instancia o Palco (Matemática e Geometria)
space = DGSpace1D(K=K, N=N, xmin=0.0, xmax=1.0, quad_type='GL')

# Instancia o Ator (A Física)
problema = InviscidBurgers(space)

# O limitador nasce recebendo o espaço!
meu_limitador = HierarchicalLimiter(space)

# =================================================================
# 4. CONDIÇÃO INICIAL E LOOP NO TEMPO
# =================================================================
uic = 0.5 + np.sin(2.0 * np.pi * space.xc)
b = np.dot(space.wi * space.psi.T, uic)
uhat_initial = space.InvM[:, None] * b

U_modal = [uhat_initial.copy()]
u_history = np.zeros((space.nip * space.K, Nsteps + 1))
u_history[:, 0] = np.dot(space.psi, uhat_initial).T.flatten()

t = tmin
print("Iniciando simulação...")
for n in range(1, Nsteps + 1):
    # Passamos o método compute_rhs do NOSSO OBJETO 'problema' para o integrador
    U_modal = dg1d_integrators.RKSSP104_Step(U_modal, t, dt, problema.compute_rhs, meu_limitador)
    #U_modal = dg1d_integrators.RKSSP104_Step(U_modal, t, dt, problema.compute_rhs) 
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
    folder="burgers_animation",
    filename="simulation.gif",
    title="Inviscid Burgers",
    xlabel="x",
    ylabel="u",
    xlim=(0.0, 1.0),
    ylim=(-0.8, 1.8),
    duration=80,
    plot_kwargs={
        "color": "tab:blue",
        "lw": 2,
    },
)