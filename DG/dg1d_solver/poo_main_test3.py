import numpy as np
import matplotlib.pyplot as plt
import math

# Importando a biblioteca
from dg1d_core import DGSpace1D
from dg1d_physics import ConservationLaw
from dg1d_limiters import HierarchicalLimiter, SlopeLimiterN
import dg1d_integrators
from dg1d_postprocessing import create_gif

class Example56Burgers(ConservationLaw):
    def __init__(self, space):
        super().__init__(space, num_flux_name='lax_friedrichs')

    def physical_flux(self, u):
        # ATENÇÃO: O fluxo aqui é u^2 inteiro!
        return u**2 

    def max_wave_speed(self, u):
        # Como f(u) = u^2, a velocidade da onda é f'(u) = 2u
        return np.max(np.abs(2.0 * u))

    def apply_bc(self, ut_list, uL_list, uR_list, t):
        ut = ut_list[0]
        
        # A imagem diz para usar a solução exata u0(x - 3t) como contorno.
        # Avaliamos a posição virtual (x - 3t) nas duas bordas (-1 e 1)
        x_left = -1.0 - 3.0 * t
        ut[0, 0] = 2.0 if x_left <= -0.5 else 1.0
        
        x_right = 1.0 - 3.0 * t
        ut[0, -1] = 2.0 if x_right <= -0.5 else 1.0

# =================================================================
# 2. CONFIGURAÇÃO DA SIMULAÇÃO
# =================================================================
# K = 20           # Conforme a imagem
# N = 1            # Conforme a imagem
# xmin, xmax = -1.0, 1.0  # Novo domínio espacial
# tmin = 0.0
# dt = 0.0075 
# Nsteps = 60     # Ajuste conforme o tempo final desejado
K = 50           # Conforme a imagem
N = 1            # Conforme a imagem
xmin, xmax = -1.0, 1.0  # Novo domínio espacial
tmin = 0.0
dt = 0.003 
Nsteps = 150     # Ajuste conforme o tempo final desejado

# Instancia o Palco e o Ator
space = DGSpace1D(K=K, N=N, xmin=xmin, xmax=xmax, quad_type='GL')
problema = Example56Burgers(space)
# meu_limitador = HierarchicalLimiter(space) # Aplique o limitador que preferir
meu_limitador = SlopeLimiterN(space)

# =================================================================
# 4. CONDIÇÃO INICIAL (O pulo do gato)
# =================================================================
# np.where varre todo o space.xc e aplica a descontinuidade
uic = np.where(space.xc <= -0.5, 2.0, 1.0)

# A projeção continua exatamente igual:
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
    folder="burgers_square_animation_wSL",
    filename="simulation.gif",
    title="Burgers",
    xlabel="x",
    ylabel="u",
    xlim=(-1.0, 1.0),
    ylim=(0.5, 2.5),
    duration=100,
    plot_kwargs={
        "color": "tab:blue",
        "lw": 2,
    },
)