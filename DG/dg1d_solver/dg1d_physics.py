import numpy as np
import dg1d_numericalfluxes as num_fluxes

class ConservationLaw:
    def __init__(self, space, num_flux_name='lax_friedrichs'):
        """
        Classe base para qualquer Lei de Conservação 1D.
        
        Parâmetros:
        ----------
        space : DGSpace1D
            Objeto contendo a malha, matrizes e polinômios pré-calculados.
        num_flux_name : str
            Nome da função de fluxo numérico a ser buscada no módulo dg1d_numericalfluxes.
        """
        self.space = space
        
        # Link dinâmico com o fluxo numérico escolhido pelo usuário
        if hasattr(num_fluxes, num_flux_name):
            self.numerical_flux = getattr(num_fluxes, num_flux_name)
        else:
            raise ValueError(f"Fluxo '{num_flux_name}' não encontrado em dg1d_numericalfluxes.py!")

    # =====================================================================
    # MÉTODOS ABSTRATOS (O usuário DEVE implementar nas classes filhas)
    # =====================================================================

    def physical_flux(self, *U_fisico):
        raise NotImplementedError("Implemente o fluxo físico da equação.")

    def max_wave_speed(self, *U_fisico):
        raise NotImplementedError("Implemente o cálculo da velocidade máxima (C).")

    def apply_bc(self, ut_list, uL_list, uR_list, t):
        """
        Aplica as condições de contorno preenchendo as células fantasmas (ghost cells)
        nas posições ut_list[i][:, 0] e ut_list[i][:, -1].
        """
        raise NotImplementedError("Implemente as Condições de Contorno.")


    # =====================================================================
    # MÉTODOS INTERNOS (O motor escondido do usuário)
    # =====================================================================

    def _ModifStiffMatrix(self, epst, epsb=None, type_form='Warburton'):
        """
        Calcula a Matriz de Rigidez Modificada para o termo difusivo.
        Puxa a matemática do self.space de forma automática.
        """
        if type_form == 'Warburton':
            if epsb is None:
                raise ValueError("Para a abordagem 'Warburton', 'epsb' não pode ser vazio.")

            Ssq1 = np.dot(np.sqrt(epst) * self.space.wi * self.space.psi.T, self.space.Dpsi)
            Boundterm = (np.sqrt(epsb[-1]) * self.space.Frk) - (np.sqrt(epsb[0]) * self.space.Flk)
            Ssq2 = Boundterm - Ssq1.T

            return Ssq1.T, Ssq2.T

        elif type_form == 'Persson':
            Ssq1 = np.dot(epst * self.space.wi * self.space.psi.T, self.space.Dpsi)
            return Ssq1.T, None
        else:
            raise ValueError("Modelo de matriz modificada não reconhecido!")


    def _FluxProjection(self, ut_list):
        """
        Projeta o fluxo físico no espaço modal através de uma projeção L2
        usando superintegração para evitar erros de aliasing.
        """
        uhat_list = [ut[:, 1:-1] for ut in ut_list]
        
        # Reconstrução da solução física nos pontos de quadratura não-linear
        uh_list = [np.dot(self.space.nlpsi, uhat) for uhat in uhat_list]   
        
        # Avaliação dos fluxos físicos (chama o método do usuário)
        fh_list = self.physical_flux(*uh_list)   

        if not isinstance(fh_list, (list, tuple)):
            fh_list = [fh_list]

        ProjectionMatrix = self.space.nlwi * self.space.nlpsi.T
        ft_list = []

        # Projeção L2 devolvendo arrays com ghost cells (K+2) preenchidas com zero nas bordas
        for fh in fh_list:
            b = np.dot(ProjectionMatrix, fh)
            fhat = self.space.InvM[:, None] * b
            
            # Aloca array com células fantasmas
            ft = np.zeros((self.space.Nldof, self.space.K + 2))
            ft[:, 1:-1] = fhat
            ft_list.append(ft)

        return ft_list


    def compute_rhs(self, U_modal_list, t):
        """
        O coração do método DG. Equivalente ao antigo Lh/EulerRHS.
        Calcula dU/dt para o integrador de tempo. Suporta sistemas de N variáveis.
        """
        # 1. Prepara as matrizes com células fantasmas (ghost cells) nas bordas
        ut_list = []
        for u in U_modal_list:
            ut = np.zeros((self.space.Nldof, self.space.K + 2))
            ut[:, 1:-1] = u.copy()
            ut_list.append(ut)

        # 2. Extrai os valores nas fronteiras esquerda (uL) e direita (uR) do domínio real
        uL_list = [np.dot(self.space.Flkp1[0, :], ut[:, 1]) for ut in ut_list]
        uR_list = [np.dot(self.space.Frk[0, :], ut[:, -2]) for ut in ut_list]

        # 3. Aplica Condições de Contorno (O usuário preenche as ghost cells aqui)
        self.apply_bc(ut_list, uL_list, uR_list, t)

        # 4. Projeção do Fluxo Físico (Volume)
        ft_list = self._FluxProjection(ut_list)

        # --- A CORREÇÃO ENTRA AQUI: Preencher o fluxo nas Ghost Cells ---
        # Pega apenas o valor médio (índice 0) que foi preenchido no apply_bc
        # state_left = [ut[0, 0] for ut in ut_list]
        # state_right = [ut[0, -1] for ut in ut_list]
        
        # # Chama a função de fluxo do usuário para as bordas!
        # f_left = self.physical_flux(*state_left)
        # f_right = self.physical_flux(*state_right)
        
        # if not isinstance(f_left, (list, tuple)):
        #     f_left = [f_left]
        #     f_right = [f_right]
            
        # for i in range(len(ft_list)):
        #     ft_list[i][0, 0] = f_left[i]
        #     ft_list[i][0, -1] = f_right[i]
        # # ----------------------------------------------------------------

        # 5. Recupera a solução física padrão para cálculo da velocidade máxima (Lax-Friedrichs)
        uh_list = [np.dot(self.space.psi, u) for u in U_modal_list]
        max_speed = self.max_wave_speed(*uh_list)

        # 6. Montagem do RHS (Itera sobre todas as equações do sistema)
        rhs_list = []
        for i in range(len(U_modal_list)):
            ut = ut_list[i]
            ft = ft_list[i]
            
            # Aqui aplicamos o fluxo numérico de Lax-Friedrichs vetorizado através das Lift Matrices.
            # NOTA: No futuro, isso será totalmente repassado para a função self.numerical_flux
            rhs = (self.space.InvM[:, None] * (self.space.J**(-1) * (
                np.dot(self.space.S.T, ft[:, 1:-1]) 
                - 0.5 * np.dot(self.space.Flkp1, (ft[:, 2:] - max_speed * ut[:, 2:]))
                - 0.5 * np.dot(self.space.Frk,   (ft[:, 1:-1] + max_speed * ut[:, 1:-1]))
                + 0.5 * np.dot(self.space.Frkm1, (ft[:, :-2] + max_speed * ut[:, :-2]))
                + 0.5 * np.dot(self.space.Flk,   (ft[:, 1:-1] - max_speed * ut[:, 1:-1]))
            )))
            
            rhs_list.append(rhs)
            
        return rhs_list