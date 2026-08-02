## 📚 Sumário do Curso

Abaixo segue a descrição do que você encontrará em cada notebook.

[Aula 00 - Introdução ao Curso do Método Galerkin Descontínuo 1D](https://github.com/properallan/CFD4SciML/blob/main/DG/Lessons/00_Introducao_Curso_DG_1D.ipynb) (**Disponível**)  
O ponto de partida da jornada para construir um solver "do zero" em aproximadamente 14 semanas. Esta introdução explora o porquê de utilizar o DG, apresentando-o como o filho do Método dos Elementos Finitos (MEF) com os Volumes Finitos (VF). O material destaca as vantagens do método e oferece um panorama da teoria que será apresentada ao longo do curso.

[Aula 01 - Polinômios de Jacobi e suas Derivadas](https://github.com/properallan/CFD4SciML/blob/main/DG/Lessons/01_JacobiPolynomials.ipynb)  (**Disponível**)  
Construção da base espectral. Explora a formulação matemática da EDO de Jacobi, derivando as famosas famílias de Legendre e Chebyshev, e implementa a avaliação computacional rápida via relações de recorrência.

[Aula 02 - Quadraturas e Integração Numérica](https://github.com/properallan/CFD4SciML/blob/main/DG/Lessons/02_Quadraturas.ipynb) (**Disponível**)  
Mapeamento de domínios físicos para o elemento padrão. Dilema das fronteiras no DG. As estratégias de Gauss-Legendre (GL) e Gauss-Lobatto-Legendre (GLL).

[Aula 03 - Zeros, Pesos e Gauss-Jacobi Numérico](https://github.com/properallan/CFD4SciML/blob/main/DG/Lessons/03_Quadraduta_via_Jacobi.ipynb) (**Disponível**)  
Quadratura utilizando Polinômios e Derivadas de Jacobi. Uso do Método de Newton-Raphson acoplado com Deflação Polinomial para encontrar raízes. Aplicação de fórmulas fechadas para o cálculo de pesos de qualquer ordem.

[Aula 04 - Matriz de Diferenciação via Vandermonde](https://github.com/properallan/CFD4SciML/blob/main/DG/Lessons/04_MatrizDifer_Vandermonde.ipynb) (**Disponível**)  
Matriz de Vandermonde Tradicional e Generalizada; Base de Legendre e Pontos de Gauss-Lobatto; Condicionamento de Matrizes; Matriz de Diferenciação via Vandermonde.

[Aula 05 - Matriz de Diferenciação via Polinômios de Jacobi](https://github.com/properallan/CFD4SciML/blob/main/DG/Lessons/05_MatrizDifer_Colocacao.ipynb) (**Disponível**)  
Demonstrações para a matriz de Diferenciação via Jacobi (Matriz de Colocação) para pontos de Gauss-Legendre e Gauss-Lobatto.

[Aula 06 - Fundamentos Geométricos e Matrizes do DG](https://github.com/properallan/CFD4SciML/blob/main/DG/Lessons/06_Stiffness_Matrix.ipynb) (**Disponível**)  
Nesta aula são apresentadas peças fundamentais para o DG1D. É feito o desenvolvimento teórico e os códigos associados a: Malha 1D, Jacobiano, Base Polinomial de Legendre, Matrizes de Massa, Rigidez e Elevação (*Lift Matrices*) 

[Aula 07 - Projeção de Fluxos](https://github.com/properallan/CFD4SciML/blob/main/DG/Lessons/07_FluxProjection.ipynb) (**Disponível**)  
Apresentação da estratégia de Projeção dos Fluxos e super-integração para evitar erros de *aliasing*.

[Aula 08 - Fluxos Numéricos](https://github.com/properallan/CFD4SciML/blob/main/DG/Lessons/08_Fluxos_Numericos.ipynb) (**Disponível**)  
Nesta aula é explicado como é feito o tratamento dos termos nas fronteira com o uso de estratégias de Fluxo Numéricos tanto para casos Convectivos quanto Difusivos.

[Aula 09 - Condições Iniciais e de Contorno]() (*Em breve*)

[Aula 10 - Construção do Operador Lh]() (*Em breve*)

[Aula 11 - Slope Limiters e o Fenômeno de Gibbs - Parte 1]() (*Em breve*)

[Aula 12 - Slope Limiters e o Fenômeno de Gibbs - Parte 2]() (*Em breve*)

[Aula 13 - Método de Runge-Kutta Strong Stability Preserving (RKSSP) - Parte 1]() (*Em breve*)

[Aula 14 - Método de Runge-Kutta Strong Stability Preserving (RKSSP) - Parte 2]() (*Em breve*)

[Aula 15 - Resolução de Problemas com solver DG1D]() (*Em breve*)
