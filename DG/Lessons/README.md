## 📚 Sumário do Curso

Abaixo segue a descrição do que você encontrará em cada notebook.

[Aula 00 - Introdução e Motivações ao Método Galerkin Descontínuo 1D]() (*Em breve*)  
O ponto de partida da jornada para construir um solver "do zero" em aproximadamente 14 semanas. Esta introdução explora o porquê de utilizar o DG, apresentando-o como o filho do Método dos Elementos Finitos (MEF) com os Volumes Finitos (VF). O material destaca as vantagens do método, como a independência dos elementos e a capacidade natural de lidar com geometrias complexas e condições de contorno, além de demonstrar o poder da sua convergência exponencial. Também oferece um panorama dos problemas 1D que serão solucionados ao longo do curso, incluindo Advecção linear, Burgers (viscoso e não-viscoso) e o Traffic Model.

[Aula 01 - Polinômios de Jacobi e suas Derivadas](https://github.com/properallan/CFD4SciML/blob/main/DG/Lessons/01_JacobiPolynomials.ipynb)  (**Disponível**)  
Construção da base espectral. Explora a formulação matemática da EDO de Jacobi, derivando as famosas famílias de Legendre e Chebyshev, e implementa a avaliação computacional rápida via relações de recorrência.

[Aula 02 - Quadraturas e Integração Numérica](https://github.com/properallan/CFD4SciML/blob/main/DG/Lessons/02_Quadraturas.ipynb) (**Disponível**)  
Mapeamento de domínios físicos para o elemento padrão. Dilema das fronteiras no DG. As estratégias de Gauss-Legendre (GL) e Gauss-Lobatto-Legendre (GLL).

[Aula 03 - Zeros, Pesos e Gauss-Jacobi Numérico](https://github.com/properallan/CFD4SciML/blob/main/DG/Lessons/03_Quadraduta_via_Jacobi.ipynb) (**Disponível**)  
Quadratura utilizando Polinômios e Derivadas de Jacobi. Uso do Método de Newton-Raphson acoplado com Deflação Polinomial para encontrar raízes. Aplicação de fórmulas fechadas para o cálculo de pesos de qualquer ordem.

[Aula 04 - Matriz de Diferenciação via Vandermonde]() (*Em breve*)  
Matriz de Vandermonde Tradicional e Generalizada; Base de Legendre e Pontos de Gauss-Lobatto; Condicionamento de Matrizes; Matriz de Diferenciação via Vandermonde.

[Aula 05 - Matriz de Diferenciação via Jacobi]() (*Em breve*)  
Definição do Polinômio de Lagrange. Demonstrações para a matriz de Diferenciação; Matriz de Diferenciação via Jacobi para pontos de Gauss-Legendre e Gauss-Lobatto.
