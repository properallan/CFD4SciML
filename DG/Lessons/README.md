## 📚 Sumário do Curso

Abaixo segue a descrição do que você encontrará em cada notebook.

[Aula 00 - Introdução e Motivações ao Método Galerkin Descontínuo 1D]() (*Em breve*)  
O ponto de partida da jornada para construir um solver "do zero" em aproximadamente 14 semanas. Esta introdução explora o porquê de utilizar o DG, apresentando-o como o filho do Método dos Elementos Finitos (MEF) com os Volumes Finitos (VF). O material destaca as vantagens do método, como a independência dos elementos e a capacidade natural de lidar com geometrias complexas e condições de contorno, além de demonstrar o poder da sua convergência exponencial. Também oferece um panorama dos problemas 1D que serão solucionados ao longo do curso, incluindo Advecção linear, Burgers (viscoso e não-viscoso) e o Traffic Model.

[Aula 01 - Polinômios de Jacobi e suas Derivadas](https://github.com/properallan/CFD4SciML/blob/main/DG/Lessons/01_JacobiPolynomials.ipynb)  
Construção da base espectral. Explora a formulação matemática da EDO de Jacobi, derivando as famosas famílias de Legendre e Chebyshev, e implementa a avaliação computacional rápida via relações de recorrência.

[Aula 02 - Quadraturas e Integração Numérica](https://github.com/properallan/CFD4SciML/blob/main/DG/Lessons/02_Quadraturas.ipynb)  
O fim das integrais analíticas. Aborda o mapeamento de domínios físicos para o elemento padrão (o papel do Jacobiano) e o dilema das fronteiras no DG, contrastando as estratégias de Gauss-Legendre (GL) e Gauss-Lobatto-Legendre (GLL).

[Aula 03 - Zeros, Pesos e Gauss-Jacobi Numérico]() (*Em breve*)  
A automatização da quadratura. Utiliza o Método de Newton-Raphson acoplado com Deflação Polinomial para encontrar raízes de forma algorítmica e aplica fórmulas fechadas para o cálculo de pesos de qualquer ordem.

[Aula 04 - Base Ortogonal e Matriz de Vandermonde]() (*Em breve*)  
A ponte entre o mundo físico e o espectral. Avaliação dos polinômios de Legendre nos nós de quadratura para a geração da Matriz de Vandermonde.
