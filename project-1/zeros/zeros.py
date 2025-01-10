# Project 1 [ZERI DI FUNZIONE]

import numpy as np
import matplotlib.pyplot as plt

# Definition of the function f
f = lambda x: np.exp(x)-x**2

# xTrue, true zero of f
xTrue = -0.703467
fTrue = f(xTrue)
print('fTrue = ', fTrue) # 8.035078391532835e-07, very close to zero

# Plot f
a = -1; b = 1
xplot = np.linspace(a, b)
fplot = f(xplot)
plt.plot(xplot,fplot)
plt.plot(xTrue,fTrue, 'or', label='$x^*$', markersize=15)
plt.legend()
plt.grid()
plt.show()

# Bisection method
def bisection(f, a, b, tolx, maxit, xTrue):
    """
    Bisection method to find the root of a function f in the interval [a, b].
    
    Parameters:
    - f: function to find the root of.
    - a, b: initial interval [a, b] where f(a)*f(b) < 0.
    - tolx: tolerance for the absolute error of the root.
    - maxit: maximum number of iterations.
    - xTrue: the true root (for comparison).
    
    Returns:
    - root: the estimated root.
    - i: the number of iterations performed.
    - err: array of absolute errors |b - a| / 2 at each iteration.
    - vecErrore: array of absolute errors |x_k - xTrue| at each iteration.
    """
    err = np.zeros(maxit + 1, dtype=np.float64)  # Error in interval size
    vecErrore = np.zeros(maxit + 1, dtype=np.float64)  # Error relative to true root
    i = 0
    
    while i < maxit:
        c = (a + b) / 2
        err[i] = np.abs(b - a) / 2
        vecErrore[i] = np.abs(c - xTrue)

        if np.abs(f(c)) < tolx or err[i] < tolx:
            break
        
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
        
        i += 1

    err = err[0:i+1]
    vecErrore = vecErrore[0:i+1]
    return c, i, err, vecErrore

# Fixed-point iteration method
def succ_app(f, g, tolf, tolx, maxit, xTrue, x0=0):
    i = 0
    err = np.zeros(maxit+1, dtype=np.float64)
    vecErrore = np.zeros(maxit+1, dtype=np.float64)
    vecErrore[0] = np.abs(x0 - xTrue)
    x = x0
    err[0] = tolx + 1

    while (err[i] > tolx and i < maxit):
        x_new = g(x)
        err[i+1] = np.abs(x_new - x)
        vecErrore[i+1] = np.abs(x_new - xTrue)
        i += 1
        x = x_new
    
    err = err[0:i+1]
    vecErrore = vecErrore[0:i+1]
    
    return x, i, err, vecErrore

# Newton's method (with fixed-point iteration)
def newton(f, df, tolf, tolx, maxit, xTrue, x0=0):
  g = lambda x: x - f(x) / df(x)  # Newton's iteration function
  (x, i, err, vecErrore) = succ_app(f, g, tolf, tolx, maxit, xTrue, x0)
  return (x, i, err, vecErrore)


df = lambda x: np.exp(x)-2*x # Derivative of f
g1 = lambda x: x-f(x)*np.exp(x/2) # Fixed-point iteration function g1
g2 = lambda x: x-f(x)*np.exp(-x/2) # Fixed-point iteration function g2
tolx= 10**(-10)
tolf = 10**(-6)
maxit=100
x0= 0
[sol_bisection, iter_bisection, err_bisection, vecErrore_bisection] = bisection(f, -1, 1, tolx, maxit, xTrue)
print('Metodo Bisezione \n x =', sol_bisection, '\n iter_bisection =', iter_bisection)
plt.plot(sol_bisection, f(sol_bisection), 'oc', label='Bisection', markersize=13) # Converges

[sol_g1, iter_g1, err_g1, vecErrore_g1]=succ_app(f, g1, tolf, tolx, maxit, xTrue, x0)
print('Metodo approssimazioni successive g1 \n x =',sol_g1,'\n iter_new=', iter_g1)
plt.plot(sol_g1,f(sol_g1), 'o', label='g1', markersize=11) # Converges

[sol_g2, iter_g2, err_g2, vecErrore_g2]=succ_app(f, g2, tolf, tolx, maxit, xTrue, x0)
print('Metodo approssimazioni successive g2 \n x =',sol_g2,'\n iter_new=', iter_g2)
plt.plot(sol_g2,f(sol_g2), 'og', label='g2') # Does not converge

[sol_newton, iter_newton, err_newton, vecErrore_newton]=newton(f, df, tolf, tolx, maxit, xTrue, x0)
print('Metodo Newton \n x =',sol_newton,'\n iter_new=', iter_newton)
plt.plot(sol_newton,f(sol_newton), 'ob', label='Newton', markersize=9) # Converges
plt.legend()
plt.grid()
plt.show()
plt.plot(vecErrore_bisection[:20], '.-', color='cyan', label='Bisection')
plt.plot(vecErrore_g1, '.-', color='blue')
plt.plot(vecErrore_g2[:20], '.-', color='green')
plt.plot(vecErrore_newton, '.-', color='red')
plt.legend( ("Bisection", "g1", "g2", "newton"))
plt.xlabel('iter')
plt.ylabel('errore')
plt.title('Errore vs Iterazioni')
plt.grid()
plt.show()