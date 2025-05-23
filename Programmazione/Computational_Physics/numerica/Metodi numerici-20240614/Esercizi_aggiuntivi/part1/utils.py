import numpy as np

def gdescent(A, b, x0, nmax = 1000, rtoll = 1e-6):
  """
  Metodo del gradiente a parametro dinamico per sistemi lineari.

  Input:
   A      Matrice del sistema
   b      Termine noto (vettore)
   x0     Guess iniziale (vettore)
   nmax   Numero massimo di iterazioni
   toll   Tolleranza sul test d'arresto (sul residuo relativo)
 
  Output:
   xiter  Lista delle iterate

  """
  norm = np.linalg.norm

  bnorm = norm(b)

  r = b - A@x0

  xiter = [x0]
  iter = 0

  while((norm(r) / bnorm)>rtoll  and iter < nmax):
      xold = xiter[-1]

      z = r
      rho = np.dot(r, z)
      q = A @ z;
      alpha = rho / np.dot(z, q)
      xnew = xold + alpha * z     
      r = r - alpha*q

      xiter.append(xnew)
      iter = iter + 1
                     
  return xiter

