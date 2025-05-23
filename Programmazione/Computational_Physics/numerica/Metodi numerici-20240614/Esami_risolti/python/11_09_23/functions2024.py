import numpy as np

def DEFsplit(A):
  D = np.diag(np.diag(A))
  E = -np.tril(A, k = -1)
  F = -np.triu(A, k = 1)
  return D, E, F

def Jacobi_Bc(A, b = None):

  D, E, F = DEFsplit(A)
  M = D
  N = E+F

  Minv = np.diag(1.0/np.diag(M))
  B = Minv @ N

  if(b is None):
    return B
  else:
    c = Minv @ b
    return B, c


from scipy.linalg import solve_triangular

def GS_Bc(A, b = None):

  D, E, F = DEFsplit(A)
  M = D-E
  N = F

  B = solve_triangular(M, N, lower = True)

  if(b is None):
    return B
  else:
    c = solve_triangular(M, b, lower = True)
    return B, c

def iterative_solve(A, b, x0, method, nmax, rtoll):

  r = A @ x0 - b
  bnorm = np.linalg.norm(b)

  if(method == 'Jacobi'):
    B, c = Jacobi_Bc(A, b)
  elif(method == 'GS'):
    B, c = GS_Bc(A, b)
  else:
    raise RuntimeError("Metodo sconosciuto.")

  k = 0
  xiter = [x0]

  while( (np.linalg.norm(r)/bnorm) > rtoll  and k < nmax):
    xold = xiter[-1]
    xnew = B @ xold + c
    xiter.append(xnew)
    r = A @ xnew - b
    k = k+1

  return xiter
