import numpy as np
import scipy

def fwsub(A,b):
  """
  Algoritmo di sostituzione in avanti - forward substitution.

  Input:
    A   (numpy.ndarray)   Matrice quadrata triangolare inferiore
    b   (numpy.ndarray)   Termine noto
  
  Output:
    x soluzione del sistema Ax = b, restituita come vettore numpy.
  """

  # dimesione termine noto b
  n = b.shape[0]

  # Verifichiamo che la matrice sia quadrata
  if A.shape[0] != A.shape[1]:
    raise RuntimeError("ERRORE: matrice non quadrata")

  # Verifichiamo che la matrice sia triangolare inferiore
  if (A != scipy.linalg.tril(A)).any():
    raise RuntimeError("ERRORE: matrice non triangolare inferiore")

  # Verifichiamo che la matrice sia invertibile
  # Essendo triangolare, i suoi autovalori si trovano sulla diagonale principale
  if np.prod(np.diag(A)) == 0:
    raise RuntimeError("ERRORE: matrice singolare")

  # inizializzo il vettore
  x = np.zeros(n)
  # costruzione forward substitution
  x[0] = b[0]/A[0,0]

  for i in range(1,n):
    x[i] = (b[i] - A[i,0:i] @ x[0:i]) / A[i,i]

  # Versione alternativa: doppio ciclo for
  #  x = np.zeros(n)
  #  x[0] = b[0] / A[0,0]
  #
  #  for i in range(1,n):
  #    s = 0
  #
  #    for j in range(0,i):
  #      s = s + A[i,j] * x[j]
  #
  #    x[i] = (b[i] - s) / A[i,i]

  return x


def bksub(A,b):
  """
  Algoritmo di sostituzione all'indietro - backward substitution.

  Input:
    A   (numpy.ndarray)   Matrice quadrata triangolare superiore
    b   (numpy.ndarray)   Termine noto
  
  Output:
    x soluzione del sistema Ax = b, restituita come vettore numpy.
  """

  # inizializzo il vettore x
  x = []
  # dimensione vettore b
  n = b.shape[0]

  # Verifichiamo che la matrice sia quadrata
  if A.shape[0] != A.shape[1]:
    raise RuntimeError("ERRORE: matrice non quadrata")

  # Verifichiamo che la matrice sia triangolare inferiore
  if (A != scipy.linalg.triu(A)).any():
    raise RuntimeError("ERRORE: matrice non triangolare superiore")

  # Verifichiamo che la matrice sia invertibile
  # Essendo triangolare, i suoi autovalori si trovano sulla diagonale principale
  if np.prod(np.diag(A)) == 0:
    raise RuntimeError("ERRORE: matrice singolare")

  x = np.zeros(n)
  #x[n-1] = b[n-1]/A[n-1,n-1]
  x[-1] = b[-1]/A[-1,-1]

  for i in range(n-2,-1,-1):
      x[i] = (b[i] - A[i,i+1:n] @ x[i+1:n]) / A[i,i]


    # Versione alternativa: doppio ciclo for
    #  x = np.zeros(n)
    #  x[-1] = b[-1] / A[-1,-1]
    #
    #  for i in range(n-2,-1,-1):
    #    s = 0
    #
    #    for j in range(i,n):
    #      s = s + A[i,j] * x[j]
    #
    #    x[i] = (b[i] - s) / A[i,i]

  return x


def pmedcomp(f, a, b, N):
  """ Formula del punto medio composita
  Input:
     f:   funzione da integrare
     a:   estremo inferiore intervallo di integrazione
     b:   estremo superiore intervallo di integrazione
     N:   numero di sottointervalli (N = 1 formula di integrazione semplice)
  Output:
     I:   integrale approssimato """

  h = (b-a)/N                 # ampiezza sottointervalli
  x = np.linspace(a, b, N+1)  # griglia spaziale
  xL, xR = x[:-1], x[1:]      # liste dei nodi "sinistri" e "destri"
  xM = 0.5*(xL + xR)          # punti medi
  I = h*f(xM).sum()           # integrale approssimato

  return I

def simpcomp(f, a, b, N):
  """ Formula di Cavalieri-Simpson composita
  Input:
     f:   funzione da integrare
     a:   estremo inferiore intervallo di integrazione
     b:   estremo superiore intervallo di integrazione
     N:   numero di sottointervalli (N = 1 formula di integrazione semplice)
  Output:
     I:   integrale approssimato """

  h = (b-a)/N                                     # ampiezza sottointervalli
  x = np.linspace(a, b, N+1)                      # griglia spaziale
  xL, xR = x[:-1], x[1:]                          # liste dei nodi "sinistri" e "destri"
  xM = 0.5*(xL + xR)                              # punti medi
  I = (h/6.0)*(f(xL)+4*f(xM)+f(xR)).sum()         # integrale approssimato

  return I