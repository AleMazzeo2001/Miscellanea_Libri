import numpy as np
import scipy
  
def newton(x0,nmax,toll,fun,dfun):
  """ Metodo di Newton per la ricerca degli zeri della funzione fun.
      Test d'arresto basato sul controllo della differenza tra due iterate successive
  
   input:
    x0          (float)               punto di partenza
    nmax        (int)                 numero massimo di iterazioni
    toll        (float)               tolleranza sul test d'arresto
    fun, dfun   (lambda functions)    la funzione e la sua derivata
  
   output:
    xvect        (numpy.ndarrray) -> vector   Iterate calcolate
                                                (l'utima componente è la soluzione)
    it            (int)                       Iterazioni effettuate
  """

  err = toll+1
  it = 0
  xvect = [x0]

  while it<nmax and err>=toll:
    xv = xvect[-1]
    if np.abs(dfun(xv)) < np.finfo(float).eps:
      disp('Arresto per azzeramento di dfun')
      it = it+1
      break
    else:
      xn = xv - fun(xv) / dfun(xv)
      err = np.abs(xn-xv)
      xvect.append(xn)
      it = it+1

  print('\n Numero di iterazioni: %d \n' %it)
  print(' Zero calcolato: %e\n' %xvect[-1])

  xvect = np.array(xvect)
      
  return xvect,it


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

