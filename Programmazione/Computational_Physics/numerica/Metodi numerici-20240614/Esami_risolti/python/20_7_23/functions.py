import numpy as np
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


def fvsolve(u0, f, df, L, T, h, dt, method):
  """ Risolve un dato problema di trasporto utilizzando il metodo ai volumi finiti 1D.

  Input:
   u0            (lambda function)        Dato al tempo t = 0 (profilo iniziale)
   f             (lambda function)        Flusso dell'equazione,  f = f(u)
   df            (lambda function)        Derivata del flusso, df = f'(u)
   L             (float)                  Lunghezza dell'intervallo spaziale
   T             (float)                  Tempo finale
   h             (float)                  Grandezza delle celle
   dt            (float)                  Passo temporale
   method        (string)                 Metodo da utilizzare per i flussi

  Output:
  xc     (numpy.ndarray)-> vector  Baricentri delle celle
  t      (numpy.ndarray)-> vector  Tempi d'evoluzione
  u      (numpy.ndarray)-> matrix  Approssimazione della soluzione. Vige la convenzione uij = u(xi,tj).
  """

  # costruzione griglie spaziali e temporali
  ncells = int(np.ceil(L/h))        # numero celle
  nt = int(np.ceil(T/dt)+1)         # numero nodi temporali
  x  = np.linspace(0, L, ncells+1)  # griglia spaziale
  xL = x[0:-1]                      # nodi sinistri
  xR = x[1:]                        # nodi destri
  xc = (xL + xR)/2.0                # centri delle celle

  t = np.linspace(0, T, nt)         # nodi temporali

  # Inizializzazione soluzione
  u = np.zeros((ncells, nt))
  u[:,0] = u0(xc)

  # Ciclo temporale
  for n in range(nt-1):
      # Soluzione estesa
      uex = np.append([u0(x[0])],u[:,n])
      uex = np.append(uex,u[-1,n])

      # Calcolo del flusso
      if method == 'UPWIND':
        flusso1 = upwind_flux(f, df, uex[0:-2], uex[1:-1])
        flusso2 = upwind_flux(f, df, uex[1:-1], uex[2:])
      elif method == 'GODUNOV':
        flusso1 = godunov_flux(f, df, uex[0:-2], uex[1:-1])
        flusso2 = godunov_flux(f, df, uex[1:-1], uex[2:])

      # Passo temporale
      u[:,n+1] = u[:,n] + (dt/h)*(flusso1-flusso2);

  return xc, t, u

# Implementazione del flusso "alla upwind"
def upwind_flux(f, df, uL, uR):
  if np.min(df(uL)) >= 0 and np.min(df(uR)) >= 0 :
    F = f(uL)
  else:
    F = f(uR)

  return F

# Implementazione del flusso "alla Godunov"
def godunov_flux(f, df, uL, uR):
  iL = np.minimum(np.array(uL), np.array(uR))
  iR = np.maximum(np.array(uL), np.array(uR))
  g = np.linspace(0,1,1000).reshape(1,1000)

  iL = iL.reshape(len(iL),1)
  iR = iR.reshape(len(iR),1)
  g = g.reshape(1,1000)

  itot = f(iL@g + iR@(1-g))

  imins = itot.min(axis=1)
  imaxs = itot.max(axis=1)

  candidates = imins
  d = np.sign(uR-uL)
  candidates[d<0] = imaxs[d<0]
  return np.array(candidates)
