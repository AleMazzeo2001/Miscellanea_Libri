# import necessari
import numpy as np

def fvsolve(u0, f, df, L, T, h, dt, flux_function):
  """ Risolve un dato problema di trasporto utilizzando il metodo ai volumi finiti 1D.

  Input:
   u0            (lambda function)        Dato al tempo t = 0 (profilo iniziale)
   f             (lambda function)        Flusso dell'equazione,  f = f(u)
   df            (lambda function)        Derivata del flusso, df = f'(u)
   L             (float)                  Lunghezza dell'intervallo spaziale
   T             (float)                  Tempo finale
   h             (float)                  Grandezza delle celle
   dt            (float)                  Passo temporale
   flux_function (function)               Function da utilizzare per i flussi

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
      flusso1 = flux_function(f, df, uex[0:-2], uex[1:-1])
      flusso2 = flux_function(f, df, uex[1:-1], uex[2:])

      # Passo temporale
      u[:,n+1] = u[:,n] + (dt/h)*(flusso1-flusso2);

  return xc, t, u
