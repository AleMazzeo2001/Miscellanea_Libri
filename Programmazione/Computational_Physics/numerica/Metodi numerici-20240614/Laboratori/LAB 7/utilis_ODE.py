import numpy as np
def puntofisso(phi, x0, nmax=100, toll=1.e-6):
  """ Metodo punto fisso
  Input:
    𝚙𝚑𝚒   (lambda function)   Funzione di iterazione
    𝑥0    (float)             Punto di partenza
    𝚗𝚖𝚊𝚡  (float)             Numero massimo di iterazione
    𝚝𝚘𝚕𝚕  (float)             Tolleranza richiesta

  Output:
    𝚡𝚟𝚎𝚌𝚝 (numpy.ndarray)     Vettore delle iterate.
  """

  # inizializzazione
  xvect=[]
  xold = x0

  for nit in range(nmax) :
    # calcolo il nuovo punto
    xnew=phi(xold)
    #carico i vettori
    xvect.append(xnew)

    # criterio di arresto e aggiorno
    if (abs(xnew-xold) < toll):
        break
    else :
        xold=xnew

  return np.array(xvect)
  

def euleroIndietro(f, t0, tN, y0, h):
  """Metodo di Eulero all'indietro
  Input:
    f   (lambda function)   Termine di destra dell'ODE, passata come
                            funzione di tempo e spazio, f = f(t, y)
    t0  (float)             Tempo iniziale
    tN  (float)             Tempo finale
    y0  (float)             Dato iniziale
    h   (float)             Passo temporale

  Output:
    t   (numpy.ndarray)     Griglia temporale
    u   (numpy.ndarray)     Approssimazioni della soluzione nei nodi temporali t_i
  """
  # valori iniziali
  u = [y0]
  t = [t0]

  # parametri per il punto fisso
  nmax_pf=300
  toll_pf=1e-5

  while t[-1]<tN:
    # definisco la lambda function phi per il metodo del punto fisso
    phi = lambda z: u[-1] + h * f(t[-1]+h, z)

    # chiamo il metodo del punto fisso
    u_pf = puntofisso(phi, u[-1], nmax_pf, toll_pf);
    # carico i vettori u e t
    u.append(u_pf[-1])
    t.append(t[-1]+h)

  t = np.array(t)
  u = np.array(u)
  return t, u


def crankNicolson(f, t0, tN, y0, h):
  """Metodo di Crank-Nicolson
  Input:
    f   (lambda function)   Termine di destra dell'ODE, passata come
                            funzione di tempo e spazio, f = f(t, y)
    t0  (float)             Tempo iniziale
    tN  (float)             Tempo finale
    y0  (float)             Dato iniziale
    h   (float)             Passo temporale

  Output:
    t   (numpy.ndarray)     Griglia temporale
    u   (numpy.ndarray)     Approssimazioni della soluzione nei nodi temporali t_i
  """
  # primi valori della lista
  t = [t0]
  u = [y0]

  # parametri per il punto fisso
  nmax_pf=300
  toll_pf=1e-5

  # implementazione eulero in avanti
  while t[-1]<tN:

    # definisco la lambda function phi per il metodo del punto fisso
    phi = lambda z: u[-1]+h*(f(t[-1], u[-1])+f(t[-1]+h, z))/2
  
    # chiamo il metodo del punto fisso
    u_pf = puntofisso(phi, u[-1], nmax_pf, toll_pf);

    u.append(u_pf[-1])
    t.append(t[-1]+h)

  t = np.array(t)
  u = np.array(u)

  return t, u

