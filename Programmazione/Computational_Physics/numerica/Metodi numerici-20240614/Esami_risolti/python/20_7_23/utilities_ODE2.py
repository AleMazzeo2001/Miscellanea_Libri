import numpy as np
from scipy.linalg import lu
import luFactorization

def eulero_avanti(f, t0, t_max, y0, h):
    """ Risolve il problema di Cauchy
    
     y'   = f(t,y)
     y(0) = y0
    
     utilizzando il metodo di Eulero in avanti (esplicito):
     u^(n+1) = u^n + h*f^n
    
     L'equazione differenziale ordinaria può essere in generale vettoriale
     (y=f(t,y) in R^d)
     per d=1 si ottiene il caso scalare.
    
     Input:
           f: lambda function che descrive il problema di Cauchy.
               Riceve in input due argomenti: f=f(t,y), con y vettore di lunghezza d
           t0, t_max: estremi dell'intervallo temporale di soluzione
           y0: dato iniziale del problema di Cauchy (vettore di lunghezza d)
           h: ampiezza de passo di discretizzazione temporale
     ATTENZIONE: controllare che l'output di f e il dato y0 siano vettori della stessa lunghezza!
    
     Output:
           t_h = vettore degli istanti in cui viene calcolata la soluzione discreta (lunghezza N)
           u_h = soluzione discreta calcolata nei nodi temporali t_h (matrice di dimensioni N x d)
    """
    
    # vettore dei nodi temporali
    t_h = np.linspace(t0, t_max, int((t_max-t0)/h)+1)

    # inizializzazione del vettore soluzione
    N = len(t_h)
    d = len(y0)
    u_h = np.zeros((N, d))

    # ciclo iterativo che calcola i passi di Eulero esplicito
    u_h[0, :] = y0

    for it in range(N-1):
        u_old = u_h[it, :]
        u_h[it+1, :] = u_old + h*f(t_h[it], u_old)
    
    return t_h, u_h


def eulero_indietro_sis_lineari(A, t0, t_max, y0, h):
    # Risolve il problema di Cauchy
    #
    # y'   = Ay
    # y(0) = y0
    #
    # utilizzando il metodo di Eulero all'indietro (implicito):
    # u^(n+1) = u^n + h*Au^(n+1) ==> (I-h*A)u^(n+1) = u^n
    #
    # Input:
    #       A: Matrice
    #       t0, t_max: estremi dell'intervallo temporale di soluzione
    #       y0: dato iniziale del problema di Cauchy (vettore di lunghezza d)
    #       h: ampiezza del passo di discretizzazione temporale
    #
    # Output:
    #       t_h = vettore degli istanti in cui viene calcolata la soluzione discreta
    #       u_h = soluzione discreta calcolata nei nodi temporali t_h (matrice di dimensioni N x d)

    # vettore degli nodi temporali
    t_h = np.linspace(t0, t_max, int((t_max-t0)/h)+1)

    # inizializzazione del vettore soluzione
    N = len(t_h)
    d = len(y0)
    u_h = np.zeros((N, d))

    # ciclo iterativo che calcola (I-h*A)u^(n+1) = u^n
    u_h[0, :] = y0

    # fattorizzazione LU della matrice I-h*A
    P, L, U = lu(np.eye(A.shape[0]) - h*A)

    for it in range(N-1):
        u_old = u_h[it, :]
        y = luFactorization.fwsub(L, P.T @ u_old)
        u_h[it+1, :] = luFactorization.bksub(U, y)

    return t_h, u_h