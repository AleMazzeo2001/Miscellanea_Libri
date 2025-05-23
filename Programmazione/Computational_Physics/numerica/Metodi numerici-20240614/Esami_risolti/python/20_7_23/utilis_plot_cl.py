import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import Colormap
import imageio

def xtplot(x,t,u, type='animation', color='blue', piecewise=True):
  """ Plotto la soluzione .
  Input:
  x         (numpy.ndarray)-> vector  Baricentri delle celle
  t         (numpy.ndarray)-> vector  Tempi d'evoluzione
  i         (int)                     Frame
  u         (numpy.ndarray)-> matrix  Approssimazione della soluzione. Vige la convenzione uij = u(xi,tj).
  type      (string)                  'animation' o 'surface'
  color     (string)                  Colore
  piecewise (bool)                    Costante a tratti True or False
  """
  match type:
    case 'animation':
      animateConservationLaws(x,t,u, color, piecewise)
    case 'surface':
      surfaceConservationLaws(x,t,u)
    case _:
      raise RuntimeError('Error: il type in input è sbagliato')


#
# animation
#
def plotFrame_ConservationLaws(x, t, i, u, color='blue', piecewise=True):
  """ Plotto la soluzione dell'i-esimo frame .
  Input:
  x         (numpy.ndarray)-> vector  Baricentri delle celle
  t         (numpy.ndarray)-> vector  Tempi d'evoluzione
  i         (int)                     Frame
  u         (numpy.ndarray)-> matrix  Approssimazione della soluzione. Vige la convenzione uij = u(xi,tj).
  color     (string)                  Colore
  piecewise (bool)                    Costante a tratti True or False
  """

  # quantità utili per sistemare i grafici
  umax = np.max(u)
  umin = np.min(u)
  # numero di step
  Nt = len(u[0,:])
  # step spostamento
  h = x[1]-x[0]

  if (i>Nt):
    raise RuntimeError('Error: il numero di step in input è sbagliato')

  # rappresentazione grafica dell'i-esimo frame
  fig = plt.figure(figsize = (6, 5))
  ax = fig.subplots()
  fvplot(x, u[:,i], color, piecewise)
  plt.xlabel('x')
  plt.ylabel('u(x,t)')
  plt.title('Current time: t = %.2f' %t[i] )
  plt.axis([x[0]-h, x[-1]+h, umin-0.3*(umax-umin), umax+0.3*(umax-umin)])

# funzione utile per plottare
def fvplot(x, v, c, pcw):
  if (pcw):
    h = (x[1]-x[0])/2
    plt.plot([x[0]-h, x[0]+h], [v[0], v[0]], color=c)

    for i in range(1,len(x)):
      plt.plot([x[i]-h, x[i]-h], [v[i-1], v[i]],'--', color=c)
      plt.plot([x[i]-h, x[i]+h], [v[i], v[i]], color=c)

  else:
      plt.plot(x,v,color=c)

# salvare formato gif
def savegif(drawframe, frames, name, dt = 1.0/24.0):
  arrays = []
  for i in range(frames):
    drawframe(i)
    fig = plt.gcf()
    fig.canvas.draw()
    arrays.append(np.array(fig.canvas.renderer.buffer_rgba()))
    plt.close(fig)

  imageio.mimsave(name.replace(".gif", "") + ".gif", arrays, duration = dt)

# costruire function per animare
def animateConservationLaws(xc, t, u, color='blue', piecewise=True):
  """ Plotto la soluzione attraverso un'animazione
  Input:
  x         (numpy.ndarray)-> vector  Baricentri delle celle
  t         (numpy.ndarray)-> vector  Tempi d'evoluzione
  u         (numpy.ndarray)-> matrix  Approssimazione della soluzione. Vige la convenzione uij = u(xi,tj).
  color     (string)                  Colore
  piecewise (bool)                    Costante a tratti True or False
  """
  def drawframe(i):
    plotFrame_ConservationLaws(xc, t, i, u, color, piecewise)

  rnd = np.random.randint(50000)
  savegif(drawframe, frames = len(t), name = "temp%d-gif.gif" % rnd, dt = 150)
  from IPython.display import Image, display
  display(Image("temp%d-gif.gif" % rnd), metadata={'image/gif': {'loop': True}})
  from os import remove
  remove("temp%d-gif.gif" % rnd)
#
# surface
#
def surfaceConservationLaws(x, t, u):
  """ Soluzione vista dall'alto
  Input:
  x         (numpy.ndarray)-> vector  Baricentri delle celle
  t         (numpy.ndarray)-> vector  Tempi d'evoluzione
  u         (numpy.ndarray)-> matrix  Approssimazione della soluzione. Vige la convenzione uij = u(xi,tj).
  """
  X,T = np.meshgrid(x, t, indexing='ij')

  # surface
  fig1 = plt.figure(1, figsize=(8,8))
  ax1 = fig1.add_subplot(projection = '3d')
  surf = ax1.plot_surface(X, T, u, linewidth=0, antialiased=False, shade = True, alpha = 0.8, cmap='plasma')
  ax1.set_xlabel('x')
  ax1.set_ylabel('t')
  ax1.set_zlabel('u(x,t)')
  plt.title("Surface")

  ax1.view_init(30, -90)

  # vista dall'alto
  fig2, ax2= plt.subplots()
  plt.figure(2, figsize=(8,8))
  plt.pcolor(X, T, u,linewidth=2,cmap='plasma')
  ax2.set_xlabel('x')
  ax2.set_ylabel('t')
  plt.title("Vista dall'alto")
  plt.colorbar()

  plt.show()
