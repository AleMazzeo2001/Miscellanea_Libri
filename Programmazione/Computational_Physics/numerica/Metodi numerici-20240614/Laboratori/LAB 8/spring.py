import numpy as np
import matplotlib.pyplot as plt
import imageio

def plotSpring(dx, springcolor = 'k', boxcolor = 'orange', springalpha = 0.5, boxalpha = 1.0):
  dh = 20*dx
  x = np.linspace(0, 15*np.pi, 81)
  y = 0.3*np.sin(2*x)
  plt.plot((x[-1]+dh)*x/x[-1], y, color = springcolor, alpha = springalpha)
  x = x + dh
  hbox = 4.5
  plt.fill([x[-1], x[-1], x[-1]+hbox, x[-1]+hbox, x[-1], x[-1]], [0, -0.6, -0.6, 0.6, 0.6, 0], color = boxcolor, alpha = boxalpha)
  plt.axis([-1, 80, -1.01, 1.7])
  plt.axis("off")

def drawSpringframe(dx, t = 0.0):
  plt.figure(figsize = (6, 0.75))
  plotSpring(0.0, springcolor = 'r', boxcolor = 'r', springalpha = 0.1, boxalpha = 0.1)
  plotSpring(dx)
  plt.text(0, 1.5, 't = %.1f' % t, fontsize = 8)
  plt.plot(0, 0, '.k', alpha = 0.5)

def savegif(drawframe, frames, name, dt = 1.0/24.0):
  arrays = []
  for i in range(frames):
    drawframe(i)
    fig = plt.gcf()
    fig.canvas.draw()
    arrays.append(np.array(fig.canvas.renderer.buffer_rgba()))
    plt.close(fig)

  imageio.mimsave(name.replace(".gif", "") + ".gif", arrays, duration = dt)

def animate(t_h, x_h):
  t = np.linspace(t_h[0], t_h[-1], int(10*(t_h[-1]-t_h[0])+1))
  ind = np.abs(t_h.reshape(-1, 1)-t.reshape(1, -1)).argmin(axis = 0)
  x = x_h[ind]

  def drawframe(i):
    drawSpringframe(x[i], t[i])
  rnd = np.random.randint(50000)
  savegif(drawframe, frames = len(t), name = "temp%d-gif.gif" % rnd, dt = 0.1)
  from IPython.display import Image, display
  display(Image("temp%d-gif.gif" % rnd), metadata={'image/gif': {'loop': True}})
  from os import remove
  remove("temp%d-gif.gif" % rnd)


