import numpy
import matplotlib.pyplot as plt
from scipy import integrate
"""
Python: Evaluating Integral for array
"""
def sin_sin(x,y):
    return x*y + x**2

def integral(x,y):
    I = integrate.quad(f, 0, x, args=(y,))[0]
    return I

def gau(x,y):
    return (1+x)*integral(x,y)

xlist = numpy.linspace(-3.0, 3.0, 100)
ylist = numpy.linspace(-3.0, 3.0, 100)
X, Y = numpy.meshgrid(xlist, ylist)
ugau = numpy.frompyfunc(gau,2,1)
Z=ugau(X,Y)
print(Z)
fig = plt.figure()
ax = fig.gca()
# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
