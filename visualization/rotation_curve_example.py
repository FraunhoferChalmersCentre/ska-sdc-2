import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import interp1d

mpl.rcParams['text.usetex'] = True
params = {'text.latex.preamble': [r'\usepackage{amsmath}']}
plt.rcParams.update(params)

plt.fill_between([0, 1], [0, 1], [1, 1], alpha=.1, color='y')

x = np.linspace(0, 1, 1000)
plt.plot(x, x)

spline_x = np.linspace(0,1,10)
spline_y = 1 - np.exp(-10 * spline_x)
spline_y[-1] = 1
spline_fct = interp1d(spline_x, spline_y, kind='cubic')
plt.plot(x, spline_fct(x))

n_splines = 20
spline_x = np.linspace(0, 1, n_splines)
spline_y = np.random.rand(n_splines) * (1 - spline_x) * .5 + spline_x * .75 + .25
spline_fct = interp1d(spline_x, spline_y, kind='cubic')

plt.plot(x, spline_fct(x))

plt.ylabel(r'$v_R(r)$')
plt.xlabel(r'$r$')
plt.yticks([0,1], [r'$0$', r'$\frac{w_{20}}{2\sin{(i)}}$'])

plt.savefig('example_velocity_profile.pdf')
plt.show()
