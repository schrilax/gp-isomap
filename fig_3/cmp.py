import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize = (12, 9))

frac = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

proc_err_gp = np.genfromtxt ('proc_err_gp.csv', delimiter=",")
proc_err_geod = np.genfromtxt ('proc_err_geod.csv', delimiter=",")
proc_err_siso = np.genfromtxt ('proc_err_siso.csv', delimiter=",")

plt.plot(frac, proc_err_gp, marker='s', linestyle='-', label='GP using standard RBF')
plt.plot(frac, proc_err_geod, marker='s', linestyle='-', label='GP using normalized geodesic distance matrix')
plt.plot(frac, proc_err_siso, marker='o', linestyle='-.', label='S-Isomap')

plt.grid(b=True, which='major', color='k', linestyle='-')
plt.grid(b=True, which='minor', color='g', linestyle='-', alpha=0.2)
plt.minorticks_on()

plt.xlabel('Fraction of data used as batch', fontsize=16)
plt.ylabel('Procrustes Error', fontsize=16)
plt.title('GP-Isomap ~ vanilla GP ~ S-Isomap', fontsize=16)

plt.legend(bbox_to_anchor=(1.0, .55), prop={'size': 16})

plt.show()
