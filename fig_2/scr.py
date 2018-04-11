import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize = (12, 9))

M = 3000
N = 1000

sigma_11 = np.genfromtxt ('sigma_11.csv', delimiter=",")
sigma_22 = np.genfromtxt ('sigma_22.csv', delimiter=",")
sigma_33 = np.genfromtxt ('sigma_33.csv', delimiter=",")

sigma_14 = np.genfromtxt ('sigma_14.csv', delimiter=",")
sigma_24 = np.genfromtxt ('sigma_24.csv', delimiter=",")
sigma_34 = np.genfromtxt ('sigma_34.csv', delimiter=",")

sigma_misc = np.zeros(M,)

for idx in range(N):
    sigma_misc[idx] = sigma_11[idx]
    
for idx in range(N):
    sigma_misc[idx + N] = sigma_22[idx]

for idx in range(N):
    sigma_misc[idx + (2*N)] = sigma_33[idx]

indices = np.random.permutation(M)
sigma_misc = sigma_misc[indices]

sigma = np.zeros(N,)

for idx in range(N):
    sigma[idx] = sigma_14[idx]
    
    if (sigma[idx] > sigma_24[idx]):
        sigma[idx] = sigma_24[idx]
    
    if (sigma[idx] > sigma_34[idx]):
        sigma[idx] = sigma_34[idx]

plt.plot(range(0, M), sigma_misc, 'g.')
plt.plot(range(M, M+N), sigma, 'b.')

plt.title('Variance of prediction')
plt.xlabel('Stream S(t)', fontsize=16)
plt.ylabel('Variance', fontsize=16)

plt.show()
