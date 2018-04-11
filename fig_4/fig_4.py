
import math
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

def get_min_stream_variance(v1, v2, v3, v4):
    N = v1.shape[0]
    var_min = np.zeros((N,))
    
    for idx in range(N):
        var_min[idx] = min(min(v1[idx], v2[idx]), min(v3[idx], v4[idx]))
    
    return var_min

S1 = 571
S2 = 707
S3 = 380
S4 = 473
S5 = 623

nvar_11 = np.genfromtxt('nvar_11.csv', delimiter=",")
nvar_12 = np.genfromtxt('nvar_12.csv', delimiter=',')
nvar_13 = np.genfromtxt('nvar_13.csv', delimiter=",")
nvar_14 = np.genfromtxt('nvar_14.csv', delimiter=',')
nvar_15 = np.genfromtxt('nvar_15.csv', delimiter=",")

nvar_21 = np.genfromtxt('nvar_21.csv', delimiter=',')
nvar_22 = np.genfromtxt('nvar_22.csv', delimiter=',')
nvar_23 = np.genfromtxt('nvar_23.csv', delimiter=',')
nvar_24 = np.genfromtxt('nvar_24.csv', delimiter=',')
nvar_25 = np.genfromtxt('nvar_25.csv', delimiter=',')

nvar_31 = np.genfromtxt('nvar_31.csv', delimiter=',')
nvar_32 = np.genfromtxt('nvar_32.csv', delimiter=',')
nvar_33 = np.genfromtxt('nvar_33.csv', delimiter=',')
nvar_34 = np.genfromtxt('nvar_34.csv', delimiter=',')
nvar_35 = np.genfromtxt('nvar_35.csv', delimiter=',')

nvar_41 = np.genfromtxt('nvar_41.csv', delimiter=',')
nvar_42 = np.genfromtxt('nvar_42.csv', delimiter=',')
nvar_43 = np.genfromtxt('nvar_43.csv', delimiter=',')
nvar_44 = np.genfromtxt('nvar_44.csv', delimiter=',')
nvar_45 = np.genfromtxt('nvar_45.csv', delimiter=',')

s1_min_var = get_min_stream_variance(nvar_11, nvar_21, nvar_31, nvar_41)
s2_min_var = get_min_stream_variance(nvar_12, nvar_22, nvar_32, nvar_42)
s3_min_var = get_min_stream_variance(nvar_13, nvar_23, nvar_33, nvar_43)
s4_min_var = get_min_stream_variance(nvar_14, nvar_24, nvar_34, nvar_44)
s5_min_var = get_min_stream_variance(nvar_15, nvar_25, nvar_35, nvar_45)

batch_var = np.concatenate((s1_min_var, s2_min_var), axis=0)
batch_var = np.concatenate((batch_var, s3_min_var), axis=0)
batch_var = np.concatenate((batch_var, s4_min_var), axis=0)

np.random.shuffle(batch_var)

plt.plot(range(0, S1+S2+S3+S4), batch_var, 'g.')
plt.plot(range(S1+S2+S3+S4, S1+S2+S3+S4+S5), s5_min_var, 'b.')

plt.title('Variance of prediction', fontsize=16)
plt.xlabel('Stream(t) ->', fontsize=16)
plt.ylabel('Variance ->', fontsize=16)

plt.show()