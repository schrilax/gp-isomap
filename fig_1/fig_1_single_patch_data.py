X2 = pd.read_csv('exp_1_batch_gt.csv',header=None)
Xs = pd.read_csv('gt_rp.csv',header=None)
xGrid = pd.read_csv('rgrid.csv',header=None)
Xhat = pd.read_csv('exp_1_batch_lde.csv',header=None)
Y1 = pd.read_csv('exp_1_batch_data.csv',header=None)
Ys = pd.read_csv('data_rp.csv',header=None)
zGrid = pd.read_csv('zgrid.csv',header=None)

X_batch = np.genfromtxt ('exp_1_batch_data.csv', delimiter=",")
y_batch = np.genfromtxt ('exp_1_batch_lde.csv', delimiter=",")

X_gt = np.genfromtxt ('exp_1_batch_gt.csv', delimiter=",")

X_stream = np.genfromtxt ('data_rp.csv', delimiter=",")
y_stream = np.genfromtxt ('gt_rp.csv', delimiter=",")

batch_G = np.genfromtxt ('exp_1_batch_G.csv', delimiter=",")
stream_G = np.genfromtxt ('exp_1_stream_G.csv', delimiter=",")

def rbf(a, b, length):
    sqdist = np.sum(a**2, 1).reshape(-1, 1) + np.sum(b**2, 1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/length) * sqdist)

def exp_rbf(x, length):
    return np.exp(-.5 * (x/length))


B = X_batch.shape[0] # number of batch points
S = X_stream.shape[0] # number of streaming points

batch_G_sq = batch_G ** 2
stream_G_sq = stream_G ** 2

length_scale = 10
sigma = 0.3

K_geod = exp_rbf(batch_G_sq, length_scale)
L_geod = np.linalg.cholesky(K_geod + sigma*np.eye(B))
Lk_geod = np.linalg.solve(L_geod, exp_rbf(stream_G_sq, length_scale))
mu_geod = np.dot(Lk_geod.T, np.linalg.solve(L_geod, y_batch))

fig = plt.figure(figsize=(12, 4))
grid = plt.GridSpec(1, 3, wspace=0.25, hspace=0.1)

ax = plt.subplot(grid[0, 2])
ax.scatter(mu_geod[:,0]+2,mu_geod[:,1]+0.5,c='gray',alpha=0.1)
ax.scatter(xGrid[0],xGrid[1],0.3,c='k',alpha=1)
ax.scatter(Xhat[0]+2,Xhat[1]+0.5,c='k')

ax = plt.subplot(grid[0, 0])
ax.scatter(X2[0],X2[1],c='k')
ax.scatter(Xs[0],Xs[1],c='gray',alpha=0.1)
ax.scatter(xGrid[0],xGrid[1],0.3,c='k',alpha=1)

ax = plt.subplot(grid[0, 1],projection='3d')
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.grid(False)
ax.xaxis.labelpad = 10
ax.yaxis.labelpad = 10
ax.view_init(60, -40)
ax.scatter(zGrid[0],zGrid[1],zGrid[2],s=0.3,c='k',alpha=0.6)
ax.scatter(Ys[0],Ys[1],Ys[2],c='gray',alpha=0.1)
ax.scatter(Y1[0],Y1[1],Y1[2],c='k')
plt.savefig('expt1_plots.png',dpi=192,transparent=True)