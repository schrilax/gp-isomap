xGrid = pd.read_csv('rgrid.csv',header=None)
yGrid = []
with open('zgrid.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        yGrid.append([float(row[0]),float(row[1]),float(row[2])])
yGrid = np.array(yGrid)
Y = []
with open('batch_data.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        Y.append([float(row[0]),float(row[1]),float(row[2])])
Y = np.array(Y)
L = np.vstack([np.ones([3000,1]),2*np.ones([1000,1])])

X = []
with open('batch_gt.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        X.append([float(row[0]),float(row[1])])
X = np.array(X)

X1 = pd.read_csv('sisomap_pp_prediction.csv',header=None)
X1 = np.array(X1)
fig = plt.figure(figsize=(12, 4))
grid = plt.GridSpec(1, 3, wspace=0.25, hspace=0.1)

ax = plt.subplot(grid[0, 2])
inds = list(np.where(L == 1)[0])
ax.scatter(X1[inds,0]+2,X1[inds,1]+0.4,c='k')
inds = list(np.where(L == 2)[0])
ax.scatter(X1[inds,0]+2,X1[inds,1]+0.4,c='gray',alpha=0.3)
ax.scatter(xGrid[0],xGrid[1],0.3,c='k',alpha=1)

ax = plt.subplot(grid[0, 0])
inds = list(np.where(L == 1)[0])
ax.scatter(X[inds,0],X[inds,1],c='k')
inds = list(np.where(L == 2)[0])
ax.scatter(X[inds,0],X[inds,1],c='gray',alpha=0.1)
ax.scatter(xGrid[0],xGrid[1],0.3,c='k',alpha=1)

ax = plt.subplot(grid[0, 1],projection='3d')
inds = list(np.where(L == 1)[0])
ax.scatter(Y[inds,0],Y[inds,1],Y[inds,2],c='k')
inds = list(np.where(L == 2)[0])
ax.scatter(Y[inds,0],Y[inds,1],Y[inds,2],c='gray',alpha=0.1)
ax.scatter(yGrid[:,0],yGrid[:,1],yGrid[:,2],s=0.3,c='gray',alpha=1)
ax.grid(False)
ax.xaxis.labelpad = 10
ax.yaxis.labelpad = 10
ax.view_init(60, -40)
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

plt.savefig('expt2_plots.png',dpi=192,transparent=True)