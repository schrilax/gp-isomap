def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

fig = plt.figure(figsize=[12,8])
d = np.hstack([batch_var,s5_min_var])
m = running_mean(d, 50)
plt.plot(range(len(batch_var)),d[0:len(batch_var)],'ok')
plt.plot(range(len(batch_var),len(d)),d[len(batch_var):],'ok',alpha=0.3)
plt.plot(range(len(m)),m,'r',linewidth=4)
lg = plt.legend(['Before Shift','After Shift','Moving Average ($w = 50$)'])
lg.get_frame().set_alpha(1.0)
plt.xlabel('Stream (t) $\Rightarrow$')
plt.ylabel('GP-Isomap Variance')
#plt.annotate('ts',[2000,0.4])
plt.savefig('gsad_results.png',dpi=192,transparent=True)