import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp


d = np.loadtxt('../../data/raw_data/sdss_cutout.csv',dtype='float', skiprows=1,delimiter=',')

print(len(d[:,2]))

RA = d[:,0]
DE = d[:,1]
Z = d[:,2]
u = d[:,3]
g = d[:,4]
r = d[:,5]
i = d[:,6]
z = d[:,7]

Indices = np.where((Z<0.12)&(Z>0.08))

S_o = np.sort(Z)
fig = plt.figure()
plt.step(S_o, np.arange(len(S_o))/len(S_o), label='empirical CDF' )
plt.xlabel('Redshift')
plt.ylabel('eCDF')
plt.legend(loc='best')
plt.show()

plt.scatter(Z,r, s=0.5, alpha=0.1)
plt.xlabel('Redshift')
plt.xlim(0,0.6)
plt.ylabel('r-band magnitude')
plt.show()


Z = Z[Indices]
RA,DE,u,g,r,i,z = RA[Indices],DE[Indices],u[Indices],g[Indices],r[Indices],i[Indices],z[Indices]
plt.scatter(Z,r, s=0.5, alpha=0.1)
plt.xlabel('Redshift 0.08 < Z < 0.12')
plt.xlim(0,0.2)
plt.ylabel('r-band magnitude')
plt.show()