import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp


d = np.loadtxt('../data/raw_data/sdss_cutout.csv', dtype='float', skiprows=1, delimiter=',')

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

#task2

color = u-r
blue = np.where(color<=2.3)
red = np.where(color>2.3)

plt.scatter(r[blue],color[blue], s=0.5, alpha=0.1,color='blue',label='Blue galaxies' )
plt.scatter(r[red],color[red], s=0.5, alpha=0.1,color='red',label='Red galaxies')
plt.legend(loc='best')
plt.xlabel('r-band mag')
plt.ylabel('u-r')
plt.show()


#task3

blue_mean, red_mean = np.mean(r[blue]), np.mean(r[red])
blue_std, red_std = np.std(r[blue]), np.std(r[red])
print(blue_mean, red_mean, blue_std, red_std)


#task4

fig, axs = plt.subplots(2, 2, figsize=(25,15))
plt.subplot(2,2,1)
plt.title('Angular Map for blue galaxies',fontsize=20)
plt.xlabel('Rektaszenion', fontsize=15)
plt.ylabel('Declination', fontsize=15)
plt.scatter(RA[blue], DE[blue], s=0.5, alpha=0.1)

plt.subplot(2,2,2)
plt.title('Angular Map for red galaxies',fontsize=20)
plt.xlabel('Rektaszension', fontsize=15)
plt.ylabel('Declination', fontsize=15)
plt.scatter(RA[red], DE[red], s=0.5, alpha=0.1)

plt.subplot(2,2,3)
plt.title('Redshift-space map for blue galaxies',fontsize=20)
plt.xlabel('RA', fontsize=15)
plt.ylabel('Redshift', fontsize=15)
plt.scatter(RA[blue], Z[blue], s=0.5, alpha=0.1)

plt.subplot(2,2,4)
plt.title('Redshift-space map for red galaxies',fontsize=20)
plt.xlabel('RA', fontsize=15)
plt.ylabel('Redshift', fontsize=15)
plt.scatter(RA[red], Z[red], s=0.5, alpha=0.1)

#fig.tight_layout(pad=30.0)