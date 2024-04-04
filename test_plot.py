#! /usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import struct
import data_loader

buf = data_loader.load_file( 're100.dat', 150 )

Vx = buf[0,0]
Vy = buf[0,1]

plt.quiver( Vx, Vy, np.sqrt(Vx**2 + Vy**2) )
# plt.imshow( Vx )
plt.colorbar()
plt.show()

plt.imshow( Vx, origin='lower' )
plt.colorbar()
plt.title( 'Vx' )
plt.show()

plt.imshow( Vy, origin='lower' )
plt.colorbar()
plt.title( 'Vy' )
plt.show()