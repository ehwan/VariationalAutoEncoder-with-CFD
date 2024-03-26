#! /usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import struct
import os

f = open( 're100.dat', 'rb' )
# f.seek( 4*3*256*256*100, os.SEEK_SET )

W = 512
H = 256

Vx = np.zeros( (H,W) )
Vy = np.zeros( (H,W) )

for y in range(H):
  for x in range(W):
    buf = f.read( 8 )
    f2 = struct.unpack( "ff", buf )
    Vx[y][x] = f2[0]
    Vy[y][x] = f2[1]

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