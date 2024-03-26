#! /usr/bin/python3 

import torch
import os
import struct

# load first N (~150) frames from a file
def load_file( filename, N=150 ):
  print( 'loading ' + filename )
  f = open( filename, 'rb' )

  # N frames per file
  ret = torch.zeros( (N, 2, 256, 512), dtype=torch.float32 )
  for frame in range(N):
    print( 'frame ' + str(frame) )
    # vels
    buffer = f.read( 4*2*256*512 )
    vel = struct.unpack( 'ff'*(256*512), buffer )
    for y in range(256):
      for x in range(512):
        idx = y*512 + x
        ret[frame, 0, y, x] = vel[idx*2]
        ret[frame, 1, y, x] = vel[idx*2+1]

  return ret

def load_files( files, N=150 ):
  ret = torch.zeros( (len(files), N, 2, 256, 512), dtype=torch.float32 )
  for i in range(len(files)):
    ret[i] = load_file( files[i], N )
  return ret.reshape( -1, 2, 256, 512)

def load_directory( dirname, N=150 ):
  files = os.listdir( dirname )
  for i in range(files):
    files[i] = dirname + '/' + files[i]
  return load_files( files, N )


def main():
  data = load_file( 're200.dat', 150 )
  print( data.shape )
  # convert binary to python object for faster loading
  torch.save( data, 're200.pt' )

if __name__ == '__main__':
  main()