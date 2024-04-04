import torch
import os
import struct

# load first N (~150) frames from a file
def load_file( filename, N=150 ):
  print( 'loading ' + filename )
  f = open( filename, 'rb' )
  buf = torch.frombuffer( f.read(), dtype=torch.float32 )
  buf = buf.reshape( N, 256, 512, 2 )
  buf = buf.permute( 0, 3, 1, 2 )
  return buf.detach().clone()

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