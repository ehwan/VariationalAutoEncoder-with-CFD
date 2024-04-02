#!/usr/bin/python3
import torch
import numpy as np
import matplotlib.pyplot as plt
import math

import vae as V

def main():
  vae = V.VariationalAutoEncoder()
  vae.load_state_dict( torch.load( 'vae.pt' ) )
  vae.train( False )

  inputs200 = torch.load( 're200.pt' )
  mu, logvar = vae.encode( inputs200 )
  print( 'std: {}'.format(math.exp(0.5*vae.decoder.logvar.item()) ) )

  plt.imshow( mu.detach().numpy() )
  plt.colorbar()
  plt.title( 'mu' )
  plt.show()

  plt.imshow( np.exp(0.5*logvar.detach().numpy()) )
  plt.colorbar()
  plt.title( 'sigma' )
  plt.show()

if __name__ == '__main__':
  main()

