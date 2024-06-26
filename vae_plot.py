#!/usr/bin/python3
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import data_loader
import vae as V

def main():
  vae = V.VariationalAutoEncoder()
  vae.load_state_dict( torch.load( 'vae.pt', map_location='cpu' ) )
  vae.train( False )

  inputs200 = data_loader.load_file( 're200.dat' )
  mu, logvar = vae.encode( inputs200 )
  x_mu = vae.decode( mu )

  error = (x_mu - inputs200).pow(2)
  error_mean = error.mean( dim=(1,2,3) )
  error_max = torch.max( error, dim=3 ).values
  error_max = torch.max( error_max, dim=2 ).values
  error_max = torch.max( error_max, dim=1 ).values
  error_max = error_max.sqrt()
  # torch.max( error, dim=(1,2,3) )
  print( error_mean )
  print( error_max )

  plt.imshow( mu.detach().numpy() )
  plt.colorbar()
  plt.title( 'mu' )
  plt.show()

  plt.imshow( np.exp(0.5*logvar.detach().numpy()) )
  plt.colorbar()
  plt.title( 'sigma' )
  plt.show()

  plt.imshow( x_mu[100,0].detach().numpy() )
  plt.colorbar()
  plt.title( 'x_mu' )
  plt.show()

if __name__ == '__main__':
  main()

