'''
this will simulate on the `stepper.py` with untrained Reynolds number ( Re = 150 ) 
and plots the error on each timestep
'''

import os

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch

import data_loader
import vae as V
import stepper as stp
import sys

encoder = V.VariationalAutoEncoder()
encoder.load_state_dict( torch.load( 'vae.pt' ) )
encoder.train( False )
stepper = stp.LatentStepper()
stepper.load_state_dict( torch.load( 'stepper.pt' ) )
stepper.train( False )

# current state ( velx, vely )
state = torch.zeros( size=(1,2,256,512), dtype=torch.float32 )
cylinder_mask = torch.ones( (1, 1, 256, 512), dtype=torch.float32 )
dx = 10.0 / 511.0
for y in range(256):
  for x in range(512):
    fx = x*dx
    fy = y*dx
    fx = fx - 2.5
    fy = fy - 2.5
    if fx*fx + fy*fy < 0.5*0.5:
      state[0, 0, y, x] = 0.0
      state[0, 1, y, x] = 0.0
      cylinder_mask[0, 0, y, x] = 0.0
    else:
      state[0, 0, y, x] = 1.0
      state[0, 1, y, x] = 0.0

latent_mu, latent_logvar = encoder.encode( state )

errors_l2 = []
errors_linf = []

# answer
answer150 = data_loader.load_file('re150.dat')
iteration = 0

def step( re ):
  global state
  global latent_mu, latent_logvar
  global iteration
  global answer150
  next_latent = stepper.step( latent_mu, re )
  next_state = encoder.decode( next_latent )
  state = next_state*cylinder_mask
  latent_mu = next_latent
  iteration += 1

  error = (answer150[iteration] - state)
  errors_l2.append( error.pow(2).mean().item() )
  errors_linf.append( error.abs().max().item() )

for i in range(149):
  print( i )
  step(float(150))


plt.plot( errors_l2, label='L_2' )
plt.plot( errors_linf, label='L_inf' )
plt.xlabel( 'iteration' )
plt.ylabel( 'error' )
plt.yscale( 'log' )
plt.legend()
plt.savefig( 'error150.png' )