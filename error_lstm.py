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
import lstm as stp
import sys

encoder = V.VariationalAutoEncoder()
encoder.load_state_dict( torch.load( 'vae.pt' ) )
encoder.train( False )
stepper = stp.LSTM( 32, 128 )
stepper.load_state_dict( torch.load( 'lstm.pt' ) )
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

errors = []

# answer
answer150 = data_loader.load_file('re150.dat')
latents, _ = encoder.encode( answer150[0:10,:,:] )
latents = latents.reshape(1,10,32)
iteration = 10

def step():
  global state
  global latents
  global iteration
  global answer150
  # latents.shape = (1, T, 32)
  # next_latents.shape = (1, 32)
  next_latents = stepper(latents)
  print( latents.shape )
  next_state = encoder.decode( next_latents )
  state = next_state*cylinder_mask
  if latents.shape[1] >= 10:
    latents = torch.concatenate( [latents[:,1:], next_latents.reshape(1,1,32)], dim=1 )
  else:
    latents = torch.concatenate( [latents, next_latents.reshape(1,1,32)], dim=1 )

  iteration += 1

  error = (answer150[iteration] - state).pow(2).mean().item()
  errors.append(error)

for i in range(139):
  step()


plt.plot( errors )
plt.xlabel( 'iteration' )
plt.ylabel( 'error' )
plt.yscale( 'log' )
plt.savefig( 'error_lstm150.png' )