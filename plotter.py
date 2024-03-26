#! /usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import torch

import autoencoder
import stepper

encoder = autoencoder.AutoEncoder()
encoder.load_state_dict( torch.load( 're200ae.pt' ) )
stepper = stepper.LatentStepper()
stepper.load_state_dict( torch.load( 're200step.pt' ) )

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

def step():
  global state
  latent = encoder.encoder( state )
  print( latent )
  next_latent = stepper( latent )
  next_state = encoder.decoder( next_latent )
  state = next_state*cylinder_mask

def plot():
  Vx = state[0][0].detach().numpy()
  Vy = state[0][1].detach().numpy()
  plt.imshow( np.sqrt(Vx**2 + Vy**2) )
  # plt.quiver( Vx, Vy, np.sqrt(Vx**2 + Vy**2) )
  plt.colorbar()
  plt.show()

for i in range(10):
  plot()
  step()
  step()