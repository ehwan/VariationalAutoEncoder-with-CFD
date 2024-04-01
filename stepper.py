#! /usr/bin/python3

import torch
import autoencoder
import vae as V
import math
import matplotlib.pyplot as plt

# map reynolds number [1, 200) -> [0, 1)
def normalize_reynolds( re ):
  # return re
  return math.log( re )

class LatentStepper(torch.nn.Module):
  def __init__(self):
    super(LatentStepper, self).__init__()

    hidden_size = 128

    self.stepper = torch.nn.Sequential(
      torch.nn.Linear( 49, hidden_size),
      torch.nn.BatchNorm1d( hidden_size ),
      torch.nn.SiLU(),

      torch.nn.Linear( hidden_size, hidden_size),
      torch.nn.BatchNorm1d( hidden_size ),
      torch.nn.SiLU(),

      torch.nn.Linear( hidden_size, hidden_size),
      torch.nn.BatchNorm1d( hidden_size ),
      torch.nn.SiLU(),

      torch.nn.Linear( hidden_size, hidden_size),
      torch.nn.BatchNorm1d( hidden_size ),
      torch.nn.SiLU(),

      torch.nn.Linear( hidden_size, hidden_size),
      torch.nn.BatchNorm1d( hidden_size ),
      torch.nn.SiLU(),

      torch.nn.Linear( hidden_size, hidden_size),
      torch.nn.BatchNorm1d( hidden_size ),
      torch.nn.SiLU(),

      torch.nn.Linear( hidden_size, hidden_size),
      torch.nn.BatchNorm1d( hidden_size ),
      torch.nn.SiLU(),

      torch.nn.Linear( hidden_size, 48 )
    )

  def forward( self, latent_and_reynolds ):
    return self.stepper( latent_and_reynolds )

  def step( self, latent, re ):
    re_tensor = torch.tensor( [[normalize_reynolds(re)]], dtype=torch.float32 )
    return self( torch.hstack( [latent, re_tensor.broadcast_to( latent.shape[0], 1 )] ) )

def main():
  print( 'loading autoencoder...' )
  encoder = V.VariationalAutoEncoder()
  encoder.load_state_dict( torch.load( 'vae.pt' ) )
  encoder.train( False )

  print( 'loading datasets...' )

  re200raw = torch.load( 're200.pt' )
  re200mu, re200logvar = encoder.encode( re200raw )
  del re200raw
  print( 're200 end' )

  re100raw = torch.load( 're100.pt' )
  re100mu, re100logvar = encoder.encode( re100raw )
  del re100raw
  print( 're100 end' )

  re60raw = torch.load( 're60.pt' )
  re60mu, re60logvar = encoder.encode( re60raw )
  del re60raw
  print( 're60 end' )

  re40raw = torch.load( 're40.pt' )
  re40mu, re40logvar = encoder.encode( re40raw )
  del re40raw
  print( 're40 end' )

  re5raw = torch.load( 're5.pt' )
  re5mu, re5logvar = encoder.encode( re5raw )
  del re5raw
  print( 're5 end' )

  N = re200mu.shape[0]

  prestep_mu = torch.concatenate(
    [
      re200mu[0:N-1],
      re100mu[0:N-1],
      re60mu[0:N-1],
      re40mu[0:N-1],
      re5mu[0:N-1]
    ],
    dim=0
  )
  prestep_logvar = torch.concatenate(
    [
      re200logvar[0:N-1],
      re100logvar[0:N-1],
      re60logvar[0:N-1],
      re40logvar[0:N-1],
      re5logvar[0:N-1]
    ],
    dim=0
  )
  prestep_reynolds = torch.concatenate(
    [
      torch.tensor( [[normalize_reynolds(200.0)]], dtype=torch.float32).broadcast_to( N-1, 1 ),
      torch.tensor( [[normalize_reynolds(100.0)]], dtype=torch.float32).broadcast_to( N-1, 1 ),
      torch.tensor( [[normalize_reynolds(60.0)]], dtype=torch.float32).broadcast_to( N-1, 1 ),
      torch.tensor( [[normalize_reynolds(40.0)]], dtype=torch.float32).broadcast_to( N-1, 1 ),
      torch.tensor( [[normalize_reynolds(5.0)]], dtype=torch.float32).broadcast_to( N-1, 1 )
    ],
    dim=0
  )
  poststep_mu = torch.concatenate(
    [
      re200mu[1:N],
      re100mu[1:N],
      re60mu[1:N],
      re40mu[1:N],
      re5mu[1:N]
    ],
    dim=0
  )
  poststep_logvar = torch.concatenate(
    [
      re200logvar[1:N],
      re100logvar[1:N],
      re60logvar[1:N],
      re40logvar[1:N],
      re5logvar[1:N]
    ],
    dim=0
  )


  Epochs = 100000
  BatchSize = 30

  stepper = LatentStepper()
  stepper.train( True )
  optimizer = torch.optim.Adam( stepper.parameters(), lr=0.001 )

  min_loss = 1e+9
  losses = []
  for epoch in range(Epochs):
    shuffled_indices = torch.randperm( prestep_mu.shape[0] )
    pre_latents = torch.concatenate( 
      [
        encoder.reparameterize( prestep_mu, prestep_logvar ),
        prestep_reynolds
      ],
      dim=1
    )
    post_latents = encoder.reparameterize( poststep_mu, poststep_logvar )
    shuffled_pre = pre_latents[shuffled_indices].detach().clone()
    shuffled_post = post_latents[shuffled_indices].detach().clone()
    avg_loss = 0.0
    for batch in range(0, prestep_mu.shape[0], BatchSize):
      pre = shuffled_pre[batch:batch+BatchSize]
      post = shuffled_post[batch:batch+BatchSize]
      predict_output = stepper( pre )
      loss = torch.nn.functional.mse_loss( predict_output, post )
      optimizer.zero_grad()
      loss.backward()
      avg_loss = avg_loss + loss.item()
      optimizer.step()
    avg_loss = avg_loss / (prestep_mu.shape[0]//BatchSize)
    losses.append( avg_loss )
    print( "Epoch {} loss: {}".format( epoch, avg_loss ) )

    if avg_loss < min_loss:
      min_loss = avg_loss
      torch.save( stepper.state_dict(), 'stepper.pt' )
      torch.save( optimizer.state_dict(), 'stepper_optim.pt' )
      torch.save( losses, 'stepper_loss.pt' )


  plt.plot( losses )
  plt.yscale( 'log' )
  plt.ylabel( 'loss' )
  plt.xlabel( 'epochs' )
  plt.show()

if __name__ == '__main__':
  main()