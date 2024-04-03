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

    latent_size = 32
    hidden_size = 256
    hidden_layers = 8

    self.stepper = torch.nn.Sequential(
      torch.nn.Linear( latent_size+1, hidden_size ),
      torch.nn.SiLU()
    )

    for l in range(hidden_layers):
      self.stepper.append( torch.nn.Linear( hidden_size, hidden_size ) )
      self.stepper.append( torch.nn.SiLU() )

    self.stepper.append( torch.nn.Linear( hidden_size, latent_size ) )


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

    shuffled_pre_mu = prestep_mu[shuffled_indices].detach().clone()
    shuffled_pre_logvar = prestep_logvar[shuffled_indices].detach().clone()
    shuffled_pre_reynolds = prestep_reynolds[shuffled_indices].detach().clone()
    shuffled_post_mu = poststep_mu[shuffled_indices].detach().clone()
    shuffled_post_logvar = poststep_logvar[shuffled_indices].detach().clone()

    L = 8

    # p(x|z) = exp( -(x-mu)^2 / 2sigma^2 ) / sqrt(2*pi*sigma^2)
    # log p(x|z) = -(x-mu)^2 / 2sigma^2 - 0.5*log(2*pi*sigma^2)
    #            = -(x-mu)^2 / 2sigma^2 - 0.5*(  log(sigma^2) + log(2*pi)  )
    # here, mu and sigma belongs to post step latent distributions

    avg_loss = 0.0

    for batch in range(0, prestep_mu.shape[0], BatchSize):
      pre_mu = shuffled_pre_mu[batch:batch+BatchSize]
      pre_logvar = shuffled_pre_logvar[batch:batch+BatchSize]
      pre_reynolds = shuffled_pre_reynolds[batch:batch+BatchSize]
      post_mu = shuffled_post_mu[batch:batch+BatchSize]
      post_logvar = shuffled_post_logvar[batch:batch+BatchSize]
      log_p = 0.0
      for sample in range(L):
        pre_z = encoder.reparameterize( pre_mu, pre_logvar )
        predict_post_z = stepper( torch.hstack( [pre_z, pre_reynolds] ) )
        lp = -(post_mu-predict_post_z).pow(2) / (2*post_logvar.exp()) - 0.5*(math.log(2*math.pi) + post_logvar)
        log_p = log_p + lp.sum()
      log_p = log_p / L / BatchSize

      loss = -log_p

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