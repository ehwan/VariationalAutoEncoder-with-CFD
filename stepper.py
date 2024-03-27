#! /usr/bin/python3

import torch
import autoencoder
import math
import matplotlib.pyplot as plt

# map reynolds number [1, 200) -> [0, 1)
def normalize_reynolds( re ):
  return math.log( re ) / 6.0

class LatentStepper(torch.nn.Module):
  def __init__(self):
    super(LatentStepper, self).__init__()

    hidden_size = 96

    self.stepper = torch.nn.Sequential(
      torch.nn.Linear( 33, hidden_size),
      # torch.nn.BatchNorm1d(hidden_size),
      torch.nn.SiLU(),

      torch.nn.Linear( hidden_size, hidden_size),
      # torch.nn.BatchNorm1d(hidden_size),
      torch.nn.SiLU(),

      torch.nn.Linear( hidden_size, hidden_size),
      # torch.nn.BatchNorm1d(hidden_size),
      torch.nn.SiLU(),

      torch.nn.Linear( hidden_size, hidden_size),
      # torch.nn.BatchNorm1d(hidden_size),
      torch.nn.SiLU(),

      torch.nn.Linear( hidden_size, hidden_size),
      # torch.nn.BatchNorm1d(hidden_size),
      torch.nn.SiLU(),

      torch.nn.Linear( hidden_size, hidden_size),
      # torch.nn.BatchNorm1d(hidden_size),
      torch.nn.SiLU(),

      torch.nn.Linear( hidden_size, hidden_size),
      # torch.nn.BatchNorm1d(hidden_size),
      torch.nn.SiLU(),

      torch.nn.Linear( hidden_size, 32)
    )

  def forward( self, latent_and_reynolds ):
    return self.stepper( latent_and_reynolds )

  def step( self, latent, re ):
    re_tensor = torch.tensor( [[normalize_reynolds(re)]], dtype=torch.float32 )
    return self( torch.hstack( [latent, re_tensor.broadcast_to( latent.shape[0], 1 )] ) )

def main():
  print( 'loading autoencoder...' )
  encoder = autoencoder.AutoEncoder()
  encoder.load_state_dict( torch.load( '../autoencoder.pt' ) )

  print( 'loading datasets...' )
  re200raw = torch.load( 're200.pt' )
  print( 're200 end' )
  re100raw = torch.load( 're100.pt' )
  print( 're100 end' )
  re40raw = torch.load( 're40.pt' )
  print( 're40 end' )
  re5raw = torch.load( 're5.pt' )
  print( 're5 end' )

  N = re200raw.shape[0]
  print( 'converting to latent space...' )
  re200latents = encoder.encoder( re200raw ).detach().clone()
  re100latents = encoder.encoder( re100raw ).detach().clone()
  re40latents = encoder.encoder( re40raw ).detach().clone()
  re5latents = encoder.encoder( re5raw ).detach().clone()

  def concat_reynolds_to_latent( latent, re ):
    return torch.concatenate( [
        latent, 
        torch.tensor( [[re]], dtype=torch.float32 ).broadcast_to( latent.shape[0], 1 )
      ],
      dim = 1
    )

  prestep_latents = torch.concatenate(
    [
      concat_reynolds_to_latent( re200latents[0:N-1], normalize_reynolds(200.0) ),
      concat_reynolds_to_latent( re100latents[0:N-1], normalize_reynolds(100.0) ),
      concat_reynolds_to_latent( re40latents[0:N-1], normalize_reynolds(40.0) ),
      concat_reynolds_to_latent( re5latents[0:N-1], normalize_reynolds(5.0) )
    ],
    dim=0
  )
  poststep_latents = torch.concatenate(
    [re200latents[1:N], re100latents[1:N], re40latents[1:N], re5latents[1:N]],
    dim=0
  )
  print( prestep_latents.shape )
  print( poststep_latents.shape )

  del re200raw
  del re100raw
  del re40raw
  del re5raw

  Epochs = 10000
  BatchSize = 30

  stepper = LatentStepper()
  stepper.train( True )
  optimizer = torch.optim.Adam( stepper.parameters(), lr=0.001 )

  losses = []
  for epoch in range(Epochs):
    print( 'Epoch: {}'.format(epoch) )
    shuffled_indices = torch.randperm( prestep_latents.shape[0] )
    shuffled_pre = prestep_latents[shuffled_indices]
    shuffled_post = poststep_latents[shuffled_indices]
    avg_loss = 0.0
    for batch in range(0, prestep_latents.shape[0], BatchSize):
      input = shuffled_pre[batch:batch+BatchSize]
      predict_output = stepper( input )
      output = shuffled_post[batch:batch+BatchSize]
      loss = torch.nn.functional.mse_loss( predict_output, output )
      print( loss.item() )
      optimizer.zero_grad()
      loss.backward()
      avg_loss = avg_loss + loss.item()
      optimizer.step()
    avg_loss = avg_loss / (prestep_latents.shape[0]//BatchSize)
    losses.append( avg_loss )

  if epoch % 10 == 9:
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