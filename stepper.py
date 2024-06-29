#! /usr/bin/python3

import torch
import data_loader
import vae as V
import math

# map reynolds number [1, 200) -> [0, 1)
def normalize_reynolds( re ):
  # return re
  return math.log( re )

# takes a latent vector and reynolds number, (N, 33)
# outputs a new latent vector (N, 32)
class LatentStepper(torch.nn.Module):
  def __init__(self):
    super(LatentStepper, self).__init__()

    latent_size = 32
    hidden_size = 128
    hidden_layers = 6

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
  device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
  print( 'loading autoencoder...' )
  encoder = V.VariationalAutoEncoder()
  encoder.load_state_dict( torch.load( 'vae.pt' ) )
  encoder.train( False )

  print( 'loading datasets...' )

  re200raw = data_loader.load_file( 're200.dat' )
  re200mu, _ = encoder.encode( re200raw )
  del re200raw
  print( 're200 end' )

  re100raw = data_loader.load_file( 're100.dat' )
  re100mu, _ = encoder.encode( re100raw )
  del re100raw
  print( 're100 end' )

  re60raw = data_loader.load_file( 're60.dat' )
  re60mu, _ = encoder.encode( re60raw )
  del re60raw
  print( 're60 end' )

  re40raw = data_loader.load_file( 're40.dat' )
  re40mu, _ = encoder.encode( re40raw )
  del re40raw
  print( 're40 end' )

  re5raw = data_loader.load_file( 're5.dat' )
  re5mu, _ = encoder.encode( re5raw )
  del re5raw
  print( 're5 end' )

  N = re200mu.shape[0]

  prestep_mu = torch.concatenate(
    [
      re200mu[0:-1],
      re100mu[0:-1],
      re60mu[0:-1],
      re40mu[0:-1],
      re5mu[0:-1]
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
  inputs = torch.concatenate( [prestep_mu, prestep_reynolds], dim=1 )
  answer = torch.concatenate(
    [
      re200mu[1:],
      re100mu[1:],
      re60mu[1:],
      re40mu[1:],
      re5mu[1:]
    ],
    dim=0
  )


  Epochs = 5000
  BatchSize = 30

  stepper = LatentStepper()
  stepper.train( True )
  optimizer = torch.optim.Adam( stepper.parameters(), lr=0.001 )
  scheduler = torch.optim.lr_scheduler.StepLR( optimizer, step_size=150, gamma=0.85 )

  stepper = stepper.to( device )
  inputs = inputs.detach().to( device )
  answer = answer.detach().to( device )

  min_loss = 1e+9
  losses = []
  for epoch in range(Epochs):
    shuffled_indices = torch.randperm( inputs.shape[0] )
    shuffled_inputs = inputs[shuffled_indices].detach().to(device)
    shuffled_answer = answer[shuffled_indices].detach().to(device)
    avg_loss = 0

    for batch in range(0, inputs.shape[0], BatchSize):
      batch_inputs = shuffled_inputs[batch:batch+BatchSize]
      batch_answer = shuffled_answer[batch:batch+BatchSize]

      batch_predict = stepper( batch_inputs )
      loss = torch.nn.functional.mse_loss( batch_predict, batch_answer )

      optimizer.zero_grad()
      loss.backward()
      avg_loss = avg_loss + loss.item()
      optimizer.step()
    
    scheduler.step()
    # print( f'lr: {optimizer.param_groups[0]['lr']}' )

    avg_loss = avg_loss / (inputs.shape[0]//BatchSize)
    losses.append( avg_loss )
    print( "Epoch {} loss: {}".format( epoch, avg_loss ) )

    if epoch % 100 == 0:
      torch.save( stepper.state_dict(), 'stepper.pt' )
      torch.save( optimizer.state_dict(), 'stepper_optim.pt' )
      torch.save( losses, 'stepper_loss.pt' )



if __name__ == '__main__':
  main()