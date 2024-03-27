#! /usr/bin/python3

import torch
import data_loader

# velocity snapshot at fixed time t ( 2, 256, 512 )
# encode to latent vector ( 32 )
class AutoEncoder(torch.nn.Module):
  def __init__(self):
    super(AutoEncoder, self).__init__()
    C = 16
    self.encoder = torch.nn.Sequential(
      torch.nn.Conv2d( 2, C, 5, padding=2, stride=2, padding_mode='replicate' ),
      torch.nn.BatchNorm2d( C ),
      torch.nn.ReLU(),
      torch.nn.Conv2d( C, C*2, 5, padding=2, stride=2 ),
      torch.nn.BatchNorm2d( C*2 ),
      torch.nn.ReLU(),
      torch.nn.Conv2d( C*2, C*4, 5, padding=2, stride=2 ),
      torch.nn.BatchNorm2d( C*4 ),
      torch.nn.ReLU(),
      torch.nn.Conv2d( C*4, C*8, 5, padding=2, stride=2 ),
      torch.nn.BatchNorm2d( C*8 ),
      torch.nn.ReLU(),
      torch.nn.Conv2d( C*8, C*16, 5, padding=2, stride=2 ),
      torch.nn.BatchNorm2d( C*16 ),
      torch.nn.ReLU(),
      torch.nn.Conv2d( C*16, C*32, 5, padding=2, stride=2 ),
      torch.nn.BatchNorm2d( C*32 ),
      torch.nn.ReLU(),

      torch.nn.Conv2d( C*32, C*64, 5, padding=2, stride=2 ),
      torch.nn.BatchNorm2d( C*64 ),
      torch.nn.ReLU(),
      torch.nn.Flatten(),

      torch.nn.Linear(8192, 2048),
      torch.nn.SiLU(),
      torch.nn.Linear( 2048, 128 )
    )
    self.decoder = torch.nn.Sequential(
      torch.nn.Linear( 128, 2048 ),
      torch.nn.SiLU(),

      torch.nn.Linear( 2048, 8192 ),
      torch.nn.Unflatten( 1, (C*64, 2, 4) ),
      torch.nn.BatchNorm2d( C*64 ),
      torch.nn.ReLU(),
      torch.nn.ConvTranspose2d( C*64, C*32, 5, padding=2, stride=2, output_padding=1 ),
      torch.nn.BatchNorm2d( C*32 ),
      torch.nn.ReLU(),

      torch.nn.ConvTranspose2d( C*32, C*16, 5, padding=2, stride=2, output_padding=1 ),
      torch.nn.BatchNorm2d( C*16 ),
      torch.nn.ReLU(),

      torch.nn.ConvTranspose2d( C*16, C*8, 5, padding=2, stride=2, output_padding=1 ),
      torch.nn.BatchNorm2d( C*8 ),
      torch.nn.ReLU(),
      torch.nn.ConvTranspose2d( C*8, C*4, 5, padding=2, stride=2, output_padding=1 ),
      torch.nn.BatchNorm2d( C*4 ),
      torch.nn.ReLU(),
      torch.nn.ConvTranspose2d( C*4, C*2, 5, padding=2, stride=2, output_padding=1 ),
      torch.nn.BatchNorm2d( C*2 ),
      torch.nn.ReLU(),
      torch.nn.ConvTranspose2d( C*2, C, 5, padding=2, stride=2, output_padding=1 ),
      torch.nn.BatchNorm2d( C ),
      torch.nn.ReLU(),
      torch.nn.ConvTranspose2d( C, 2, 5, padding=2, stride=2, output_padding=1 ),
    )

def main():
  autoencoder = AutoEncoder()
  inputs100 = torch.load( 're100.pt' )
  inputs200 = torch.load( 're200.pt' )
  inputs40 = torch.load( 're40.pt' )
  inputs5 = torch.load( 're5.pt' )

  inputs = torch.concatenate( (inputs5, inputs40, inputs100, inputs200), dim=0 )

  print( inputs.shape )

  N = inputs.shape[0]

  # no train in cylinder range
  cylinder_mask = torch.ones( (1, 1, 256, 512), dtype=torch.float32 )
  dx = 10.0 / 511.0
  for y in range(256):
    for x in range(512):
      fx = x * dx
      fy = y * dx
      fx = fx - 2.5
      fy = fy - 2.5
      if fx*fx + fy*fy < 0.5*0.5:
        cylinder_mask[0, 0, y, x] = 0.0

  losses = []
  Epochs = 5000
  BatchSize = 30
  optimizer = torch.optim.Adam( autoencoder.parameters(), lr=0.001 )
  for epoch in range(Epochs):
    print( 'Epoch: {}'.format(epoch) )
    shuffled = inputs[ torch.randperm(N) ]
    for batch in range(0, N, BatchSize):
      x = shuffled[batch:batch+BatchSize]
      print( 'train batch: ', x.shape )
      latent = autoencoder.encoder( x )
      decoded = autoencoder.decoder( latent )*cylinder_mask
      loss = torch.nn.functional.mse_loss( decoded, x )
      print( 'Loss: {}'.format(loss.item()) )
      losses.append( loss.item() )
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    if epoch % 10 == 9:
      torch.save( autoencoder.state_dict(), 'autoencoder.pt' )
      torch.save( optimizer.state_dict(), 'autoencoder_optim.pt' )
      torch.save( losses, 'autoencoder_loss.pt' )


if __name__ == '__main__':
  main()