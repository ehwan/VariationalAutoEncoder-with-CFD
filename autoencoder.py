#! /usr/bin/python3

import torch
import data_loader

class AutoEncoder(torch.nn.Module):
  def __init__(self):
    super(AutoEncoder, self).__init__()
    self.encoder = torch.nn.Sequential(
      torch.nn.Conv2d(2, 32, 3, padding=1 ), # 512x256 -> 512x256
      torch.nn.ReLU(),
      torch.nn.Conv2d(32, 32, 3, padding=1 ),
      torch.nn.ReLU(),
      torch.nn.Conv2d(32, 64, 3, padding=1, stride=2 ), # 512x256 -> 256x128
      torch.nn.ReLU(),
      torch.nn.Conv2d(64, 64, 3, padding=1 ),
      torch.nn.ReLU(),
      torch.nn.Conv2d(64, 128, 3, padding=1, stride=2 ), # 256x128 -> 128x64
      torch.nn.ReLU(),
      torch.nn.Conv2d(128, 128, 3, padding=1),
      torch.nn.ReLU(),
      torch.nn.Conv2d(128, 256, 3, padding=1, stride=2), # 128x64 -> 64x32
      torch.nn.ReLU(),
      torch.nn.Conv2d(256, 256, 3, padding=1),
      torch.nn.ReLU(),
      torch.nn.Conv2d(256, 512, 3, padding=1, stride=2), # 64x32 -> 32x16 
      torch.nn.ReLU(),
      torch.nn.Conv2d(512, 512, 3, padding=1),
      torch.nn.ReLU(),
      torch.nn.Conv2d(512, 512, 3, padding=1, stride=2), # 32x16 -> 16x8
      torch.nn.ReLU(),
      torch.nn.Conv2d(512, 512, 3, padding=1),
      torch.nn.ReLU(),
      torch.nn.Conv2d(512, 512, 3, padding=1, stride=2), # 16x8 -> 8x4
      torch.nn.ReLU(),
      torch.nn.Flatten(),
      torch.nn.Linear( 16384, 4096 ),
      torch.nn.SiLU(),
      torch.nn.Linear( 4096, 1024 ),
      torch.nn.SiLU(),
      torch.nn.Linear( 1024, 256 ),
      torch.nn.SiLU(),
      torch.nn.Linear( 256, 32 )
    )
    self.decoder = torch.nn.Sequential(
      torch.nn.Linear( 32, 256 ),
      torch.nn.SiLU(),
      torch.nn.Linear( 256, 1024),
      torch.nn.SiLU(),
      torch.nn.Linear( 1024, 4096),
      torch.nn.SiLU(),
      torch.nn.Linear( 4096, 16384 ),
      torch.nn.SiLU(),
      torch.nn.Unflatten(1, (512, 4, 8)),
      torch.nn.ConvTranspose2d( 512, 512, 3, padding=1 ),
      torch.nn.ReLU(),
      torch.nn.ConvTranspose2d( 512, 512, 3, padding=1, stride=2, output_padding=1 ), # 8x4 -> 16x8
      torch.nn.ReLU(),
      torch.nn.ConvTranspose2d( 512, 512, 3, padding=1 ),
      torch.nn.ReLU(),
      torch.nn.ConvTranspose2d( 512, 512, 3, padding=1, stride=2, output_padding=1 ), # 16x8 -> 32x16
      torch.nn.ReLU(),
      torch.nn.ConvTranspose2d( 512, 512, 3, padding=1 ),
      torch.nn.ReLU(),
      torch.nn.ConvTranspose2d( 512, 256, 3, padding=1, stride=2, output_padding=1 ), # 32x16 -> 64x32
      torch.nn.ReLU(),
      torch.nn.ConvTranspose2d( 256, 256, 3, padding=1 ),
      torch.nn.ReLU(),
      torch.nn.ConvTranspose2d( 256, 128, 3, padding=1, stride=2, output_padding=1 ), # 64x32 -> 128x64
      torch.nn.ReLU(),
      torch.nn.ConvTranspose2d( 128, 128, 3, padding=1 ),
      torch.nn.ReLU(),
      torch.nn.ConvTranspose2d( 128, 64, 3, padding=1, stride=2, output_padding=1 ), # 128x64 -> 256x128
      torch.nn.ReLU(),
      torch.nn.ConvTranspose2d( 64, 64, 3, padding=1 ),
      torch.nn.ReLU(),
      torch.nn.ConvTranspose2d( 64, 32, 3, padding=1, stride=2, output_padding=1 ), # 256x128 -> 512x256
      torch.nn.ReLU(),
      torch.nn.ConvTranspose2d( 32, 32, 3, padding=1 ),
      torch.nn.ReLU(),
      torch.nn.ConvTranspose2d( 32, 2, 3, padding=1 ),
    )



def main():
  autoencoder = AutoEncoder()
  inputs = torch.load( 're200.pt' )
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

  Epochs = 100
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
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

  torch.save( autoencoder.state_dict(), 're200ae.pt' )
  torch.save( optimizer.state_dict(), 're200ae_optim.pt' )

if __name__ == '__main__':
  main()