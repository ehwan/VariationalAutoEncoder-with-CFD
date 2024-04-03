#! /usr/bin/python3

import torch
import math

class Encoder(torch.nn.Module):
  def __init__(self, latent_dim=32):
    super(Encoder, self).__init__()
    C = 8
    conv_activation = torch.nn.SiLU()
    self.latent_dim = latent_dim
    self.encoder = torch.nn.Sequential(
      torch.nn.BatchNorm2d(2),

      torch.nn.Conv2d( 2, C, 3, padding=1, stride=1 ),
      torch.nn.BatchNorm2d( C ),
      conv_activation,
      torch.nn.MaxPool2d( 2, 2 ), # 128x256

      torch.nn.Conv2d( C, C*2, 5, padding=2, stride=1 ),
      torch.nn.BatchNorm2d( C*2 ),
      conv_activation,
      torch.nn.MaxPool2d( 2, 2 ), # 64x128

      torch.nn.Conv2d( C*2, C*4, 5, padding=2, stride=1 ),
      torch.nn.BatchNorm2d( C*4 ),
      conv_activation,
      torch.nn.MaxPool2d( 2, 2 ), # 32x64

      torch.nn.Conv2d( C*4, C*8, 5, padding=2, stride=1 ),
      torch.nn.BatchNorm2d( C*8 ),
      conv_activation,
      torch.nn.MaxPool2d( 2, 2 ), # 16x32

      torch.nn.Conv2d( C*8, C*16, 3, padding=1, stride=1 ),
      torch.nn.BatchNorm2d( C*16 ),
      conv_activation,
      torch.nn.MaxPool2d( 2, 2 ), # 8x16

      torch.nn.Conv2d( C*16, C*32, 3, padding=1, stride=1 ),
      torch.nn.BatchNorm2d( C*32 ),
      conv_activation,
      torch.nn.MaxPool2d( 2, 2 ), # 4x8

      torch.nn.Conv2d( C*32, C*64, 3, padding=1, stride=1 ),
      torch.nn.BatchNorm2d( C*64 ),
      conv_activation,
      torch.nn.MaxPool2d( 2, 2 ), # 2x4

      torch.nn.Flatten(),
    )
    self.mu_layer = torch.nn.Sequential(
      torch.nn.Linear( 4096, 1024 ),
      torch.nn.BatchNorm1d( 1024 ),
      torch.nn.SiLU(),
      torch.nn.Linear( 1024, latent_dim )
    )
    self.logvar_layer = torch.nn.Sequential(
      torch.nn.Linear( 4096, 1024 ),
      torch.nn.BatchNorm1d( 1024 ),
      torch.nn.SiLU(),
      torch.nn.Linear( 1024, latent_dim )
    )

  def forward( self, x ):
    x = self.encoder( x )
    mu = self.mu_layer( x )
    logvar = self.logvar_layer( x )
    return mu, logvar


class Decoder(torch.nn.Module):
  def __init__(self, latent_dim=32):
    super(Decoder, self).__init__()
    C = 8
    self.latent_dim = latent_dim
    conv_activation = torch.nn.SiLU()
    self.decoder = torch.nn.Sequential(
      torch.nn.Linear( latent_dim, 1024 ),
      torch.nn.BatchNorm1d( 1024 ),
      torch.nn.SiLU(),

      torch.nn.Linear( 1024, 4096 ),
      torch.nn.Unflatten( 1, (C*64, 2, 4) ),
      torch.nn.BatchNorm2d( C*64 ),
      conv_activation,

      torch.nn.ConvTranspose2d( C*64, C*32, 3, padding=1, stride=2, output_padding=1 ), # 4x8
      torch.nn.BatchNorm2d( C*32 ),
      conv_activation,

      torch.nn.ConvTranspose2d( C*32, C*16, 3, padding=1, stride=2, output_padding=1 ), # 8x16
      torch.nn.BatchNorm2d( C*16 ),
      conv_activation,

      torch.nn.ConvTranspose2d( C*16, C*8, 3, padding=1, stride=2, output_padding=1 ), # 16x32
      torch.nn.BatchNorm2d( C*8 ),
      conv_activation,

      torch.nn.ConvTranspose2d( C*8, C*4, 5, padding=2, stride=2, output_padding=1 ), # 32x64
      torch.nn.BatchNorm2d( C*4 ),
      conv_activation,

      torch.nn.ConvTranspose2d( C*4, C*2, 5, padding=2, stride=2, output_padding=1 ), # 64x128
      torch.nn.BatchNorm2d( C*2 ),
      conv_activation,

      torch.nn.ConvTranspose2d( C*2, C, 5, padding=2, stride=2, output_padding=1 ), # 128x256
      torch.nn.BatchNorm2d( C ),
      conv_activation,

      torch.nn.ConvTranspose2d( C, 2, 3, padding=1, stride=2, output_padding=1 ), # 256x512
    )

  def forward( self, z ):
    return self.decoder( z )


  

# velocity snapshot at fixed time t ( 2, 256, 512 )
# encode to latent vector ( 32 )
class VariationalAutoEncoder(torch.nn.Module):
  def __init__(self, latent_dim=32):
    super(VariationalAutoEncoder, self).__init__()
    self.encoder = Encoder( latent_dim )
    self.decoder = Decoder( latent_dim )
    self.latent_dim = latent_dim

  def encode( self, x ):
    return self.encoder( x )

  def decode( self, z ):
    return self.decoder( z )

  def reparameterize( self, mu, logvar ):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps*std


  def loss( self, x ):

    # Problem: maximize ELBO
    # --> loss = -ELBO

    # ELBO = E[ log p(x|z) ] - KL[ q(z|x) || p(z) ]

    # KL = -0.5 * sum( 1 + log(sigma^2) - mu^2 - sigma^2 )
    # here, (mu, sigma) is output of the encoder

    # E[ log p(x|z) ] = ?
    # assume the output of decoder follows a normal distribution
    # p(x|z) = exp( -(x-mu)^2 / 2sigma^2 ) / sqrt(2*pi*sigma^2)
    # log p(x|z) = -(x-mu)^2 / 2sigma^2 - 0.5*log(2*pi*sigma^2)
    #            = -(x-mu)^2 / 2sigma^2 - 0.5*(  log(sigma^2) + log(2*pi)  )
    # here, mu is output of the decoder and sigma is parameter of the decoder

    BatchN = x.shape[0]
    mu, logvar = self.encode( x )

    # force the latent distribution to be close to a standard normal distribution
    kl_divergence = -0.5 * torch.sum( 1 + logvar - mu.pow(2) - logvar.exp() )

    # monte carlo samples for z
    L = 8

    l = 0.0
    for sample in range(L):
      z = self.reparameterize( mu, logvar )
      x_hat = self.decode( z )
      l = l + (x_hat-x).pow(2).sum()

    l = l / L
    l = l + kl_divergence
    l = l / BatchN

    return l

def main():
  autoencoder = VariationalAutoEncoder()
  inputs200 = torch.load( 're200.pt' )
  inputs100 = torch.load( 're100.pt' )
  inputs40 = torch.load( 're40.pt' )
  inputs5 = torch.load( 're5.pt' )
  inputs60 = torch.load( 're60.pt' )

  inputs = torch.concatenate( (inputs5, inputs40, inputs60, inputs100, inputs200), dim=0 )

  print( inputs.shape )

  N = inputs.shape[0]

  losses = []
  Epochs = 2000
  BatchSize = 30
  optimizer = torch.optim.Adam( autoencoder.parameters(), lr=0.001 )
  for epoch in range(Epochs):
    print( 'Epoch: {}'.format(epoch) )
    shuffled = inputs[ torch.randperm(N) ]
    for batch in range(0, N, BatchSize):
      x = shuffled[batch:batch+BatchSize]
      print( 'train batch: ', x.shape )
      loss = autoencoder.loss( x )
      print( 'Loss: {}'.format(loss.item()) )
      losses.append( loss.item() )
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    if epoch % 10 == 9:
      torch.save( autoencoder.state_dict(), 'vae.pt' )
      torch.save( optimizer.state_dict(), 'vae_optim.pt' )
      torch.save( losses, 'vae_loss.pt' )


if __name__ == '__main__':
  main()