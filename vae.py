#! /usr/bin/python3

import torch
import math
import data_loader

class Encoder(torch.nn.Module):
  def __init__(self, latent_dim=32):
    super(Encoder, self).__init__()
    C = 8
    conv_activation = torch.nn.SiLU()

    # input: 2x256x512
    self.latent_dim = latent_dim
    self.encoder = torch.nn.Sequential(
      torch.nn.Conv2d( 2, C, 3 ), # 254x510
      torch.nn.BatchNorm2d( C ),
      conv_activation,
      torch.nn.MaxPool2d( 2, 2, ceil_mode=True ), # 127x255

      torch.nn.Conv2d( C, C*2, 3 ), # 125x253
      torch.nn.BatchNorm2d( C*2 ),
      conv_activation,
      torch.nn.MaxPool2d( 2, 2, ceil_mode=True ), # 63x127

      torch.nn.Conv2d( C*2, C*4, 3 ), # 61x125
      torch.nn.BatchNorm2d( C*4 ),
      conv_activation,
      torch.nn.MaxPool2d( 2, 2, ceil_mode=True ), # 31x63

      torch.nn.Conv2d( C*4, C*8, 3 ), # 29x61
      torch.nn.BatchNorm2d( C*8 ),
      conv_activation,
      torch.nn.MaxPool2d( 2, 2, ceil_mode=True ), # 15x31

      torch.nn.Conv2d( C*8, C*16, 3 ), # 13x29
      torch.nn.BatchNorm2d( C*16 ),
      conv_activation,
      torch.nn.MaxPool2d( 2, 2, ceil_mode=True ), # 7x15

      torch.nn.Conv2d( C*16, C*32, 3 ), # 5x13
      torch.nn.BatchNorm2d( C*32 ),
      conv_activation,
      torch.nn.MaxPool2d( 2, 2, ceil_mode=True ), # 3x7

      torch.nn.Flatten(),
    )
    flatten_dim = C*32*3*7
    self.mu_layer = torch.nn.Sequential(
      torch.nn.Linear( flatten_dim, 1024 ),
      torch.nn.BatchNorm1d( 1024 ),
      torch.nn.SiLU(),
      torch.nn.Linear( 1024, latent_dim )
    )
    self.logvar_layer = torch.nn.Sequential(
      torch.nn.Linear( flatten_dim, 1024 ),
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
    flatten_dim = C*32*3*7
    self.latent_dim = latent_dim
    conv_activation = torch.nn.SiLU()
    self.decoder = torch.nn.Sequential(
      torch.nn.Linear( latent_dim, 1024 ),
      torch.nn.BatchNorm1d( 1024 ),
      torch.nn.SiLU(),

      torch.nn.Linear( 1024, flatten_dim ),
      torch.nn.Unflatten( 1, (C*32, 3, 7) ),
      torch.nn.BatchNorm2d( C*32 ),
      conv_activation,

      torch.nn.ConvTranspose2d( C*32, C*16, 3, padding=0, stride=2, output_padding=0 ),
      torch.nn.BatchNorm2d( C*16 ),
      conv_activation,

      torch.nn.ConvTranspose2d( C*16, C*8, 3, padding=0, stride=2, output_padding=0 ),
      torch.nn.BatchNorm2d( C*8 ),
      conv_activation,

      torch.nn.ConvTranspose2d( C*8, C*4, 3, padding=0, stride=2, output_padding=0 ),
      torch.nn.BatchNorm2d( C*4 ),
      conv_activation,

      torch.nn.ConvTranspose2d( C*4, C*2, 3, padding=0, stride=2, output_padding=0 ),
      torch.nn.BatchNorm2d( C*2 ),
      conv_activation,

      torch.nn.ConvTranspose2d( C*2, C, 3, padding=0, stride=2, output_padding=0 ),
      torch.nn.BatchNorm2d( C ),
      conv_activation,

      torch.nn.ConvTranspose2d( C, 2, 3, padding=0, stride=2, output_padding=1 ),
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
    # p(x|z) = exp( -0.5 * (x-mu)^2 / sigma^2 ) / sqrt(2*pi*sigma^2)
    # log p(x|z) = -0.5 * (x-mu)^2 / sigma^2 - 0.5*log(2*pi*sigma^2)
    #            = -0.5 * (x-mu)^2 / sigma^2 - 0.5*(  log(sigma^2) + log(2*pi)  )
    # here, (mu, sigma) is output of the decoder

    BatchN = x.shape[0]
    mu, logvar = self.encode( x )

    # force the latent distribution to be close to a standard normal distribution
    kl_divergence = -0.5 * torch.sum( 1 + logvar - mu.pow(2) - logvar.exp() )

    # monte carlo samples for z
    L = 8

    reconcstruction_error = 0.0
    for sample in range(L):
      z = self.reparameterize( mu, logvar )
      x_mu = self.decode( z )
      lpxz = -0.5 * (x-x_mu).pow(2)
      reconcstruction_error = reconcstruction_error - lpxz.sum()

    reconcstruction_error = reconcstruction_error / L
    l = (reconcstruction_error + kl_divergence)/BatchN

    return l

def main():
  device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
  autoencoder = VariationalAutoEncoder()
  autoencoder = autoencoder.to( device )
  inputs200 = data_loader.load_file( 're200.dat' )
  inputs100 = data_loader.load_file( 're100.dat' )
  inputs60 = data_loader.load_file( 're60.dat' )
  inputs40 = data_loader.load_file( 're40.dat' )
  inputs5 = data_loader.load_file( 're5.dat' )

  inputs = torch.concatenate( (inputs5, inputs40, inputs60, inputs100, inputs200), dim=0 )

  print( inputs.shape )

  N = inputs.shape[0]
  inputs = inputs.detach().to( device )

  losses = []
  Epochs = 300
  BatchSize = 30
  optimizer = torch.optim.Adam( autoencoder.parameters(), lr=0.001 )
  for epoch in range(Epochs):
    print( 'Epoch: {}'.format(epoch) )
    shuffled = inputs[ torch.randperm(N) ].detach().to(device)
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