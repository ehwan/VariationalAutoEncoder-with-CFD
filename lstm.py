#!/usr/bin/python3
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import data_loader
import vae as V

import torch
import torch.nn as nn
import numpy as np

class LSTM(nn.Module):
  def __init__(self, input_size, hidden_size):
    super(LSTM, self).__init__()
    self.hidden_size = hidden_size
    self.lstm = nn.LSTM(input_size, hidden_size)
    self.post_lstm = nn.Linear(hidden_size, input_size)

  def forward(self, x):
    # x.shape = (T, 32)
    x, _ = self.lstm(x)
    # x.shape = (T, hidden)
    x = self.post_lstm(x)
    # x.shape = (T, 32)
    return x

def main():
  device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )

  vae = V.VariationalAutoEncoder()
  vae.load_state_dict( torch.load( 'vae.pt' ) )
  vae.train( False )

  inputs200 = data_loader.load_file( 're200.dat' )
  z_mu, _ = vae.encode( inputs200 )


  num_epochs = 1000
  input_size = 32
  hidden_size = 128

  lstm = LSTM( input_size, hidden_size )
  lstm.train()
  lstm = lstm.to(device)

  mseloss = nn.MSELoss()
  optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001)

  for epoch in range(num_epochs):
    # predict.shape = (T-1, 32)
    predict = lstm(z_mu[:-1])
    answer = z_mu[1:]
    loss = mseloss(predict, answer)
    l = loss.item()

    optimizer.zero_grad()
    loss.backward( retain_graph=True )
    optimizer.step()


    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {l:.4f}')
    if epoch % 10 == 9:
      torch.save( lstm.state_dict(), 'lstm.pt' )
      torch.save( optimizer.state_dict(), 'lstm_optim.pt' )


if __name__ == '__main__':
  main()

