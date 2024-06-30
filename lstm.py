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
    self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
    self.post_lstm = nn.Sequential(
      nn.Linear(hidden_size, hidden_size),
      nn.SiLU(),
      nn.Linear(hidden_size, hidden_size),
      nn.SiLU(),
      nn.Linear(hidden_size, input_size)
    )


  def forward(self, x):
    # x.shape = (Batch, T, 32)
    x, _ = self.lstm(x)
    # x.shape = (Batch, T, hidden)
    x = self.post_lstm(x[:,-1])
    # x.shape = (Batch, 32)
    return x

# make sequence (N, 32) to training set ( ( N-seq_len, seq_len, 32 ), (N-seq_len, 32) )
def make_training_set( inputs, seq_len ):
  N = inputs.shape[0]
  seq = []
  out = []
  for i in range( N - seq_len ):
    seq.append( inputs[i:i+seq_len] )
    out.append( inputs[i+seq_len] )
  return ( torch.stack(seq), torch.stack(out) )

def main():
  device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )

  vae = V.VariationalAutoEncoder()
  vae.load_state_dict( torch.load( 'vae.pt' ) )
  vae.train( False )

  inputs200 = data_loader.load_file( 're200.dat' )
  z_mu, _ = vae.encode( inputs200 )
  (inputs200, answers200) = make_training_set( z_mu, 10 )
  inputs100 = data_loader.load_file( 're100.dat' )
  z_mu, _ = vae.encode( inputs100 )
  (inputs100, answers100) = make_training_set( z_mu, 10 )
  inputs60 = data_loader.load_file( 're60.dat' )
  z_mu, _ = vae.encode( inputs60 )
  (inputs60, answers60) = make_training_set( z_mu, 10 )
  inputs40 = data_loader.load_file( 're40.dat' )
  z_mu, _ = vae.encode( inputs40 )
  (inputs40, answers40) = make_training_set( z_mu, 10 )
  inputs5 = data_loader.load_file( 're5.dat' )
  z_mu, _ = vae.encode( inputs5 )
  (inputs5, answers5) = make_training_set( z_mu, 10 )

  inputs = torch.concatenate( [inputs200, inputs100, inputs60, inputs40, inputs5], dim=0 )
  answers = torch.concatenate( [answers200, answers100, answers60, answers40, answers5], dim=0 )
  print( f'input shape: {inputs.shape}' )


  num_epochs = 3000
  input_size = 32
  hidden_size = 128
  batch_size = 35

  lstm = LSTM( input_size, hidden_size )
  lstm.train()
  lstm = lstm.to(device)

  mseloss = nn.MSELoss()
  optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.75)

  for epoch in range(num_epochs):
    # shuffle data
    indices = torch.randperm( inputs.shape[0] )
    shuffled_inputs = inputs[indices].detach().to(device)
    shuffled_answers = answers[indices].detach().to(device)
    epoch_loss = 0.0
    for batch in range(0,inputs.shape[0],batch_size):
      batch_input = shuffled_inputs[batch:batch+batch_size]
      batch_answer = shuffled_answers[batch:batch+batch_size]
      predict = lstm(batch_input)
      loss = mseloss(predict, batch_answer)
      epoch_loss += loss.item()
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    scheduler.step()
    print( f'LR: {scheduler.get_last_lr()}' )

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}')
    if epoch % 10 == 9:
      torch.save( lstm.state_dict(), 'lstm.pt' )
      torch.save( optimizer.state_dict(), 'lstm_optim.pt' )


if __name__ == '__main__':
  main()

