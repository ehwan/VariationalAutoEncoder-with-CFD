# Variational Auto Encoder with CFD
A simulation of wake behind cylinder. dimensionality reduction by variational auto encoder


## Introduction

This is a simple simulation of wake behind a cylinder. 
The simulation is done using Lattice Boltzmann Method. ( see `cylinder.cpp` )

The simulation is done for three different Reynolds numbers($Re = 5, 40, 200$). 

The simulation data is then used to train an auto encoder to reduce the dimensionality of the data to 32-sized latent space. ( `autoencoder.py`)

We then defined a neural network to predict time integral `step()` function on the latent space.
Neural network takes 32-sized latent vector **z** and Reynolds number $Re$ as input and predicts the next latent vector **z'**. ( `stepper.py` )

We will see that the neural network is able to predict the next latent vector with untrained Reynolds number.

## Results

### Loss of AutoEncoder

![](result/autoencoder_loss.png)


### Loss of LatentStepper

![](result/latent_stepper_loss.png)

### $Re = 5$

![](result/plots5/plot0010.png)

### $Re = 30$

![](result/plots30/plot0030.png)

### $Re = 150$

![](result/plots150/plot0100.png)