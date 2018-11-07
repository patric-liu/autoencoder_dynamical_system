# dynamical

Goal of this project is to model and/or predict complex dynamical systems with the aid of machine learning. 

Core idea: non-linear dynamical systems can be represented in a latent space where the evolution of the system is linear

tester and trainer jupyter notebooks - currently train an autoencoder to reconstruct MNIST data. Eventually, dynamical systems whose observables are high-dimensional and evolve non-linearly will be encoded into a linearly evolving latent space by training an autoencoder on both reconstruction and prediction errors. 

ml_only_prediction - trains a neural network to predict the next state of a simple non-linear dynamical system of a bouncing ball (non-linear at bounces).

matrix_only_prediction notebook - generates samples of a simple class of dynamical system (a ball falling with different initial hieghts, velocity, and acceleration). Given only one observable, height, and state [ht,ht-1,ht-3] a matrix is learned which predicts the next state [ht+1]. 

Currently experimenting with methods that can combine ml_prediction and linear modeling to gain the benefits of both. A linear model cannot learn non-linearities, such as bouncing, while an ml_model is both inaccurate and provides little insight into the nature of the system. The goal is a general framework which will learn the dynamics non-linear system with both linear models and ml_models, and automatically recognize and switch between models depending on the current state. 
