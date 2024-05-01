# SpeedNeuron: Enhanced DRL for SuperTuxKart Racing

Developed by Yuyang Kang, **SpeedNeuron** is an advanced Deep Reinforcement Learning model designed to autonomously navigate SuperTuxKart, an open-source 3D racing simulator. This project enhances traditional reinforcement learning approaches by integrating Convolutional Neural Networks (CNNs) and Decision Transformers, allowing for real-time decision-making based on dynamic visual inputs. It significantly outperforms both baseline and in-game AI models.

This initiative builds upon the foundational framework developed by Balmaseda, V. & Tomotaki, L. (2022) in their project "TransformerKart: A Full SuperTuxKart AI Controller". [View the original project](https://github.com/vibalcam/deep-rl-supertux-race.git).


## Training Process

The following steps have to be done to train a model:

- Choose the tracks ([Environment](#environment))
- Gather data ([Get Data](#get-data))
- Train CNN autoencoder ([CNN Autoencoder](#cnn-autoencoder))
- Train model

Once the model has been trained and saved, we can evaluate the models by specifying the models to be evaluated in run.py.


## Environment

Gym environment that wraps the [PySuperTuxKart](https://github.com/philkr/pystk) game environment.

To check the environment options or select which tracks to use check environments/pytux.py.
