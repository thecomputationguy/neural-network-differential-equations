import numpy as np
import torch
import nodepy.linear_multistep_method as lm
import torch.nn as nn
import matplotlib.pyplot as plt

class ODE_NN(nn.Module):
    """Class that implements the neural-ODE solver based on the paper https://arxiv.org/abs/1801.01236

    """    
    def __init__(self, dt: float, X: np.ndarray, M: int, scheme: str, num_neurons: int = 128, num_layers: int =1):
        """Constructor for the class.

        Parameters
        ----------
        dt : float
            time-step for the ODE.
        X : np.ndarray
            training data.
        M : int
            number of Adam-Moulton steps.
        scheme : str
            Multistep scheme to be used.
        num_neurons : int, optional
            number of neurons in each layer, by default 128
        num_layers : int, optional
            number of layers, by default 1
        """        
        super().__init__()
        self.dt = dt
        self._X_reshaped = torch.Tensor(X.reshape((1, X.shape[0], X.shape[1])))

        self._num_trajectories = self._X_reshaped.shape[0] # S
        self._num_timesteps = self._X_reshaped.shape[1] # N
        self._num_dimensions = self._X_reshaped.shape[2] # D
        self.num_neurons = num_neurons
        self.num_layers = num_layers

        # Create the neural network

        self.model = nn.Sequential(
            nn.Linear(self._num_dimensions, self.num_neurons), 
            nn.Tanh(), 
            nn.Linear(self.num_neurons, self._num_dimensions))
        print(self.model)
        self._losses = []

        # Specifics of the ODE scheme used

        self.M = M # Adam-moulton steps
        self._switch = {'AM': lm.Adams_Moulton,
                  'AB': lm.Adams_Bashforth,
                  'BDF': lm.backward_difference_formula}
        self._method = self._switch[scheme](M)
        self.alpha = np.float32(-self._method.alpha[::-1])
        self.beta = np.float32(self._method.beta[::-1])

    def forward(self, X) -> torch.Tensor:        
        _X_reshaped = torch.reshape(X, (self._num_trajectories*(self._num_timesteps - self.M), self._num_dimensions))
        f = self.model(_X_reshaped).reshape((-1, self._num_dimensions))

        return f

    def _y_predicted(self) -> torch.Tensor:
        """Calculates the quantity y_n, in Equation 5 of https://arxiv.org/abs/1801.01236

        Returns
        -------
        torch.Tensor
            y_n
        """              
        y = self.alpha[0] * self._X_reshaped[:, self.M:, :] + self.dt * self.beta[0] * self.model(torch.Tensor(self._X_reshaped[:, self.M:, :]))
        for m in range(1, self.M + 1):
            y = y + self.alpha[m] * self._X_reshaped[:, self.M-m:-m, :] + self.dt * self.beta[m] * self.model(torch.Tensor(self._X_reshaped[:, self.M-m:-m, :]))

        return y

    def _plot_loss(self) -> None:
        """Plots the loss history.
        """        
        plt.plot(self._losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss History')
        plt.show()

    def train(self, epochs: int) -> None:
        """Main training loop.

        Parameters
        ----------
        epochs : int
            number of epoch for which to train the network.
        """        
        optimizer = torch.optim.Adam(self.model.parameters())
        for epoch in range(epochs):
            optimizer.zero_grad()            
            y = self._y_predicted().reshape((-1, self._num_dimensions))
            loss = self._num_dimensions * torch.sum(y**2)
            self._losses.append(loss.detach())

            if epoch == epochs-1:
                self._plot_loss()
            if(epoch % 200 == 0):
                print(f"Epoch : {epoch}, Loss : {loss}")

            loss.backward()
            optimizer.step()   
