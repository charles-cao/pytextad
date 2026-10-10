import torch
import torch.nn as nn
import math
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader

class DiffusionDenoiser(nn.Module):
    def __init__(self, input_dim, hidden_dims=[1024], time_emb_dim=16, activation='lrelu', time_emb_type='sinusoidal'):
        super(DiffusionDenoiser, self).__init__()
        # Timestep embedding: learnable or sinusoidal
        if time_emb_type == 'learnable':
            self.timestep_embedding = nn.Linear(1, time_emb_dim)
        elif time_emb_type == 'sinusoidal':
            self.timestep_embedding = self.sine_cosine_transform_timesteps
        else:
            raise ValueError(f"Invalid time_embedding: {time_emb_type}")

        # Network architecture
        self.time_emb_dim = time_emb_dim
        self.time_emb_type = time_emb_type

        # Main network
        layers = []

        # case when timestep embedding are not used
        if time_emb_dim > 0:
            prev_dim = input_dim + time_emb_dim
        else:
            prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'lrelu':
                layers.append(nn.LeakyReLU())            
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, input_dim))  # Predict noise
        self.network = nn.Sequential(*layers)

    def forward(self, x, t):
        # Embed t
        if self.time_emb_type == 'learnable':
            t = t.unsqueeze(1)  # Ensure t has shape (batch_size, 1)
        t_embedded = self.timestep_embedding(t)  # Shape: (batch_size, embed_dim)
        
        # Concatenate x and embedded t
        if self.time_emb_dim > 0:
            x = torch.cat([x, t_embedded], dim=1)  # Shape: (batch_size, input_dim + embed_dim)

        # Pass through the network
        return self.network(x)  # Predict noise
    
    # define sinusodial time step embedding
    def sine_cosine_transform_timesteps(self, timesteps, max_period=10000):
        
        # dimension of output
        dim_out = self.time_emb_dim

        # half output dimension
        half_dim_out = dim_out // 2

        # determine tensor of frequencies
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half_dim_out, dtype=torch.float32) / half_dim_out)

        # push to compute device
        freqs = freqs.to(device=timesteps.device)

        # create timestep vs. frequency grid
        args = timesteps[:, None].float() * freqs[None]

        # creating the time embedding
        time_embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        # case: odd output dimension
        if dim_out % 2:
            # append additional dimension
            time_embedding = torch.cat([time_embedding, torch.zeros_like(time_embedding[:, :1])], dim=-1)

        # return timestep embedding
        return time_embedding    
    
# Forward Diffusion Scheduler
class DiffusionScheduler:
    """
    A class to handle the forward diffusion process in a diffusion model.
    This class initializes the noise schedule and provides methods to sample from the forward diffusion process.
    """
    def __init__(self, num_timesteps, device, beta_start=1e-4, beta_end=0.02, scheduler='linear'):
        self.num_timesteps = num_timesteps
        self.device = device
        self.beta_start = beta_start
        self.beta_end = beta_end

        # initialize beta and alpha values
        self.beta = self.init_scheduler(scheduler=scheduler).to(device)
        # self.beta = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def q_sample(self, x_0, t):
        """Forward diffusion process: Sample x_t given x_0."""
        noise = torch.randn_like(x_0)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1)
        return torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise, noise
    
    def init_scheduler(self, scheduler):
        """
        Initialize the scheduler.
        
        Parameters:
            scheduler (str): Name of the scheduler.
            
        Returns:
            callable: Scheduler function.
        """
        if scheduler == 'linear':
            return self.linear_noise_schedule(timesteps=self.num_timesteps, beta_start=self.beta_start, beta_end=self.beta_end)
        elif scheduler == 'quadratic':
            return self.quadratic_noise_schedule(timesteps=self.num_timesteps, beta_start=self.beta_start, beta_end=self.beta_end)
        elif scheduler == 'cosine':
            return self.cosine_noise_schedule(timesteps=self.num_timesteps)
        elif scheduler == 'sigmoid':
            return self.sigmoid_noise_schedule(timesteps=self.num_timesteps, beta_start=self.beta_start, beta_end=self.beta_end)
        elif scheduler == 'exponential':
            return self.exponential_noise_schedule(timesteps=self.num_timesteps, beta_start=self.beta_start, beta_end=self.beta_end)
        else:
            raise ValueError(f"Invalid scheduler: {scheduler}")
            
    def linear_noise_schedule(self, timesteps, beta_start=0.0001, beta_end=0.02):
        """
        Generates a linear noise schedule.
        
        Parameters:
            timesteps (int): Total number of timesteps.
            beta_start (float): Initial beta value.
            beta_end (float): Final beta value.
            
        Returns:
            np.ndarray: Array of beta values for each timestep.
        """
        return torch.linspace(beta_start, beta_end, timesteps)
    
    def quadratic_noise_schedule(self, timesteps, beta_start=0.0001, beta_end=0.02):
        """
        Generates a quadratic noise schedule.
        
        Parameters:
            timesteps (int): Total number of timesteps.
            beta_start (float): Initial beta value.
            beta_end (float): Final beta value.
            
        Returns:
            np.ndarray: Array of beta values for each timestep.
        """
        return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2
    
    def cosine_noise_schedule(self, timesteps, s=0.008):
        """
        Generates a cosine noise schedule.
        
        Parameters:
            timesteps (int): Total number of timesteps.
            s (float): Small offset to prevent division by zero.
            
        Returns:
            np.ndarray: Array of alpha_bar values for each timestep.
        """
        steps = torch.arange(timesteps + 1)
        f = lambda t: torch.cos((t / timesteps + s) / (1 + s) * torch.pi / 2) ** 2
        alphas_bar = f(steps) / f(torch.zeros(1))
        return alphas_bar[:-1]

    def sigmoid_noise_schedule(self, timesteps, beta_start=0.0001, beta_end=0.02):
        """
        Generates a sigmoid noise schedule.
        
        Parameters:
            timesteps (int): Total number of timesteps.
            beta_start (float): Initial beta value.
            beta_end (float): Final beta value.
            
        Returns:
            np.ndarray: Array of beta values for each timestep.
        """
        betas = torch.linspace(-6, 6, timesteps)
        # betas = 1 / (1 + np.exp(-betas))
        betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
        return betas
    
    def exponential_noise_schedule(self, timesteps, beta_start=0.0001, beta_end=0.02):
        """
        Generates an exponential noise schedule.
        
        Parameters:
            timesteps (int): Total number of timesteps.
            beta_start (float): Initial beta value.
            beta_end (float): Final beta value.
            
        Returns:
            np.ndarray: Array of beta values for each timestep.
        """
        return torch.logspace(torch.log10(torch.tensor(beta_start)),
                                torch.log10(torch.tensor(beta_end)),
                                timesteps)

class DDAE:
    """
    Easy wrapper for diffusion denoising autoencoder.
    Combines DiffusionDenoiser (network) and DiffusionScheduler (noise injection).
    Provides fit and predict methods.
    """
    def __init__(
            self, 
            # Denoiser parameters
            input_dim,            
            hidden_dim=[64, 64], 
            activation='lrelu', 
            # Diffusion parameters
            num_timesteps=100, 
            beta_start=1e-4,
            beta_end=0.02,
            scheduler='linear',
            # Time embedding parameters
            time_emb_dim=4, 
            time_emb_type='sinusoidal', 
            # training parameters
            epochs=100,
            batch_size=64,
            learning_rate=1e-3,
            device=None,
            eval_epochs=None,
        ):
        self.input_dim = input_dim
        self.T = num_timesteps
        self.device = device if device else torch.device("cpu")
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.eval_epochs = eval_epochs if eval_epochs is not None else epochs

        # Denoiser network
        self.denoiser = DiffusionDenoiser(
            input_dim=input_dim, 
            hidden_dims=hidden_dim, 
            time_emb_dim=time_emb_dim,
            time_emb_type=time_emb_type,
            activation=activation
        ).to(self.device)

        # Diffusion scheduler
        self.scheduler = DiffusionScheduler(
            num_timesteps=num_timesteps, 
            device=self.device,
            beta_start=beta_start,
            beta_end=beta_end,
            scheduler=scheduler
        )

    def fit(self, x_train, y_train=None, y_test=None):
        """
        Fit the diffusion denoising autoencoder to the training data.
        Args:
            x_train (torch.Tensor): Training data tensor of shape (num_samples, input_dim).
            y_train (torch.Tensor, optional): Labels for training data (not used in unsupervised setting).
        """
        x_train = torch.as_tensor(x_train, dtype=torch.float32)
        optimizer = Adam(self.denoiser.parameters(), lr=self.learning_rate)
        dataloader = DataLoader(x_train, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            # exp_run_statistics = {'epoch': epoch + 1, 'loss': 0.0}
            self.denoiser.train()
            total_loss = 0
            for x_0 in dataloader:
                x_0 = x_0.to(self.device)
                batch_size = x_0.size(0)
                t = torch.randint(1, self.T, (batch_size,), device=self.device).long()
                # Forward process: Sample x_t and noise
                x_t, noise = self.scheduler.q_sample(x_0, t)
                # Predict noise
                x_0_hat = self.denoiser(x_t, t.float())
                # Loss: MSE between true noise and predicted noise
                loss = nn.MSELoss()(x_0_hat, x_0)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Log statistics for the epoch
            loss = total_loss / len(dataloader)
            # exp_run_statistics['loss'] = loss
            # exp_run_statistics['epoch'] = epoch + 1

            # evaluate model
            # if (epoch + 1) % self.eval_epochs == 0:
            #     anomaly_scores = self.decision_function(x_test, y_test)
            #     metrics = evaluate_anomaly_detection(anomaly_scores, y_test)
            #     # metrics = {**exp_run_statistics, **metrics}
            #     print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss:.4f} | PR-AUC: {metrics['PR-AUC']:.4f}, ROC-AUC: {metrics['ROC-AUC']:.4f}")

    # Predict anomaly scores for the input data
    def decision_function(self, x_test, y_test=None):
        """
        Predict anomaly scores for the input data using the trained denoiser.
        Args:
            x_test (torch.Tensor): Test data tensor of shape (num_samples, input_dim).
            y_test (torch.Tensor, optional): Labels for test data (not used in unsupervised setting).

        Returns:
            torch.Tensor: Anomaly scores for the input data.
        """
        x_test = torch.as_tensor(x_test, dtype=torch.float32)
        self.denoiser.eval()
        batch_size = 8192 # 8192 is a good batch size for most GPUs
        num_samples = x_test.size(0)
        anomaly_scores = torch.zeros(num_samples)

        with torch.no_grad():
            for t in range(1, self.T):
                for i in range(0, num_samples, batch_size):
                    x_batch = x_test[i:i + batch_size].to(self.device)
                    t_tensor = torch.tensor(t, dtype=torch.int64).repeat(x_batch.size(0)).to(self.device)
                    # alpha_bar_t = self.scheduler.alpha_bar[t]
                    # x_t = torch.sqrt(alpha_bar_t) * x_batch + torch.sqrt(1 - alpha_bar_t) * torch.randn_like(x_batch)
                    x_t, _ = self.scheduler.q_sample(x_batch, t_tensor)
                    x_0_hat = self.denoiser(x_t, t_tensor)
                    scores = torch.norm(x_batch - x_0_hat, dim=1).cpu().numpy()
                    anomaly_scores[i:i + batch_size] += scores
        return anomaly_scores.numpy()
