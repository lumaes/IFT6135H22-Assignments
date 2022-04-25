import math
import numpy as np
import torch


def log_likelihood_bernoulli(mu, target):
    """ 
    :param mu: (FloatTensor) - shape: (batch_size x input_size) - The mean of Bernoulli random variables p(x=1).
    :param target: (FloatTensor) - shape: (batch_size x input_size) - Target samples (binary values).
    :return: (FloatTensor) - shape: (batch_size,) - log-likelihood of target samples on the Bernoulli random variables.
    """
    # init
    batch_size = mu.size(0)
    mu = mu.view(batch_size, -1)
    target = target.view(batch_size, -1)

    log_prob_target = (mu.log() * target) + ((1-mu).log() * (1-target))
    log_likelihood_bernoulli = log_prob_target.sum(dim=1)

    return log_likelihood_bernoulli


def log_likelihood_normal(mu, logvar, z):
    """ 
    :param mu: (FloatTensor) - shape: (batch_size x input_size) - The mean of Normal distributions.
    :param logvar: (FloatTensor) - shape: (batch_size x input_size) - The log variance of Normal distributions.
    :param z: (FloatTensor) - shape: (batch_size x input_size) - Target samples.
    :return: (FloatTensor) - shape: (batch_size,) - log probability of the sames on the given Normal distributions.
    """
    # init
    batch_size = mu.size(0)
    mu = mu.view(batch_size, -1)
    logvar = logvar.view(batch_size, -1)
    z = z.view(batch_size, -1)
    log_normal = torch.empty(batch_size)

    for i in range(batch_size):
      distrib = torch.distributions.multivariate_normal.MultivariateNormal(mu[i], torch.diag(logvar[i].exp()))
      log_normal[i] = distrib.log_prob(z[i])

    # log normal
    return log_normal

def log_mean_exp(y):
    """
    :param y: (FloatTensor) - shape: (batch_size x sample_size) - Values to be evaluated for log_mean_exp. For example log proababilies
    :return: (FloatTensor) - shape: (batch_size,) - Output for log_mean_exp.
    """
    # init
    batch_size = y.size(0)
    sample_size = y.size(1)

    a_i = torch.max(y, dim=1).values.unsqueeze(-1)
    mean_exp = ((y-a_i).exp().sum(dim=1) * 1/sample_size).squeeze()
    log_mean_exp = torch.log(mean_exp) + a_i.squeeze()
    return log_mean_exp

def kl_gaussian_gaussian_analytic(mu_q, logvar_q, mu_p, logvar_p):
    """ 
    :param mu_q: (FloatTensor) - shape: (batch_size x input_size) - The mean of first distributions (Normal distributions).
    :param logvar_q: (FloatTensor) - shape: (batch_size x input_size) - The log variance of first distributions (Normal distributions).
    :param mu_p: (FloatTensor) - shape: (batch_size x input_size) - The mean of second distributions (Normal distributions).
    :param logvar_p: (FloatTensor) - shape: (batch_size x input_size) - The log variance of second distributions (Normal distributions).
    :return: (FloatTensor) - shape: (batch_size,) - kl-divergence of KL(q||p).
    """
    # init
    batch_size = mu_q.size(0)
    mu_q = mu_q.view(batch_size, -1)
    logvar_q = logvar_q.view(batch_size, -1)
    mu_p = mu_p.view(batch_size, -1)
    logvar_p = logvar_p.view(batch_size, -1)
    logstd_p = logvar_p.exp().sqrt().log()
    logstd_q = logvar_q.exp().sqrt().log()
    # KLD analytc
    # BASED ON: https://tiao.io/post/density-ratio-estimation-for-kl-divergence-minimization-between-implicit-distributions/
    out = logstd_p - logstd_q - 0.5*(1-(logvar_q.exp() + (mu_q - mu_p).pow(2))/(logvar_p.exp()))
    return out.sum(dim=-1)

def kl_gaussian_gaussian_mc(mu_q, logvar_q, mu_p, logvar_p, num_samples=1):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu_q: (FloatTensor) - shape: (batch_size x input_size) - The mean of first distributions (Normal distributions).
    :param logvar_q: (FloatTensor) - shape: (batch_size x input_size) - The log variance of first distributions (Normal distributions).
    :param mu_p: (FloatTensor) - shape: (batch_size x input_size) - The mean of second distributions (Normal distributions).
    :param logvar_p: (FloatTensor) - shape: (batch_size x input_size) - The log variance of second distributions (Normal distributions).
    :param num_samples: (int) - shape: () - The number of sample for Monte Carlo estimate for KL-divergence
    :return: (FloatTensor) - shape: (batch_size,) - kl-divergence of KL(q||p).
    """
    # init
    batch_size = mu_q.size(0)
    input_size = np.prod(mu_q.size()[1:])
    mu_q = mu_q.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    logvar_q = logvar_q.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    mu_p = mu_p.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    logvar_p = logvar_p.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    
    # generate samples from q distrib
    samples = torch.distributions.Normal(mu_q, ((1/2) * logvar_q).exp()).rsample()

    # kld
    p_gauss = (samples - mu_p) ** 2 / (2 * logvar_p.exp())
    q_gauss = (samples - mu_q) ** 2 / (2 * logvar_q.exp())
    diff = (1/2) * (logvar_p - logvar_q)

    kld = (p_gauss - q_gauss + diff).sum(2)

    return kld.mean(1)

