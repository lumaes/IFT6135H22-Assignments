import torch
from q2_sampler import svhn_sampler
from q2_model import Critic, Generator
from torch import optim
from torchvision.utils import save_image



def lp_reg(x, y, critic):
    """
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** The notation used for the parameters follow the one from Petzka et al: https://arxiv.org/pdf/1709.08894.pdf
    In other word, x are samples from the distribution mu and y are samples from the distribution nu. The critic is the
    equivalent of f in the paper. Also consider that the norm used is the L2 norm. This is important to consider,
    because we make the assumption that your implementation follows this notation when testing your function. ***

    :param x: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution P.
    :param y: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution Q.
    :param critic: (Module) - torch module that you want to regularize.
    :return: (FloatTensor) - shape: (1,) - Lipschitz penalty
    """
    t = torch.zeros(x.shape).uniform_(0,1).to(x.device).requires_grad_()

    x_hat = t*x + (1-t)*y
    x_hat.requires_grad_()

    cx_hat = critic(x_hat)

    grad_x_hat = torch.autograd.grad(outputs=cx_hat, inputs=x_hat,
							   grad_outputs=torch.ones(cx_hat.shape).to(x.device),
							   create_graph=True, retain_graph=True)[0]
    grad_x_hat.requires_grad_()

    grad_x_hat = grad_x_hat.reshape((x.shape[0], -1))

    reg = (grad_x_hat.norm(2, dim=-1)-1)

    ZEROS = torch.zeros(reg.shape).to(x.device)
    reg = torch.max(ZEROS, reg)
    reg = reg.pow(2).mean()
    
    return reg


def vf_wasserstein_distance(p, q, critic):
    """
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** The notation used for the parameters follow the one from Petzka et al: https://arxiv.org/pdf/1709.08894.pdf
    In other word, x are samples from the distribution mu and y are samples from the distribution nu. The critic is the
    equivalent of f in the paper. This is important to consider, because we make the assuption that your implementation
    follows this notation when testing your function. ***

    :param p: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution p.
    :param q: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution q.
    :param critic: (Module) - torch module used to compute the Wasserstein distance
    :return: (FloatTensor) - shape: (1,) - Estimate of the Wasserstein distance
    """
    return critic(p).mean() - critic(q).mean()

if __name__ == '__main__':
    # Example of usage of the code provided and recommended hyper parameters for training GANs.
    data_root = './'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_iter = 50000 # N training iterations
    n_critic_updates = 5 # N critic updates per generator update
    lp_coeff = 10 # Lipschitz penalty coefficient
    train_batch_size = 64
    test_batch_size = 64
    lr = 1e-4
    beta1 = 0.5
    beta2 = 0.9
    z_dim = 100

    train_loader, valid_loader, test_loader = svhn_sampler(data_root, train_batch_size, test_batch_size)

    import copy
    import os
    import os.path

    if not os.path.exists("gan_gen.pt"):

      generator = Generator(z_dim=z_dim).to(device)
      critic = Critic().to(device)

      optim_critic = optim.Adam(critic.parameters(), lr=lr, betas=(beta1, beta2))
      optim_generator = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))

      # COMPLETE TRAINING PROCEDURE
      train_iter = iter(train_loader)
      valid_iter = iter(valid_loader)
      test_iter = iter(test_loader)
      for i in range(n_iter):
          generator.train()
          critic.train()
          for _ in range(n_critic_updates):
              try:
                  data = next(train_iter)[0].to(device)
              except Exception:
                  train_iter = iter(train_loader)
                  data = next(train_iter)[0].to(device)
              #####
              # train the critic model here
              #####
              z = torch.randn(64, z_dim, device=device)
              y = generator(z)
              optim_critic.zero_grad()
              loss = vf_wasserstein_distance(y,data, critic) + lp_coeff * lp_reg(data, y, critic)      
              loss.backward()
              optim_critic.step()
          #####
          # train the generator model here
          #####
          z = torch.randn(64, z_dim, device=device)
          fake_img = generator(z)
          optim_generator.zero_grad()
          loss = -critic(fake_img).mean()
          loss.backward()
          optim_generator.step()

          # Save sample images 
          if i % 100 == 0:
              z = torch.randn(64, z_dim, device=device)
              imgs = generator(z)
              save_image(imgs, f'gan_images/imgs_{i}.png', normalize=True, value_range=(-1, 1))


      # COMPLETE QUALITATIVE EVALUATION


      torch.save(generator, 'gan_gen.pt')
      torch.save(critic, 'gan_critic.pt')
    else:
      generator = torch.load('gan_gen.pt')
      critic = torch.load('gan_critic.pt')

    generator.eval()
    critic.eval()
    epsilon_perturbation = torch.arange(0, 10, 0.5)
    with torch.no_grad():

      # DISANTEGLED REPRESENTATION
      z = torch.randn(1, z_dim, device=device).squeeze()
      # For each dim
      for i in range(z_dim):
        # make perturbation
        saved_z = torch.empty((len(epsilon_perturbation), z_dim),  device=device)
        for idx, eps in enumerate(epsilon_perturbation):
          z_copy = copy.deepcopy(z).to(device)
          z_copy[i] += eps.item()
          saved_z[idx] = z_copy
        # generate images
        imgs = generator(saved_z)
        train_iter = iter(train_loader)
        dat =  next(train_iter)[0]
        save_image(dat, f'gan_images/train_img.png', normalize=True, value_range=(-1, 1))
        save_image(imgs, f'gan_images/disent_rep_dim{i}.png', normalize=True, value_range=(-1, 1), nrow=len(imgs))
      # LINEAR INTERPOLATION

      # in latent space
      for i in range(200):
        z_0 = torch.randn(1, z_dim, device=device).squeeze()
        z_1 = torch.randn(1, z_dim, device=device).squeeze()
        lambdas = torch.arange(0, 1.1, 0.1)
        saved_z = torch.empty((len(lambdas), z_dim), device=device)
        for idx, lbda in enumerate(lambdas):
          lbda = lbda.item()
          z = lbda * z_0 + (1-lbda) * z_1
          saved_z[idx] = z
        # gen img
        imgs = generator(saved_z)
        save_image(imgs, f'gan_images/disent_latent_interp{i}.png', normalize=True, value_range=(-1, 1), nrow=len(imgs))

        #out latent space
        lambdas = torch.arange(0, 1.1, 0.1, device=device)
        z_0 = z_0.unsqueeze(0)
        z_1 = z_1.unsqueeze(0)    
        z = torch.cat((z_0, z_1), dim=0)
        imgs = generator(z)

        img_0 = imgs[0]
        img_1 = imgs[1]

        saved_imgs = torch.empty((len(lambdas), img_0.shape[0], img_0.shape[1], img_0.shape[2]),  device=device)

        for idx, lbda in enumerate(lambdas):
          lbda = lbda.item()
          inter_img = lbda * img_0 + (1-lbda) * img_1
          saved_imgs[idx] = inter_img

        save_image(saved_imgs, f'gan_images/disent_dataspace_interp{i}.png', normalize=True, value_range=(-1, 1), nrow=len(saved_imgs))

