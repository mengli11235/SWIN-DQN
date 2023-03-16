from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import copy
import numpy as np
from IPython import embed
from collections import Counter
import torch
torch.set_num_threads(2)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime
import time
from dqn_model import EnsembleNet, NetWithPrior
from swin_mlp import SwinMLP
from swin_model import SwinTransformer
from dqn_utils import seed_everything, write_info_file, generate_gif, save_checkpoint
from env import Environment
from replay import ReplayMemory
import config
import cv2


def norm_by(x, scale=255):
    return (x-x.min())/(x.max()-x.min())*scale

def plot_kernel():
    color = 'gray' #'viridis'
    state0 = env.reset()
    for i in range(100):
        state0, reward, life_lost, terminal = env.step(1)
    #cv2.imwrite('patch.jpg', state[-1])
    plt.imshow(state0[-1], cmap=color)
    plt.savefig('patch.jpg')
    plt.close()
    plt.imshow(state0[-1], cmap=color)
    #plt.colorbar()
    state = torch.Tensor(state0.astype(np.float)/255)[None,:].to('cpu')
    avgpool = nn.AdaptiveAvgPool1d(1)
    m = nn.Upsample(scale_factor=12, mode='nearest')
    m0 = nn.Upsample(scale_factor=6, mode='nearest')
    m1 = nn.Upsample(size=(84,84), mode='nearest')

    # patch_embedding = policy_net.patch_embed.proj(state)
    # x = policy_net.patch_embed.norm(patch_embedding.flatten(2).transpose(1, 2))
    # print(x.size())
    # x0 = avgpool(x.reshape(28, 28, 96)).detach().squeeze()
    # cv2.imwrite('patch0.jpg', norm_by(x0.detach().squeeze().numpy()).astype(np.uint8))
    x = policy_net.patch_embed(state)
    x = policy_net.pos_drop(x)

    x = policy_net.layers[0](x)
    x0 = avgpool(x.reshape(14, 14, 192)).detach().squeeze()
    plt.imshow(norm_by(m0(x0.unsqueeze(0).unsqueeze(0)).squeeze().numpy(), 100), cmap='viridis', alpha=0.6)
    #plt.colorbar()
    plt.savefig('layer1.pdf')
    plt.close()
    plt.imshow(state0[-1], cmap=color)

    x = policy_net.layers[1](x)
    x1 = avgpool(x.reshape(7, 7, 384)).detach().squeeze()
    plt.imshow(norm_by(m(x1.unsqueeze(0).unsqueeze(0)).squeeze().numpy(), 100), cmap='viridis', alpha=0.6)
    
    plt.savefig('layer2.pdf')
    plt.close()
    plt.imshow(state0[-1], cmap=color)
    x = policy_net.layers[2](x)
    y = F.relu(policy_net2.core_net.conv1(state))
    y0 = avgpool(y.reshape(32,20,20).transpose(0,1).transpose(1,2)).detach().squeeze()
    plt.imshow(norm_by(m1(y0.unsqueeze(0).unsqueeze(0)).squeeze().numpy(), 100), cmap='viridis', alpha=0.6)
    
    plt.savefig('conv1.pdf')
    plt.close()
    plt.imshow(state0[-1], cmap=color)
    y = F.relu(policy_net2.core_net.conv2(y))
    y1 = avgpool(y.reshape(64,9,9).transpose(0,1).transpose(1,2)).detach().squeeze()
    plt.imshow(norm_by(m1(y1.unsqueeze(0).unsqueeze(0)).squeeze().numpy(), 100), cmap='viridis', alpha=0.6)
    
    plt.savefig('conv2.pdf')
    plt.close()
    plt.imshow(state0[-1], cmap=color)

    y = F.relu(policy_net2.core_net.conv3(y))

    y = avgpool(y.reshape(64,7,7).transpose(0,1).transpose(1,2)).detach().squeeze()
    #x = policy_net.norm(x)
    x = avgpool(x.reshape(7, 7, 384)).detach().squeeze()
    #print(norm_by(m(y.unsqueeze(0).unsqueeze(0)).squeeze().numpy(), 100).max(),)
    plt.imshow(norm_by(m(x.unsqueeze(0).unsqueeze(0)).squeeze().numpy(), 100), cmap='viridis', alpha=0.6)
    
    plt.savefig('layer3.pdf')
    plt.close()
    plt.imshow(state0[-1], cmap=color)
    plt.imshow(norm_by(m(y.unsqueeze(0).unsqueeze(0)).squeeze().numpy(), 100), cmap='viridis', alpha=0.6)
    plt.savefig('conv3.pdf')

    #x = avgpool(x.reshape(7, 7, 384)).detach().squeeze()
    #cv2.imwrite('patch1.jpg', norm_by(x.detach().squeeze().numpy()).astype(np.uint8))

if __name__ == '__main__':
    env = Environment(rom_file='roms/time_pilot.bin', frame_skip=4,
                      num_frames=4, no_op_start=30, rand_seed=101,
                      dead_as_end=True, max_episode_steps=27000)
    policy_net = SwinMLP(img_size=84, patch_size=3, in_chans=4, num_classes=env.num_actions, depths=[2, 3, 2], num_heads=[3, 3, 6], window_size=7).to('cpu')

    policy_net2 = EnsembleNet(n_ensemble=1, n_actions=env.num_actions, network_output_size=84, num_channels=4, dueling=False)

    model_dict = torch.load('../swin_results/model_savedir/time_pilot00/time_pilot_bestq.pkl', map_location=torch.device('cpu'))
    model_dict2 = torch.load('../swin_results/model_savedir/time_pilot01/time_pilot_bestq.pkl', map_location=torch.device('cpu'))

    policy_net.load_state_dict(model_dict['policy_net_state_dict'])
    policy_net2.load_state_dict(model_dict2['policy_net_state_dict'])
    plot_kernel()