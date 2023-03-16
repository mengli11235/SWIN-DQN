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

def rolling_average(a, n=5) :
    if n == 0:
        return a
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_dict_losses(plot_dict, name='loss_example.png', rolling_length=4, plot_title=''):
    f,ax=plt.subplots(1,1,figsize=(6,6))
    for n in plot_dict.keys():
        print('plotting', n)
        ax.plot(rolling_average(plot_dict[n]['index']), rolling_average(plot_dict[n]['val']), lw=1)
        ax.scatter(rolling_average(plot_dict[n]['index']), rolling_average(plot_dict[n]['val']), label=n, s=3)
    ax.legend()
    if plot_title != '':
        plt.title(plot_title)
    plt.savefig(name)
    plt.close()

def matplotlib_plot_all(p):
    epoch_num = len(p['steps'])
    epochs = np.arange(epoch_num)
    steps = p['steps']
    plot_dict_losses({'episode steps':{'index':epochs,'val':p['episode_step']}}, name=os.path.join(model_base_filedir, 'episode_step.png'), rolling_length=0)
    plot_dict_losses({'episode steps':{'index':epochs,'val':p['episode_relative_times']}}, name=os.path.join(model_base_filedir, 'episode_relative_times.png'), rolling_length=10)
    plot_dict_losses({'episode head':{'index':epochs, 'val':p['episode_head']}}, name=os.path.join(model_base_filedir, 'episode_head.png'), rolling_length=0)

    episode_loss_mask = np.isfinite(p['episode_loss'])
    plot_dict_losses({'steps loss':{'index':np.array(steps)[episode_loss_mask], 'val':np.array(p['episode_loss'])[episode_loss_mask]}}, name=os.path.join(model_base_filedir, 'steps_loss.png'))

    q_mask = np.isfinite(p['q_record'])
    plot_dict_losses({'max q values':{'index':np.array(steps)[q_mask], 'val':np.array(p['q_record'])[q_mask]}}, name=os.path.join(model_base_filedir, 'q_record.png'))

    plot_dict_losses({'steps eps':{'index':steps, 'val':p['eps_list']}}, name=os.path.join(model_base_filedir, 'steps_mean_eps.png'), rolling_length=0)
    plot_dict_losses({'steps reward':{'index':steps,'val':p['episode_reward']}},  name=os.path.join(model_base_filedir, 'steps_reward.png'), rolling_length=0)
    plot_dict_losses({'episode reward':{'index':epochs, 'val':p['episode_reward']}}, name=os.path.join(model_base_filedir, 'episode_reward.png'), rolling_length=0)
    plot_dict_losses({'episode times':{'index':epochs,'val':p['episode_times']}}, name=os.path.join(model_base_filedir, 'episode_times.png'), rolling_length=5)
    plot_dict_losses({'steps avg reward':{'index':steps,'val':p['avg_rewards']}}, name=os.path.join(model_base_filedir, 'steps_avg_reward.png'), rolling_length=0)

    eval_steps_mask = np.isfinite(p['eval_steps'])
    eval_rewards_mask = np.isfinite(p['eval_rewards'])
    eval_score_mask = np.isfinite(p['highest_eval_score'])

    plot_dict_losses({'eval rewards':{'index':np.array(p['eval_steps'])[eval_steps_mask], 'val':np.array(p['eval_rewards'])[eval_rewards_mask]}}, name=os.path.join(model_base_filedir, 'eval_rewards_steps.png'), rolling_length=0)
    plot_dict_losses({'highest eval score':{'index':np.array(p['eval_steps'])[eval_steps_mask], 'val':np.array(p['highest_eval_score'])[eval_score_mask]}}, name=os.path.join(model_base_filedir, 'highest_eval_score.png'), rolling_length=0)

def handle_checkpoint(last_save, cnt):
    if (cnt-last_save) >= info['CHECKPOINT_EVERY_STEPS']:
        st = time.time()
        print("beginning checkpoint", st)
        last_save = cnt
        state = {'info':info,
                 'optimizer':opt.state_dict(),
                 'cnt':cnt,
                 'policy_net_state_dict':policy_net.state_dict(),
                 'target_net_state_dict':target_net.state_dict(),
                 'perf':perf,
                }
        filename = os.path.abspath(model_base_filepath +"_bestq.pkl")
        save_checkpoint(state, filename)
        # npz will be added
        #buff_filename = os.path.abspath(model_base_filepath + "_%010dq_train_buffer"%cnt)
        #replay_memory.save_buffer(buff_filename)
        print("finished checkpoint", time.time()-st)
        return last_save
    else: return last_save


class ActionGetter:
    """Determines an action according to an epsilon greedy strategy with annealing epsilon"""
    """This class is from fg91's dqn."""
    def __init__(self, n_actions, eps_initial=1, eps_final=0.1, eps_final_frame=0.01,
                 eps_evaluation=0.0, eps_annealing_frames=100000,
                 replay_memory_start_size=50000, max_steps=25000000, random_seed=122):
        """
        Args:
            n_actions: Integer, number of possible actions
            eps_initial: Float, Exploration probability for the first
                replay_memory_start_size frames
            eps_final: Float, Exploration probability after
                replay_memory_start_size + eps_annealing_frames frames
            eps_final_frame: Float, Exploration probability after max_frames frames
            eps_evaluation: Float, Exploration probability during evaluation
            eps_annealing_frames: Int, Number of frames over which the
                exploration probabilty is annealed from eps_initial to eps_final
            replay_memory_start_size: Integer, Number of frames during
                which the agent only explores
            max_frames: Integer, Total number of frames shown to the agent
        """
        self.n_actions = n_actions
        self.eps_initial = eps_initial
        self.eps_final = eps_final
        self.eps_final_frame = eps_final_frame
        self.eps_evaluation = eps_evaluation
        self.eps_annealing_frames = eps_annealing_frames
        self.replay_memory_start_size = replay_memory_start_size
        self.max_steps = max_steps
        self.random_state = np.random.RandomState(random_seed)

        # Slopes and intercepts for exploration decrease
        if self.eps_annealing_frames > 0:
            self.slope = -(self.eps_initial - self.eps_final)/self.eps_annealing_frames
            self.intercept = self.eps_initial - self.slope*self.replay_memory_start_size
            self.slope_2 = -(self.eps_final - self.eps_final_frame)/(self.max_steps - self.eps_annealing_frames - self.replay_memory_start_size)
            self.intercept_2 = self.eps_final_frame - self.slope_2*self.max_steps

    def pt_get_action(self, step_number, state, active_head=None, evaluation=False):
        """
        Args:
            step_number: int number of the current step
            state: A (4, 84, 84) sequence of frames of an atari game in grayscale
            active_head: number of head to use, if None, will run all heads and vote
            evaluation: A boolean saying whether the agent is being evaluated
        Returns:
            An integer between 0 and n_actions
        """

        if evaluation:
            eps = self.eps_evaluation
        elif step_number < self.replay_memory_start_size:
            eps = self.eps_initial
        elif self.eps_annealing_frames > 0:
            if step_number >= self.replay_memory_start_size and step_number < self.replay_memory_start_size + self.eps_annealing_frames:
                eps = self.slope*step_number + self.intercept
            elif step_number >= self.replay_memory_start_size + self.eps_annealing_frames:
                eps = self.slope_2*step_number + self.intercept_2
        else:
            eps = 0
        if self.random_state.rand() < eps:
            return eps, self.random_state.randint(0, self.n_actions)
        else:
            state = torch.Tensor(state.astype(np.float)/info['NORM_BY'])[None,:].to(info['DEVICE'])
            vals = policy_net(state, active_head)
            if active_head is not None:
                action = torch.argmax(vals, dim=1).item()
                return eps, action
            else:
                # vote
                acts = [torch.argmax(vals[h],dim=1).item() for h in range(info['N_ENSEMBLE'])]
                data = Counter(acts)
                action = data.most_common(1)[0][0]
                heads_chosen = [0]*info['N_ENSEMBLE']
                for i,head in enumerate(acts):
                    if action == head:
                        heads_chosen[i] += 1
                return heads_chosen, action

def ptlearn(states, actions, rewards, next_states, terminal_flags, active_heads, masks):
    states = torch.Tensor(states.astype(np.float)/info['NORM_BY']).to(info['DEVICE'])
    next_states = torch.Tensor(next_states.astype(np.float)/info['NORM_BY']).to(info['DEVICE'])
    rewards = torch.Tensor(rewards).to(info['DEVICE'])
    actions = torch.LongTensor(actions).to(info['DEVICE'])
    terminal_flags = torch.Tensor(terminal_flags.astype(np.int)).to(info['DEVICE'])
    active_heads = torch.LongTensor(active_heads).to(info['DEVICE'])
    masks = torch.FloatTensor(masks.astype(np.int)).to(info['DEVICE'])
    # min history to learn is 200,000 frames in dqn - 50000 steps
    losses = [0.0 for _ in range(info['N_ENSEMBLE'])]

    opt.zero_grad()
    q_policy_vals = policy_net(states, None)
    next_q_target_vals = target_net(next_states, None)
    next_q_policy_vals = policy_net(next_states, None)

    cnt_losses = []
    q_record = torch.stack(next_q_target_vals).detach().max()

    if 'PRIOR' in info['IMPROVEMENT']:
        info['PRIOR_SCALE'] = 1+torch.stack(next_q_target_vals).detach().max()/20
        prior_next_pi = torch.empty(info['N_ENSEMBLE'], info['BATCH_SIZE'], q_policy_vals[0].size(-1)).to(info['DEVICE'])
        # sample priors
        nn.init.normal_(prior_next_pi, 0, 0.02)

    for k in range(info['N_ENSEMBLE']):
        # finish masking
        total_used = torch.sum(masks[:,k])
        if total_used > 0.0:
            next_q_vals = next_q_target_vals[k].data
            next_policy_vals = next_q_policy_vals[k].data
            if 'PRIOR' in info['IMPROVEMENT']:
                # add priors to the target value
                next_q_vals += info['PRIOR_SCALE']*prior_next_pi[k].detach()
                next_policy_vals += info['PRIOR_SCALE']*prior_next_pi[k].detach()


            if info['DOUBLE_DQN']:
                next_actions = next_policy_vals.max(1, True)[1]
                next_qs = next_q_vals.gather(1, next_actions).squeeze(1)
            else:
                next_qs = next_q_vals.max(1)[0] # max returns a pair

            preds = q_policy_vals[k].gather(1, actions[:,None]).squeeze(1) 

            targets = info['GAMMA'] * next_qs * (1-terminal_flags) + rewards
            l1loss = F.smooth_l1_loss(preds, targets, reduction='mean')

            full_loss = masks[:,k]*l1loss #batch*1
            loss = torch.sum(full_loss/total_used)
            cnt_losses.append(loss)
            losses[k] = loss.cpu().detach().item()

    loss = sum(cnt_losses)/info['N_ENSEMBLE']
    loss.backward()
    #for param in policy_net.core_net.parameters():
    for param in policy_net.parameters():
        if param.grad is not None:
            # divide grads in core
            param.grad.data *=1.0/float(info['N_ENSEMBLE'])
    nn.utils.clip_grad_norm_(policy_net.parameters(), info['CLIP_GRAD'])

    opt.step()

    return q_record.cpu(), np.mean(losses)

def train(step_number, last_save):
    """Contains the training and evaluation loops"""
    epoch_num = len(perf['steps'])
    highest_eval_score = -np.inf
    waves = 0
    epoch_frame_episode_last = 0

    while step_number < info['MAX_STEPS']:
        ########################
        ####### Training #######
        ########################
        epoch_frame = 0
        while epoch_frame < info['EVAL_FREQUENCY']:
            terminal = False
            life_lost = True
            state = env.reset()
            start_steps = step_number
            st = time.time()
            episode_reward_sum = 0
            epoch_frame_episode = 0
            random_state.shuffle(heads)
            active_head = heads[0]
            epoch_num += 1
            ep_eps_list = []
            ptloss_list = []
            q_list = []
            action_list = []
            while not terminal:
                if life_lost:
                    action = 1
                    eps = 0

                eps,action = action_getter.pt_get_action(step_number, state=state, active_head=active_head)



                ep_eps_list.append(eps)
                next_state, reward, life_lost, terminal = env.step(action)
                # Store transition in the replay memory

                replay_memory.add_experience(action=action,
                                                frame=next_state[-1],
                                                reward=np.sign(reward),
                                                terminal=life_lost,
                                                active_head= active_head)

                step_number += 1
                epoch_frame += 1
                epoch_frame_episode += 1
                episode_reward_sum += reward
                state = next_state

                if step_number % info['LEARN_EVERY_STEPS'] == 0 and step_number > info['MIN_HISTORY_TO_LEARN']:
                    _states, _actions, _rewards, _next_states, _terminal_flags, _active_heads, _masks = replay_memory.get_minibatch(info['BATCH_SIZE'])
                    q_record, ptloss = ptlearn(_states, _actions, _rewards, _next_states, _terminal_flags, _active_heads, _masks)
                    ptloss_list.append(ptloss)
                    q_list.append(q_record)

                if step_number % info['TARGET_UPDATE'] == 0 and step_number >  info['MIN_HISTORY_TO_LEARN']:
                    print("++++++++++++++++++++++++++++++++++++++++++++++++")
                    print('updating target network at %s'%step_number)
                    target_net.load_state_dict(policy_net.state_dict())

            et = time.time()
            ep_time = et-st
            epoch_frame_episode_last = epoch_frame_episode

            perf['steps'].append(step_number)
            perf['episode_step'].append(step_number-start_steps)
            perf['episode_head'].append(active_head)
            perf['eps_list'].append(np.mean(ep_eps_list))
            perf['episode_loss'].append(np.mean(ptloss_list))
            perf['q_record'].append(np.mean(q_list))
            perf['episode_reward'].append(episode_reward_sum)
            perf['episode_times'].append(ep_time)
            perf['episode_relative_times'].append(time.time()-info['START_TIME'])
            perf['avg_rewards'].append(np.mean(perf['episode_reward'][-100:]))
            last_save = handle_checkpoint(last_save, step_number)

            if not epoch_num%info['PLOT_EVERY_EPISODES'] and step_number > info['MIN_HISTORY_TO_LEARN']:
                print('avg reward', perf['avg_rewards'][-1])
                print('last rewards', perf['episode_reward'][-info['PLOT_EVERY_EPISODES']:])

                matplotlib_plot_all(perf)
        avg_eval_reward, avg_eval_stds, highest_eval_score = evaluate(step_number, highest_eval_score)
        perf['eval_rewards'].append(avg_eval_reward)
        perf['highest_eval_score'].append(highest_eval_score)

        perf['eval_stds'].append(avg_eval_stds)
        perf['eval_steps'].append(step_number)
        matplotlib_plot_all(perf)

def evaluate(step_number, highest_eval_score):
    print("""
         #########################
         ####### Evaluation ######
         #########################
         """)
    eval_rewards = []
    evaluate_step_number = 0
    frames_for_gif = []
    heads_chosen = [0]*info['N_ENSEMBLE']

    # use different seed for each eval
    for i in range(info['NUM_EVAL_EPISODES']):
        eval_env = Environment(rom_file=info['GAME'], frame_skip=info['FRAME_SKIP'],
                    num_frames=info['HISTORY_SIZE'], no_op_start=info['MAX_NO_OP_FRAMES'], rand_seed=np.random.randint(255),
                    dead_as_end=info['DEAD_AS_END'], max_episode_steps=info['MAX_EPISODE_STEPS'])
        state = eval_env.reset()
        episode_reward_sum = 0
        terminal = False
        life_lost = True
        episode_steps = 0
        while not terminal:
            if life_lost:
                action = 1
            else:
                active_head=None
                eps,action = action_getter.pt_get_action(step_number, state, active_head=active_head, evaluation=True)
                heads_chosen = [x+y for x,y in zip(heads_chosen, eps)]
            next_state, reward, life_lost, terminal = eval_env.step(action)
            evaluate_step_number += 1
            episode_steps +=1
            episode_reward_sum += reward
            # only save the episode with highest scores
            frames_for_gif.append(eval_env.ale.getScreenRGB())
            if not episode_steps%100:
                print('eval', episode_steps, episode_reward_sum)
            state = next_state
        eval_rewards.append(episode_reward_sum)
        if episode_reward_sum > highest_eval_score:
            highest_eval_score = episode_reward_sum
            generate_gif(model_base_filedir, 0, frames_for_gif, 0, name='test')
        frames_for_gif = []
        eval_env.close()
    print("Evaluation score:\n", np.mean(eval_rewards))

    efile = os.path.join(model_base_filedir, 'eval_rewards.txt')
    with open(efile, 'a') as eval_reward_file:
        print(step_number, np.mean(eval_rewards), highest_eval_score, file=eval_reward_file)
    return np.mean(eval_rewards), np.std(eval_rewards), highest_eval_score

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-l', '--model_loadpath', default='', help='.pkl model file full path')
    parser.add_argument('-b', '--buffer_loadpath', default='', help='.npz replay buffer file full path')
    args = parser.parse_args()
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print("running on %s"%device)

    info = {
        "GAME":'roms/bowling.bin', # gym prefix
        "DEVICE":device, #cpu vs gpu set by argument
        "NAME":'FRANKbootstrap_fasteranneal_pong', # start files with name
        "DUELING":False, # use dueling dqn
        "DOUBLE_DQN":True, # use double dqn
        "PRIOR":True, # turn on to use randomized prior
        "PRIOR_SCALE":1, # what to scale prior by
        "N_ENSEMBLE":1, # number of bootstrap heads to use. when 1, this is a normal dqn
        "LEARN_EVERY_STEPS":4, # updates every 4 steps in osband
        "BERNOULLI_PROBABILITY": 1.0, # Probability of experience to go to each head - if 1, every experience goes to every head
        "TARGET_UPDATE":10000, # how often to update target network
        "MIN_HISTORY_TO_LEARN":50000, # in environment frames
        "NORM_BY":255.,  # divide the float(of uint) by this number to normalize - max val of data is 255
        "EPS_INITIAL":1.0, # should be 1
        "EPS_FINAL":0.01, # 0.01 in osband
        "EPS_EVAL":0.0, # 0 in osband, .05 in others....
        "EPS_ANNEALING_FRAMES":int(1e6), # this may have been 1e6 in osband
        #"EPS_ANNEALING_FRAMES":0, # if it annealing is zero, then it will only use the bootstrap after the first MIN_EXAMPLES_TO_LEARN steps which are random
        "EPS_FINAL_FRAME":0.01,
        "NUM_EVAL_EPISODES":5, # num examples to average in eval
        "BUFFER_SIZE":int(1e6), # Buffer size for experience replay
        "CHECKPOINT_EVERY_STEPS":10000000, # how often to write pkl of model and npz of data buffer
        "EVAL_FREQUENCY":250000, # how often to run evaluation episodes
        "ADAM_LEARNING_RATE":6.25e-5,
        "RMS_LEARNING_RATE": 0.00025, # according to paper = 0.00025
        "RMS_DECAY":0.95,
        "RMS_MOMENTUM":0.0,
        "RMS_EPSILON":0.00001,
        "RMS_CENTERED":True,
        "HISTORY_SIZE":4, # how many past frames to use for state input
        "N_EPOCHS":90000,  # Number of episodes to run
        "BATCH_SIZE":32, # Batch size to use for learning
        "GAMMA":.99, # Gamma weight in Q update
        "PLOT_EVERY_EPISODES": 50,
        "CLIP_GRAD":5, # Gradient clipping setting
        "SEED":101,
        "RANDOM_HEAD":-1, # just used in plotting as demarcation
        "NETWORK_INPUT_SIZE":(84,84),
        "START_TIME":time.time(),
        "MAX_STEPS":int(50e6), # 50e6 steps is 200e6 frames
        "MAX_EPISODE_STEPS":27000, # Orig dqn give 18k steps, Rainbow seems to give 27k steps
        "FRAME_SKIP":4, # deterministic frame skips to match deepmind
        "MAX_NO_OP_FRAMES":30, # random number of noops applied to beginning of each episode
        "DEAD_AS_END":True, # do you send finished=true to agent while training when it loses a life
        "IMPROVEMENT": ['swin', ''],
    }

    info['FAKE_ACTS'] = [info['RANDOM_HEAD'] for x in range(info['N_ENSEMBLE'])]
    info['args'] = args
    info['load_time'] = datetime.date.today().ctime()
    info['NORM_BY'] = float(info['NORM_BY'])
    info['NAME'] = info['GAME'][5:-4]


    # create environment
    env = Environment(rom_file=info['GAME'], frame_skip=info['FRAME_SKIP'],
                      num_frames=info['HISTORY_SIZE'], no_op_start=info['MAX_NO_OP_FRAMES'], rand_seed=info['SEED'],
                      dead_as_end=info['DEAD_AS_END'], max_episode_steps=info['MAX_EPISODE_STEPS'])

    # create replay buffer
    replay_memory = ReplayMemory(size=info['BUFFER_SIZE'],
                                 frame_height=info['NETWORK_INPUT_SIZE'][0],
                                 frame_width=info['NETWORK_INPUT_SIZE'][1],
                                 agent_history_length=info['HISTORY_SIZE'],
                                 batch_size=info['BATCH_SIZE'],
                                 num_heads=info['N_ENSEMBLE'],
                                 bernoulli_probability=info['BERNOULLI_PROBABILITY'])

    random_state = np.random.RandomState(info["SEED"])
    action_getter = ActionGetter(n_actions=env.num_actions,
                                 eps_initial=info['EPS_INITIAL'],
                                 eps_final=info['EPS_FINAL'],
                                 eps_final_frame=info['EPS_FINAL_FRAME'],
                                 eps_annealing_frames=info['EPS_ANNEALING_FRAMES'],
                                 eps_evaluation=info['EPS_EVAL'],
                                 replay_memory_start_size=info['MIN_HISTORY_TO_LEARN'],
                                 max_steps=info['MAX_STEPS'])

    if args.model_loadpath != '':
        # load data from loadpath - save model load for later. we need some of
        # these parameters to setup other things
        print('loading model from: %s' %args.model_loadpath)
        model_dict = torch.load(args.model_loadpath, map_location=torch.device(device))
        info = model_dict['info']
        info['DEVICE'] = device
        # set a new random seed
        info["SEED"] = model_dict['cnt']
        model_base_filedir = os.path.split(args.model_loadpath)[0]
        start_step_number = start_last_save = model_dict['cnt']
        perf = model_dict['perf']
        start_step_number = perf['steps'][-1]
    else:
        # create new project
        perf = {'steps':[],
                'avg_rewards':[],
                'episode_step':[],
                'episode_head':[],
                'eps_list':[],
                'episode_loss':[],
                'q_record':[],
                'episode_reward':[],
                'episode_times':[],
                'episode_relative_times':[],
                'eval_rewards':[],
                'highest_eval_score':[],
                'eval_stds':[],
                'eval_steps':[]}

        start_step_number = 0
        start_last_save = 0
        # make new directory for this run in the case that there is already a
        # project with this name
        run_num = 0
        model_base_filedir = os.path.join(config.model_savedir, info['NAME'] + '%02d'%run_num)
        while os.path.exists(model_base_filedir):
            run_num +=1
            model_base_filedir = os.path.join(config.model_savedir, info['NAME'] + '%02d'%run_num)
        os.makedirs(model_base_filedir)
        print("----------------------------------------------")
        print("starting NEW project: %s"%model_base_filedir)

    model_base_filepath = os.path.join(model_base_filedir, info['NAME'])
    write_info_file(info, model_base_filepath, start_step_number)
    heads = list(range(info['N_ENSEMBLE']))
    seed_everything(info["SEED"])
    #3,3,6 *2
    if 'swin' in info['IMPROVEMENT']:
        policy_net = SwinMLP(img_size=84, patch_size=3, in_chans=4, num_classes=env.num_actions, depths=[2, 3, 2], num_heads=[3, 3, 6], window_size=7).to(info['DEVICE'])
        target_net = SwinMLP(img_size=84, patch_size=3, in_chans=4, num_classes=env.num_actions, depths=[2, 3, 2], num_heads=[3, 3, 6] , window_size=7).to(info['DEVICE'])
    else:
        policy_net = EnsembleNet(n_ensemble=info['N_ENSEMBLE'],
                                        n_actions=env.num_actions,
                                        network_output_size=info['NETWORK_INPUT_SIZE'][0],
                                        num_channels=info['HISTORY_SIZE'], dueling=info['DUELING']).to(info['DEVICE'])
        target_net = EnsembleNet(n_ensemble=info['N_ENSEMBLE'],
                                        n_actions=env.num_actions,
                                        network_output_size=info['NETWORK_INPUT_SIZE'][0],
                                        num_channels=info['HISTORY_SIZE'], dueling=info['DUELING']).to(info['DEVICE'])


    target_net.load_state_dict(policy_net.state_dict())

    # create optimizer
    # opt = optim.RMSprop(policy_net.parameters(),
    #                    lr=info["RMS_LEARNING_RATE"],
    #                    momentum=info["RMS_MOMENTUM"],
    #                    eps=info["RMS_EPSILON"],
    #                    centered=info["RMS_CENTERED"],
    #                    alpha=info["RMS_DECAY"])
    opt = optim.Adam(policy_net.parameters(), lr=info['ADAM_LEARNING_RATE'])

    kl_loss = nn.KLDivLoss()
    ce_loss = nn.CrossEntropyLoss()
    #eval_states = []
    if args.model_loadpath != '':
        # nb: need to load replay memory and random states when use loading in training
        target_net.load_state_dict(model_dict['target_net_state_dict'])
        policy_net.load_state_dict(model_dict['policy_net_state_dict'])
        opt.load_state_dict(model_dict['optimizer'])
        print("loaded model state_dicts")
        # if args.buffer_loadpath == '':
        #     args.buffer_loadpath = args.model_loadpath.replace('.pkl', '_train_buffer.npz')
        #     print("auto loading buffer from:%s" %args.buffer_loadpath)
        #     try:
        #         replay_memory.load_buffer(args.buffer_loadpath)
        #     except Exception as e:
        #         print(e)
        #         print('not able to load from buffer: %s. exit() to continue with empty buffer' %args.buffer_loadpath)
        #final_mean, final_std, final_highest_score = evaluate(50e6, -1e6)
    else:
        train(start_step_number, start_last_save)

