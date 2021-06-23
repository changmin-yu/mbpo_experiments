import argparse
import time
import gym
import torch
import numpy as np
from itertools import count

import logging

import os
import os.path as osp
import json
import datetime
import pickle
import time

from torch.utils.tensorboard import SummaryWriter

from sac.replay_memory import ReplayMemory
from sac.sac import SAC
from model import EnsembleDynamicsModel
from predict_env import PredictEnv, PredictEnv_dm_control
from sample_env import EnvSampler
from tf_models.constructor import construct_model, format_samples_for_training
from wrappers import DeepMindControl

'''
TODO list: 
    - change the inner training iteration (l. 115). Done
    - remove printing holdout mse loss in dynamics model training
    - Add TensorBoard support
    - Add checkpoints for policy and dynamics model
'''


def readParser():
    parser = argparse.ArgumentParser(description='MBPO')
    parser.add_argument('--domain_name', default="walker",
                        help='DeepMind Control domain name (default: Walker)')
    parser.add_argument('--task_name', default='walk', 
                        help='DeepMind Control sub-task name (default: Walk)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')

    parser.add_argument('--use_decay', type=bool, default=True, metavar='G',
                        help='discount factor for reward (default: 0.99)')

    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')

    parser.add_argument('--num_networks', type=int, default=4, metavar='E',
                        help='ensemble size (default: 7)')
    parser.add_argument('--num_elites', type=int, default=3, metavar='E',
                        help='elite size (default: 5)')
    parser.add_argument('--pred_hidden_size', type=int, default=200, metavar='E',
                        help='hidden size for predictive model')
    parser.add_argument('--reward_size', type=int, default=1, metavar='E',
                        help='environment reward size')

    parser.add_argument('--replay_size', type=int, default=10000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')

    parser.add_argument('--model_retain_epochs', type=int, default=1, metavar='A',
                        help='retain epochs')
    parser.add_argument('--model_train_freq', type=int, default=250, metavar='A',
                        help='frequency of training')
    parser.add_argument('--rollout_batch_size', type=int, default=100000, metavar='A',
                        help='rollout number M')
    parser.add_argument('--epoch_length', type=int, default=1000, metavar='A',
                        help='steps per epoch')
    parser.add_argument('--rollout_min_epoch', type=int, default=20, metavar='A',
                        help='rollout min epoch')
    parser.add_argument('--rollout_max_epoch', type=int, default=150, metavar='A',
                        help='rollout max epoch')
    parser.add_argument('--rollout_min_length', type=int, default=1, metavar='A',
                        help='rollout min length')
    parser.add_argument('--rollout_max_length', type=int, default=15, metavar='A',
                        help='rollout max length')
    parser.add_argument('--num_epoch', type=int, default=1000, metavar='A',
                        help='total number of epochs')
    parser.add_argument('--min_pool_size', type=int, default=1000, metavar='A',
                        help='minimum pool size')
    parser.add_argument('--real_ratio', type=float, default=0.05, metavar='A',
                        help='ratio of env samples / model samples')
    parser.add_argument('--train_every_n_steps', type=int, default=1, metavar='A',
                        help='frequency of training policy')
    parser.add_argument('--num_train_repeat', type=int, default=20, metavar='A',
                        help='times to training policy per step')
    parser.add_argument('--max_train_repeat_per_step', type=int, default=5, metavar='A',
                        help='max training times per step')
    parser.add_argument('--policy_train_batch_size', type=int, default=256, metavar='A',
                        help='batch size for training policy')
    parser.add_argument('--init_exploration_steps', type=int, default=5000, metavar='A',
                        help='exploration steps initially')
    parser.add_argument('--num_eval_episode', type=int, default=5, metavar="A", 
                        help='number of episodes during evluation')
    parser.add_argument('--save_model_freq', type=int, default=100, metavar='A', 
                        help='frequency (in terms of number of episodes) for saving the model')

    # parser.add_argument('--model_type', default='tensorflow', metavar='A',
    #                     help='predict model -- pytorch or tensorflow')
    parser.add_argument('--model_type', default='pytorch', metavar='A',
                        help='predict model -- pytorch or tensorflow')

    parser.add_argument('--cuda', default=True, action="store_true",
                        help='run on CUDA (default: True)')
    parser.add_argument('--predict_done', default=False, action='store_true', 
                        help='Indicator of done (bool) prediction with the ensemble dynamics model')
    return parser.parse_args()


def train(args, env_sampler, predict_env, agent, env_pool, model_pool, writer, save_dir):
    dynamics_model_savedir = save_dir+'dynamics_model_cache/'
    policy_network_savedir = save_dir+'policy_network_cache/'
    os.makedirs(dynamics_model_savedir)
    os.makedirs(policy_network_savedir)
    model_pool_save_fname = save_dir+'model_buffer/'
    env_pool_save_fname = save_dir+'env_buffer/'
    os.makedirs(model_pool_save_fname)
    os.makedirs(env_pool_save_fname)
    total_step = 0
    reward_sum = 0
    rollout_length = 1
    exploration_before_start(args, env_sampler, env_pool, agent)

    for epoch_step in range(args.num_epoch):
        episode_start_time = time.time()
        start_step = total_step
        train_policy_steps = 0
        for i in count():
            cur_step = total_step - start_step

            # if cur_step >= start_step + args.epoch_length and len(env_pool) > args.min_pool_size:
            if cur_step >= args.epoch_length and len(env_pool) > args.min_pool_size:
                break

            if cur_step > 0 and cur_step % args.model_train_freq == 0 and args.real_ratio < 1.0:
                train_loss, elite_holdout_mse_loss = train_predict_model(args, env_pool, predict_env)
                writer.add_scalar('losses/dynamics_model_training_loss', train_loss, total_step)
                writer.add_scalar('losses/dynamics_model_elite_holdout_mse_loss', elite_holdout_mse_loss, total_step)
                
                new_rollout_length = set_rollout_length(args, epoch_step)
                if rollout_length != new_rollout_length:
                    rollout_length = new_rollout_length
                    model_pool = resize_model_pool(args, rollout_length, model_pool)
                writer.add_scalar('parameters/dynamics_model_horizon', new_rollout_length, total_step)

                rollout_model(args, predict_env, agent, model_pool, env_pool, rollout_length)
                

            cur_state, action, next_state, reward, done, info = env_sampler.sample(agent)
            env_pool.push(cur_state, action, reward, next_state, int(done))

            if len(env_pool) > args.min_pool_size:
                train_policy_step_increment, d = train_policy_repeats(args, total_step, train_policy_steps, cur_step, env_pool, model_pool, agent)
                if d is not None:
                    writer.add_scalar('losses/q1_loss', d['q1_loss'], total_step)
                    writer.add_scalar('losses/q2_loss', d['q2_loss'], total_step)
                    writer.add_scalar('losses/q_loss', d['q_loss'], total_step)
                    writer.add_scalar('losses/policy_loss', d['policy_loss'], total_step)
                    writer.add_scalar('losses/alpha_loss', d['alpha_loss'], total_step)
                    writer.add_scalar('losses/alpha', d['alpha'], total_step)
                
                # train_policy_steps += train_policy_repeats(args, total_step, train_policy_steps, cur_step, env_pool, model_pool, agent)
                train_policy_steps += train_policy_step_increment
                
            total_step += 1

            if total_step % 1000 == 0:
                '''
                avg_reward_len = min(len(env_sampler.path_rewards), 5)
                avg_reward = sum(env_sampler.path_rewards[-avg_reward_len:]) / avg_reward_len
                logging.info("Step Reward: " + str(total_step) + " " + str(env_sampler.path_rewards[-1]) + " " + str(avg_reward))
                print(total_step, env_sampler.path_rewards[-1], avg_reward)
                '''
                avg_episode_reward = 0
                for j in range(args.num_eval_episode):
                    env_sampler.current_state = None
                    sum_reward = 0
                    done = False
                    while not done:
                        cur_state, action, next_state, reward, done, info = env_sampler.sample(agent, eval_t=True)
                        sum_reward += reward
                        logging.info('step reward: {} | eval num: {} | sum reward: {}'.format(total_step, j, sum_reward))
                    avg_episode_reward += sum_reward
                writer.add_scalar('returns/mean_eval_return', avg_episode_reward/args.num_eval_episode, total_step)

                env_sampler.current_state = None
                sum_reward = 0
                done = False
                while not done:
                    cur_state, action, next_state, reward, done, info = env_sampler.sample(agent, eval_t=True)
                    sum_reward += reward
                # logger.record_tabular("total_step", total_step)
                # logger.record_tabular("sum_reward", sum_reward)
                # logger.dump_tabular()
                logging.info("Step Reward: " + str(total_step) + " " + str(sum_reward))
                # print(total_step, sum_reward)
        if (epoch_step) % args.save_model_freq == 0 or epoch_step == args.num_epoch-1:
            torch.save({
                'epoch': epoch_step, 
                'dynamics_model_state_dict': predict_env.model.ensemble_model.state_dict(), 
                'dynamics_model_optimiser_state_dict': predict_env.model.ensemble_model.optimizer.state_dict(), 
                'dynamics_model_loss': train_loss if epoch_step > 0 else 0, 
            }, dynamics_model_savedir+str(epoch_step)+'.pt')
            torch.save({
                'epoch': epoch_step, 
                'policy_model_state_dict': agent.policy.state_dict(), 
                'policy_model_optimiser_state_dict': agent.policy_optim.state_dict(), 
                'policy_model_loss': d['policy_loss'] if epoch_step > 0 else 0, 
            }, policy_network_savedir+str(epoch_step)+'.pt')
            logging.info("Epoch step: " + str(epoch_step) + " EnsembleDynamicsModel and PolicyNetwork checkpoint saved")
            # torch.save(predict_env.model.state_dict(), args.save_model_path+'EnsembleDynamicsModel_'+str(epoch_step+1)+'.pt')
            # torch.save(predict_)
            env_pool.save(env_pool_save_fname+str(epoch_step)+'.pkl')
            model_pool.save(model_pool_save_fname+str(epoch_step)+'.pkl')
            scalar_mu_std = [predict_env.model.scaler.mu, predict_env.model.scaler.std]
            with open(save_dir+'scaler_mu_std_{}.pkl'.format(epoch_step), 'wb') as f:
                pickle.dump(scalar_mu_std, f)
        print('epoch: {} | steps: {}'.format(epoch_step, i))
        writer.add_scalar('time/episode_running_time', time.time()-episode_start_time, epoch_step)


def exploration_before_start(args, env_sampler, env_pool, agent):
    for i in range(args.init_exploration_steps):
        cur_state, action, next_state, reward, done, info = env_sampler.sample(agent)
        env_pool.push(cur_state, action, reward, next_state, int(done))


def set_rollout_length(args, epoch_step):
    rollout_length = (min(max(args.rollout_min_length + (epoch_step - args.rollout_min_epoch)
                              / (args.rollout_max_epoch - args.rollout_min_epoch) * (args.rollout_max_length - args.rollout_min_length),
                              args.rollout_min_length), args.rollout_max_length))
    return int(rollout_length)


def train_predict_model(args, env_pool, predict_env):
    # Get all samples from environment
    state, action, reward, next_state, done = env_pool.sample(len(env_pool))
    delta_state = next_state - state
    inputs = np.concatenate((state, action), axis=-1)
    if args.predict_done:
        labels = np.concatenate((np.reshape(reward, (reward.shape[0], -1)), delta_state, np.reshape(done*1., (done.shape[0], -1))), axis=-1)
    else:
        labels = np.concatenate((np.reshape(reward, (reward.shape[0], -1)), delta_state), axis=-1)
    train_loss, holdout_mse_loss = predict_env.model.train(inputs, labels, batch_size=256, holdout_ratio=0.2)
    elite_holdout_mse_loss = torch.mean(torch.topk(holdout_mse_loss, args.num_elites)[0]).item()
    return train_loss, elite_holdout_mse_loss

def resize_model_pool(args, rollout_length, model_pool):
    rollouts_per_epoch = args.rollout_batch_size * args.epoch_length / args.model_train_freq
    model_steps_per_epoch = int(rollout_length * rollouts_per_epoch)
    new_pool_size = args.model_retain_epochs * model_steps_per_epoch

    sample_all = model_pool.return_all()
    new_model_pool = ReplayMemory(new_pool_size)
    new_model_pool.push_batch(sample_all)

    return new_model_pool


def rollout_model(args, predict_env, agent, model_pool, env_pool, rollout_length):
    state, action, reward, next_state, done = env_pool.sample_all_batch(args.rollout_batch_size)
    for i in range(rollout_length):
        # TODO: Get a batch of actions
        action = agent.select_action(state)
        next_states, rewards, terminals, info = predict_env.step(state, action)
        # TODO: Push a batch of samples
        model_pool.push_batch([(state[j], action[j], rewards[j], next_states[j], terminals[j]) for j in range(state.shape[0])])
        terminals_bool = terminals.astype(int) > 0
        nonterm_mask = ~terminals_bool.squeeze(-1)
        if nonterm_mask.sum() == 0:
            break
        state = next_states[nonterm_mask]


def train_policy_repeats(args, total_step, train_step, cur_step, env_pool, model_pool, agent):
    if total_step % args.train_every_n_steps > 0:
        return 0, None

    if train_step > args.max_train_repeat_per_step * total_step:
        return 0, None

    qf1_loss_list, qf2_loss_list, qf_loss_list, policy_loss_list, alpha_loss_list, alpha_list = [], [], [], [], [], [] 
    for i in range(args.num_train_repeat):
        env_batch_size = int(args.policy_train_batch_size * args.real_ratio)
        model_batch_size = args.policy_train_batch_size - env_batch_size

        env_state, env_action, env_reward, env_next_state, env_done = env_pool.sample(int(env_batch_size))

        if model_batch_size > 0 and len(model_pool) > 0:
            model_state, model_action, model_reward, model_next_state, model_done = model_pool.sample_all_batch(int(model_batch_size))
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = np.concatenate((env_state, model_state), axis=0), \
                                                                                    np.concatenate((env_action, model_action), axis=0), \
                                                                                    np.concatenate((np.reshape(env_reward, (env_reward.shape[0], -1)), model_reward), axis=0), \
                                                                                    np.concatenate((env_next_state, model_next_state), axis=0), \
                                                                                    np.concatenate((np.reshape(env_done, (env_done.shape[0], -1)), model_done), axis=0)
        else:
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = env_state, env_action, env_reward, env_next_state, env_done

        batch_reward, batch_done = np.squeeze(batch_reward), np.squeeze(batch_done)
        batch_mask = 1-batch_done
        qf1_loss, qf2_loss, policy_loss, alpha_loss, alpha_tlogs = agent.update_parameters((batch_state, batch_action, batch_reward, batch_next_state, batch_mask), args.policy_train_batch_size, i)
        qf1_loss_list.append(qf1_loss)
        qf2_loss_list.append(qf2_loss)
        qf_loss_list.append(qf1_loss + qf2_loss)
        policy_loss_list.append(policy_loss)
        alpha_loss_list.append(alpha_loss)
        alpha_list.append(alpha_tlogs)
    
    qf1_loss = torch.mean(torch.tensor(qf1_loss_list)).item()
    qf2_loss = torch.mean(torch.tensor(qf2_loss_list)).item()
    qf_loss = torch.mean(torch.tensor(qf_loss_list)).item()
    policy_loss = torch.mean(torch.tensor(policy_loss_list)).item()
    alpha_loss = torch.mean(torch.tensor(alpha_loss_list)).item()
    mean_alpha = torch.mean(torch.tensor(alpha_list)).item()
    return args.num_train_repeat, {'q1_loss': qf1_loss, 'q2_loss': qf2_loss, 'q_loss': qf_loss, 'policy_loss': policy_loss, 'alpha_loss': alpha_loss, 'alpha': mean_alpha}


from gym.spaces import Box


class SingleEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super(SingleEnvWrapper, self).__init__(env)
        obs_dim = env.observation_space.shape[0]
        obs_dim += 2
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        torso_height, torso_ang = self.env.sim.data.qpos[1:3]  # Need this in the obs for determining when to stop
        obs = np.append(obs, [torso_height, torso_ang])

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        torso_height, torso_ang = self.env.sim.data.qpos[1:3]
        obs = np.append(obs, [torso_height, torso_ang])
        return obs


def main(args=None):
    if args is None:
        args = readParser()

    # Initial environment
    # env = gym.make(args.env_name)
    timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    job_name = 'MBPO_dm_control_{}_{}_{}_{}_{}'.format(args.domain_name, args.task_name, args.model_type, args.seed, timestamp)
    save_dir = './save_dir/{}/'.format(job_name)
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
        
    env = DeepMindControl('{}_{}'.format(args.domain_name, args.task_name), args.seed)

    writer = SummaryWriter(save_dir+"tensorboard/{}".format(job_name))
    writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
        '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))
    


    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # env.seed(args.seed)

    # Intial agent
    agent = SAC(env.observation_space.shape[0], env.action_space, args)

    # Initial ensemble model
    state_size = np.prod(env.observation_space.shape)
    action_size = np.prod(env.action_space.shape)
    if args.model_type == 'pytorch':
        env_model = EnsembleDynamicsModel(args.num_networks, args.num_elites, state_size, action_size, args.reward_size, args.pred_hidden_size,
                                          use_decay=args.use_decay, predict_done=args.predict_done)
    else:
        env_model = construct_model(obs_dim=state_size, act_dim=action_size, hidden_dim=args.pred_hidden_size, num_networks=args.num_networks,
                                    num_elites=args.num_elites)

    # Predict environments
    predict_env = PredictEnv_dm_control(env_model, args.model_type)

    # Initial pool for env
    env_pool = ReplayMemory(args.replay_size)
    # Initial pool for model
    rollouts_per_epoch = args.rollout_batch_size * args.epoch_length / args.model_train_freq
    model_steps_per_epoch = int(1 * rollouts_per_epoch)
    new_pool_size = args.model_retain_epochs * model_steps_per_epoch
    model_pool = ReplayMemory(new_pool_size)

    # Sampler of environment
    env_sampler = EnvSampler(env)

    train(args, env_sampler, predict_env, agent, env_pool, model_pool, writer, save_dir)


if __name__ == '__main__':
    main()
