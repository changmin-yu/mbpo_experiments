import numpy as np
import torch
import logging
import json
import os
import gym
import argparse
import pickle

from model import *
from sac.sac import SAC
from sample_env import EnvSampler, Predict_EnvSample
from predict_env import PredictEnv



def readParser():
    parser = argparse.ArgumentParser(description='MBPO')
    parser.add_argument('--env_name', default="Hopper-v2",
                        help='Mujoco Gym environment (default: Hopper-v2)')
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

    parser.add_argument('--num_networks', type=int, default=7, metavar='E',
                        help='ensemble size (default: 7)')
    parser.add_argument('--num_elites', type=int, default=5, metavar='E',
                        help='elite size (default: 5)')
    parser.add_argument('--pred_hidden_size', type=int, default=200, metavar='E',
                        help='hidden size for predictive model')
    parser.add_argument('--reward_size', type=int, default=1, metavar='E',
                        help='environment reward size')

    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
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
    parser.add_argument('--num_epoch', type=int, default=125, metavar='A',
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
    
    parser.add_argument('--save_model_freq', type=int, default=20, metavar='A', 
                        help='frequency (in terms of number of episodes) for saving the model')
    parser.add_argument('--save_model_path', type=str, default='test_policy_dependent_results_2/dynamics_model/', metavar='A', 
                        help='directory for saving the dynamics model')
    parser.add_argument('--save_policy_path', type=str, default='test_policy_dependent_results_2/policy/', metavar='A', 
                        help='directory for saving the policy network')
    parser.add_argument('--save_scaler_path', type=str, default='test_policy_dependent_results_2/scaler_mu_std_40.pkl', metavar='A', 
                        help='directory for saving the policy network')

    # parser.add_argument('--model_type', default='tensorflow', metavar='A',
    #                     help='predict model -- pytorch or tensorflow')
    parser.add_argument('--model_type', default='pytorch', metavar='A',
                        help='predict model -- pytorch or tensorflow')

    parser.add_argument('--cuda', default=True, action="store_true",
                        help='run on CUDA (default: True)')
    return parser.parse_args()



def test_policy_dependent_models(args, env, state_size, action_size, env_sampler):
    save_freq = args.save_model_freq
    # checkpoint_epochs = np.arange(0, args.num_epoch, save_freq)
    # checkpoint_epochs = np.arange(20, 40, 2)
    checkpoint_epochs = [20, 26, 32, 38]
    # checkpoint_epochs = np.append(checkpoint_epochs, args.num_epoch-1)
    model_policy_return_dict = {}
    state_error_dict = {}
    reward_error_dict = {}
    with open(args.save_scaler_path, 'rb') as f:
        mean, std = pickle.load(f)
    for model_epoch in checkpoint_epochs:
        dynamics_model_checkpoint = torch.load(args.save_model_path+'EnsembleDynamicsModel_'+str(int(model_epoch))+'.pt')
        env_model = EnsembleDynamicsModel(args.num_networks, args.num_elites, state_size, action_size, args.reward_size, args.pred_hidden_size, 
                                          use_decay=args.use_decay)
        env_model.ensemble_model.load_state_dict(dynamics_model_checkpoint['dynamics_model_state_dict'])
        env_model.scaler.mu = mean
        env_model.scaler.std = std
        print('dynamics_model_{} loaded'.format(model_epoch))
        predict_env = PredictEnv(env_model, args.env_name, args.model_type)
        predict_env_sampler = Predict_EnvSample(env, predict_env)
        for policy_epoch in checkpoint_epochs:
            policy_network_checkpoint = torch.load(args.save_policy_path+'PolicyNetwork_'+str(int(policy_epoch))+'.pt')
            agent = SAC(env.observation_space.shape[0], env.action_space, args)
            agent.policy.load_state_dict(policy_network_checkpoint['policy_model_state_dict'])
            avg_episode_reward = []
            for i in range(args.num_eval_episode):
                predict_env_sampler.current_state = None
                sum_reward = 0
                done = False
                counter = 0
                state_error = []
                reward_error = []
                while not done and counter < args.epoch_length:
                    cur_state, action, next_state, reward, done, info, model_error = predict_env_sampler.sample(agent, eval_t=True, ret_true_reward=False)
                    sum_reward += reward
                    counter += 1
                    state_error.append(model_error[0])
                    reward_error.append(model_error[1])
                # logging.info('Policy epoch{} | DynamicsModel epoch{} | number of steps: {} | inner eval num: {} | sum reward: {} | model_error: {}'.format(policy_epoch, model_epoch, counter, i, sum_reward, np.sum(model_error_list)))
                avg_episode_reward.append(sum_reward)
                # writer.add_scalar('returns/mean_eval_return_model_{}_policy_{}'.format(model_epoch, policy_epoch), sum_reward, i)
            mean_episode_reward = torch.mean(torch.tensor(avg_episode_reward)*1.)
            std_episode_reward = torch.std(torch.tensor(avg_episode_reward)*1.)
            model_policy_return_dict['model_{}_policy_{}'.format(model_epoch, policy_epoch)] = [mean_episode_reward.item(), std_episode_reward.item()]
            state_error_dict['model_{}_policy_{}'.format(model_epoch, policy_epoch)] = state_error
            reward_error_dict['model_{}_policy_{}'.format(model_epoch, policy_epoch)] = reward_error
            print('model epoch: {} | policy epoch: {} | mean return: {:.3f} | state error: {:.2f} | reward error: {:.2f} | total steps: {} | Done'.format(model_epoch, policy_epoch, mean_episode_reward, np.mean(state_error), np.mean(reward_error), counter))
    with open('test_policy_dependent_results_2/mean_std_evaluated_policy_20_6_38.json', 'w') as f:
        json.dump(model_policy_return_dict, f)
    with open('test_policy_dependent_results_2/state_error_dict_20_6_38.json', 'w') as f:
        json.dump(state_error_dict, f)
    with open('test_policy_dependent_results_2/reward_error_dict_20_6_38.json', 'w') as f:
        json.dump({k: np.array(v).astype(np.float64).tolist() for k, v in reward_error_dict.items()}, f)
    f.close()
    
    
def main(args=None):
    if args is None:
        args = readParser()
        
    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)
    if not os.path.exists(args.save_policy_path):
        os.makedirs(args.save_policy_path)

    # Initial environment
    env = gym.make(args.env_name)
    
    # job_name = 'MBPO_test_policy_dependent_models_{}_{}_{}'.format(args.env_name, args.model_type, args.seed)
    # writer = SummaryWriter("test_policy_dependent_results/tensorboard/{}".format(job_name))
    # writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
    #     '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))

    # # Set random seed
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # env.seed(args.seed)

    # Intial agent
    agent = SAC(env.observation_space.shape[0], env.action_space, args)

    # Initial ensemble model
    state_size = np.prod(env.observation_space.shape)
    action_size = np.prod(env.action_space.shape)
    if args.model_type == 'pytorch':
        env_model = EnsembleDynamicsModel(args.num_networks, args.num_elites, state_size, action_size, args.reward_size, args.pred_hidden_size,
                                          use_decay=args.use_decay)
    # else:
    #     env_model = construct_model(obs_dim=state_size, act_dim=action_size, hidden_dim=args.pred_hidden_size, num_networks=args.num_networks,
    #                                 num_elites=args.num_elites)

    # Predict environments
    # predict_env = PredictEnv(env_model, args.env_name, args.model_type)

    # Initial pool for env
    # env_pool = ReplayMemory(args.replay_size)
    # # Initial pool for model
    # rollouts_per_epoch = args.rollout_batch_size * args.epoch_length / args.model_train_freq
    # model_steps_per_epoch = int(1 * rollouts_per_epoch)
    # new_pool_size = args.model_retain_epochs * model_steps_per_epoch
    # model_pool = ReplayMemory(new_pool_size)

    # Sampler of environment
    env_sampler = EnvSampler(env)


    # train(args, env_sampler, predict_env, agent, env_pool, model_pool, writer)
    
    print('Training complete!')
    print('---------------------------------------------------------------------')
    print('Start evaluating different policies at different model checkpoints...')
    print('---------------------------------------------------------------------')
    test_policy_dependent_models(args, env, state_size, action_size, env_sampler)    


if __name__ == '__main__':
    main()