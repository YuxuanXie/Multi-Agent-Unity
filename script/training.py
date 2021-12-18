from time import time
import ray
import sys
import argparse

sys.path.append('..')

from ray import tune
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy as PPOPolicyGraph
from ray.rllib.agents.ppo.appo_torch_policy import AsyncPPOTorchPolicy
from ray.rllib.agents.impala.vtrace_torch_policy import VTraceTorchPolicy

from ray.rllib.models import ModelCatalog
from ray.tune import run_experiments
from ray.tune.registry import register_env

from env.rllib_ma import Gc
from model.conv2mlp import TorchRNNModel

hparams = {
    'lr_init': 1e-5,
    'lr_final': 1e-5,
}
ppo_params = {
    'entropy_coeff': 0.0,
    # 'entropy_coeff_schedule': [[0, 0.01],[50000000, 0.001]],
    'use_gae': True,
    'kl_coeff': 0.0,
    "clip_param" : 0.1,
    "sgd_minibatch_size" : 256,
    "train_batch_size" : 1024,
    "num_sgd_iter" : 4,
    "rollout_fragment_length" : 64,
    "grad_clip" : 30,
    # "sgd_minibatch_size" : 128*5,
    # "train_batch_size" : 5000,
    # "num_sgd_iter" : 8,
    # "evaluation_interval" : 100,
    # "evaluation_num_episodes" : 50,
    # "evaluation_config" : {"explore": False},
    # "batch_mode" : "complete_episodes",
}

appo_param = {
    'entropy_coeff': 0.0,
    'use_gae': True,
    'kl_coeff': 0.0,
    "clip_param" : 0.1,
    "train_batch_size" : 2048,
    "minibatch_buffer_size" : 1024,
    "num_sgd_iter" : 4,
    "rollout_fragment_length" : 128,
    "grad_clip" : 30,
}


def setup(args):

    def env_creator(_):
        return Gc(args.env_path, no_graphics=args.render, numKeyFirst=args.keyFir, numKeySecond=args.keySec, time_scale=args.speed)

    single_env = Gc(args.env_path,)
    env_name = "gc_env"
    register_env(env_name, env_creator)

    obs_space = single_env.observation_space
    act_space = single_env.action_space
    single_env.close()

    # Each policy can have a different configuration (including custom model)
    def gen_policy():
        if args.algorithm == 'PPO':
            return (PPOPolicyGraph, obs_space, act_space, {})
        elif args.algorithm == "APPO":
            return (AsyncPPOTorchPolicy, obs_space, act_space, {})
        elif args.algorithm == "IMPALA":
            return (VTraceTorchPolicy, obs_space, act_space, {})
            

    # Setup PPO with an ensemble of `num_policies` different policy graphs
    policy_graphs = {}
    policy_graphs['policy-0'] = gen_policy()

    def policy_mapping_fn(agent_id):
        return 'policy-0'


    # # register the custom model
    model_name = "conv2mlp"
    ModelCatalog.register_custom_model(model_name, TorchRNNModel)

    agent_cls = get_agent_class(args.algorithm)
    config = agent_cls._default_config.copy()

    # information for replay
    config['env_config']['func_create'] = tune.function(env_creator)
    config['env_config']['env_name'] = env_name
    config['env_config']['run'] = args.algorithm

    # Calculate device configurations
    gpus_for_driver = int(args.use_gpu_for_driver)
    cpus_for_driver = 1 - gpus_for_driver
    if args.use_gpus_for_workers:
        spare_gpus = (args.num_gpus - gpus_for_driver)
        num_workers = int(spare_gpus * args.num_workers_per_device)
        num_gpus_per_worker = spare_gpus / num_workers
        num_cpus_per_worker = 0
    else:
        spare_cpus = (args.num_cpus - cpus_for_driver)
        num_workers = int(spare_cpus * args.num_workers_per_device)
        num_gpus_per_worker = 0
        num_cpus_per_worker = spare_cpus / num_workers

    # hyperparams
    config.update({
            # "train_batch_size": args.train_batch_size,
            # "horizon": 3 * one_layer_length, # it dosnot make done in step function true
            "lr_schedule":
            [[0, hparams['lr_init']],
                [5000000, hparams['lr_final']]],
            "num_workers": num_workers,
            "num_gpus": gpus_for_driver,  # The number of GPUs for the driver
            "num_cpus_for_driver": cpus_for_driver,
            "num_gpus_per_worker": num_gpus_per_worker,   # Can be a fraction
            "num_cpus_per_worker": num_cpus_per_worker,   # Can be a fraction
            "framework" : "torch",
            "multiagent": {
                "policies": policy_graphs,
                "policy_mapping_fn": tune.function(policy_mapping_fn),
            },
            "model": {
                "custom_model": "conv2mlp", 
                "lstm_cell_size": 128 ,
                "max_seq_len" : 8,
                # "custom_model_config": {
                #     "obs_shape" : 50,
                #     "entity_shape" : 31,
                #     "obs_embedding_size" : 128,
                #     "entity_embedding_size" : 128,
                #     "all_embedding_size" : 128,
                # }
            },
            "lambda" : args.lam,
            "gamma" : args.gamma,
        })
    if args.algorithm == 'PPO':
        config.update(ppo_params)
    elif args.algorithm == 'APPO':
        config.update(appo_param)

    config.update({"callbacks": {
        "on_episode_end": tune.function(on_episode_end),
    }})

    return args.algorithm, env_name, config


def on_episode_end(info):
    episode = info["episode"]
    info = episode._agent_to_last_info
    # if info["0"]:
    #     for i in range(4):
    #         episode.custom_metrics[f"reward{i}"] = info["0"]["final_reward"][i]
    #         episode.custom_metrics[f"size{i}"] = info["0"]["size"][str(i)]
    #         episode.custom_metrics["rank{}".format(info["0"]["rank"][i])] = i+1
    #         episode.custom_metrics["total_size{}".format(info["0"]["total_size"][i])] = i+1

def main():
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--exp_name', default='gc', help='Name of the ray_results experiment directory where results are stored.')
    parser.add_argument('--env_path', default='/Users/yuxuan/git/maUnity/env/unity_envs/GcMaze.app', help='Path to the game')
    parser.add_argument('--algorithm', default='PPO', help='Name of the rllib algorithm to use.')
    parser.add_argument('--restore', default='', help='Path to the checkpoint restored.')
    parser.add_argument('--train_batch_size', default=1024, type=int, help='Size of the total dataset over which one epoch is computed.')
    parser.add_argument('--checkpoint_frequency', default=100, type=int, help='Number of steps before a checkpoint is saved.')
    parser.add_argument('--training_iterations', default=20000, type=int, help='Total number of steps to train for')
    parser.add_argument('--num_cpus', default=32, type=int, help='Number of available CPUs')
    parser.add_argument('--num_gpus', default=0, type=int, help='Number of available GPUs')
    parser.add_argument('--use_gpus_for_workers', action='store_true', help='Set to true to run workers on GPUs rather than CPUs')
    parser.add_argument('--use_gpu_for_driver', action='store_true', help='Set to true to run driver on GPU rather than CPU.')
    parser.add_argument('--num_workers_per_device', default=1., type=float, help='Number of workers to place on a single device (CPU or GPU)')
    parser.add_argument('--num_cpus_for_driver', default=1., type=float, help='Number of workers to place on a single device (CPU or GPU)')
    parser.add_argument('--lam', default=0.95, type=float, help='lambda')
    parser.add_argument('--gamma', default=0.99, type=float, help='gamma')
    parser.add_argument('--render', action='store_false', help='Set to true to render the game')
    parser.add_argument('--keyFir', default=100, type=int, help='The number of keys in the first layer')
    parser.add_argument('--keySec', default=50, type=int, help='The number of keys in the first layer')
    parser.add_argument('--speed', default=5, type=int, help='The number of keys in the first layer')

    args = parser.parse_args()


    ray.init(num_cpus=args.num_cpus)
    # alg_run, env_name, config = setup(FLAGS.env, hparams, FLAGS.algorithm,
    #                                   FLAGS.train_batch_size,
    #                                   FLAGS.num_cpus,
    #                                   FLAGS.num_gpus,
    #                                   FLAGS.use_gpus_for_workers,
    #                                   FLAGS.use_gpu_for_driver,
    #                                   FLAGS.num_workers_per_device)

    alg_run, env_name, config = setup(args)

    if args.exp_name is None:
        exp_name = args.env + '_' + args.algorithm
    else:
        exp_name = args.exp_name + f'/{args.keyFir}_{args.keySec}/'

    print('Commencing experiment', exp_name)

    run_experiments({
            exp_name: {
                "run": alg_run,
                "env": env_name,
                "stop": {
                    "training_iteration": args.training_iterations
                },
                'checkpoint_freq': args.checkpoint_frequency,
                "config": config,
                # "restore": "/Users/yuxuan/git/gobigger/my_submission/entry/results/checkpoint_000800/checkpoint-800",
            }
        },
    )



if __name__ == '__main__':
    main()


