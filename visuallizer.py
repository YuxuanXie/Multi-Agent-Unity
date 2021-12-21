import os
import argparse
import numpy as np
import pickle
import torch
from env.rllib_ma import Gc
from model.conv2mlp import TorchRNNModel

"""Defines a multi-agent controller to rollout environment episodes w/
   agent policies."""

import argparse
import json
import numpy as np
import os
import shutil
import sys

import ray
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.cloudpickle import cloudpickle
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID

from model.conv2mlp import TorchRNNModel


def get_rllib_config(path):
    """Return the data from the specified rllib configuration file."""
    jsonfile = path + '/params.json'  # params.json is the config file
    jsondata = json.loads(open(jsonfile).read())
    return jsondata


def get_rllib_pkl(path):
    """Return the data from the specified rllib configuration file."""
    pklfile = path + '/params.pkl'  # params.json is the config file
    with open(pklfile, 'rb') as file:
        pkldata = cloudpickle.load(file)
    return pkldata


def visualizer_rllib(args):
    result_dir = args.result_dir if args.result_dir[-1] != '/' else args.result_dir[:-1]

    config = get_rllib_config(result_dir)
    pkl = get_rllib_pkl(result_dir)

    # check if we have a multiagent scenario but in a
    # backwards compatible way
    if config.get('multiagent', {}).get('policies', {}):
        multiagent = True
        config['multiagent'] = pkl['multiagent']
    else:
        multiagent = False

    def env_creator(_):
        return Gc("env/linux/GcMaze.x86_64", no_graphics=False, numKeyFirst=100, numKeySecond=50, time_scale=1)

    # Create and register a gym+rllib env
    env_name = config['env_config']['env_name']
    register_env(env_name, env_creator)

    ModelCatalog.register_custom_model("conv2mlp", TorchRNNModel)

    # Determine agent and checkpoint
    config_run = config['env_config']['run'] if 'run' in config['env_config'] else None

    if (config_run):
        agent_cls = get_agent_class(config_run)
    else:
        print('visualizer_rllib.py: error: could not find flow parameter '
              '\'run\' in params.json, '
              'add argument --run to provide the algorithm or model used '
              'to train the results\n e.g. '
              'python ./visualizer_rllib.py /tmp/ray/result_dir 1 --run PPO')
        sys.exit(1)

    # Run on only one cpu for rendering purposes if possible; A3C requires two
    if config_run == 'A3C':
        config['num_workers'] = 1
        config["sample_async"] = False
    else:
        config['num_workers'] = 0

    # create the agent that will be used to compute the actions
    config['sample_collector'] = None
    agent = agent_cls(env=env_name, config=config)

    checkpoint = result_dir + '/checkpoint_' + f'{int(args.checkpoint_num):06d}'
    checkpoint = checkpoint + '/checkpoint-' + args.checkpoint_num
    print('Loading checkpoint', checkpoint)
    agent.restore(checkpoint)
    if hasattr(agent, "workers"):
        env = agent.workers.local_worker().env

    # import pdb; pdb.set_trace()
    if hasattr(agent, "workers"):
        multiagent = agent.workers.local_worker().multiagent
        if multiagent:
            policy_agent_mapping = agent.config["multiagent"][
                "policy_mapping_fn"]
            mapping_cache = {}
        policy_map = agent.workers.local_worker().policy_map
        state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
        use_lstm = {p: len(s) > 0 for p, s in state_init.items()}
    else:
        multiagent = False
        use_lstm = {DEFAULT_POLICY_ID: False}

    steps = 0
    # while steps < (config['horizon'] or steps + 1):
    state = env.reset()
    done = False
    reward_total = 0.0
    while not done and steps < (config['horizon'] or steps + 1):
        if multiagent:
            action_dict = {}
            for agent_id in state.keys():
                a_state = state[agent_id]
                if a_state is not None:
                    policy_id = mapping_cache.setdefault(
                        agent_id, policy_agent_mapping(agent_id))
                    p_use_lstm = use_lstm[policy_id]
                    if p_use_lstm:
                        a_action, p_state_init, _ = agent.compute_action(
                            a_state,
                            state=state_init[policy_id],
                            policy_id=policy_id,
                            )
                        state_init[policy_id] = p_state_init
                    else:
                        a_action = agent.compute_action(
                            a_state, policy_id=policy_id)
                    action_dict[agent_id] = a_action
            action = action_dict
        else:
            if use_lstm[DEFAULT_POLICY_ID]:
                action, state_init, _ = agent.compute_action(
                    state, state=state_init)
            else:
                action = agent.compute_action(state)

        if agent.config["clip_actions"]:
            # clipped_action = clip_action(action, env.action_space)
            next_state, reward, done, info = env.step(action)
        else:
            next_state, reward, done, info = env.step(action)
        print(steps, action) 
        if multiagent:
            done = done["__all__"]
            reward_total += sum(reward.values())
        else:
            reward_total += reward

        steps += 1
        state = next_state
        print(f"step = {steps}")



def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Evaluates a reinforcement learning agent '
                    'given a checkpoint.')

    # required input parameters
    parser.add_argument('result_dir', type=str, help='Directory containing results')
    parser.add_argument('checkpoint_num', type=str, help='Checkpoint number.')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    ray.init(num_cpus=2)
    visualizer_rllib(args)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='arguments')
#     parser.add_argument('--binPath', type=str, default='./env/visuallizer.app')
#     parser.add_argument('--checkpoint', type=str, default='results/model/2021-10-17-11-59-50/900000.pth')
#     args = parser.parse_args()
#     main(args)




