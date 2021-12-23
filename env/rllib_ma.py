from time import sleep, time

from numpy import random
from env.gym_wrapper import unityEnv
from ray.rllib.env import MultiAgentEnv
import logging  

class Gc(MultiAgentEnv):
    def __init__(self, 
            env_path, 
            base_port=5005,
            worker_id=0,
            no_graphics=True,
            numKeyFirst = 100,
            numKeySecond = 50,
            time_scale = 5
            ) -> None:
        super().__init__()

        self._env = unityEnv(env_path, base_port, worker_id, no_graphics=no_graphics)
        self.num_agents = 4
        self.set_env_params(dict({
                "numKeyFirst":numKeyFirst,
                "numKeySecond":numKeySecond,
                "numFood":500,
                }))
        if time_scale >= 1:
            self._env.set_time_scale(time_scale=time_scale)

        self.done_agents = []
        self.observation_space = self._env.observation_shape
        self.action_space = self._env.action_shape
        self.current_step = 0


    def reset(self):
        self.done_agents = []
        self.current_step = 0
        obs = self._env.reset()
        self.num_agents = len(obs)
        return obs

    def step(self, actions):
        # avoid one agent not match when done
        obs, reward, done, info = self._env.step(actions)
        done["__all__"] = False
        self.current_step += 1

        for agent_id in list(obs.keys()):
            if agent_id in self.done_agents:
                obs.pop(agent_id, None)
                reward.pop(agent_id, None)
                done.pop(agent_id, None)
                info.pop(agent_id, None)
        logging.debug(f"step = {self.current_step} actions = {actions.keys()} agent_id = {obs.keys()} done_agents = {self.done_agents} reward = {reward}")
        if len(done):
            for agent_id in done.keys():
                if done[agent_id] and agent_id is not "__all__" and agent_id not in self.done_agents:
                    self.done_agents.append(agent_id)
    
        if len(self.done_agents) == self.num_agents:
            done["__all__"] = True

        return obs, reward, done, info

    def set_env_params(self, params):
        for k, v in params.items():
            self._env.set_env_parameters(k, v)

    def close(self) -> None:
        self._env.close()


if __name__ == "__main__":
    gc = Gc("./unity_envs/GcMaze100_slow.app", no_graphics=False, time_scale=0.01)
    obs = gc.reset()
    for _ in range(1000):
        actions = {}
        for agent in obs.keys():
            actions[agent] = [random.randint(0,3),random.randint(0,3),random.randint(0,3),random.randint(0,2)]
        new_obs, reward, done, info = gc.step(actions)
        if done["__all__"]:
            new_obs = gc.reset()
        obs = new_obs
    gc.close()
        




