import numpy as np
import datetime
import gym
import os
import robosuite as suite
from robosuite.controllers import load_controller_config
import pickle
from tqdm import tqdm
from .sac import SAC
from .replay_memory import ReplayMemory

class EvaluateSAC:
    def __init__(self, config):
        self.config = config
        self.env_name = config['task']['name']
        self.run_name = config['run_name']
        self.object = None if config['object'] == '' else config['object']
        self.render = config['render']
        self.num_steps = config['sac']['num_steps']
        self.num_eval = config['sac']['num_eval']
        self.args = config['sac']

        self.robot = config['task']['env']['robot']
        self.eval()

    def reset_env(self, env, get_objs=False):
        obs = env.reset()
        if get_objs:
            if self.env_name == 'Lift':
                objs = obs['cube_pos']
            elif self.env_name == 'Stack':
                objs = np.concatenate((obs['cubeA_pos'], obs['cubeB_pos']), axis=-1)
            elif self.env_name == 'NutAssembly':
                nut = 'RoundNut'
                objs = obs[nut + '_pos']
            elif self.env_name == 'PickPlace':
                objs = obs[self.object + '_pos']
            elif self.env_name == 'Door':
                objs = np.concatenate((obs['door_pos'], obs['handle_pos']), axis=-1)
            return obs, objs
        return obs

    def get_state(self, obs, objs):
        state = list(obs['robot0_eef_pos']) + list(objs)
        return np.array(state)

    def eval(self):
        save_data = {'episode': [], 'reward': []}
        save_name = f"models/{self.env_name}/{self.run_name}" if self.object is None else f"models/{self.env_name}/{self.object}/{self.run_name}"
        os.makedirs(save_name, exist_ok=True)

        controller_config = load_controller_config(default_controller="OSC_POSE")
        env = suite.make(
            env_name=self.env_name,
            robots=self.robot,
            controller_configs=controller_config,
            has_renderer=self.render,
            reward_shaping=True,
            control_freq=10,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            initialization_noise=None,
            single_object_mode=2,
            object_type=self.object,
            use_latch=False,
        )

        obs, objs = self.reset_env(env, get_objs=True)
        env_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float64)
        agent = SAC(len(obs['robot0_eef_pos']) + len(objs), env_action_space, self.args)
        agent.load_checkpoint(f"{save_name}/models", evaluate=True)

        total_numsteps = 0
        for i_episode in tqdm(range(1, self.num_eval + 1)):
            episode_reward = 0
            episode_steps = 0
            obs, objs = self.reset_env(env, get_objs=True)
            state = self.get_state(obs, objs)

            for timestep in range(1, self.num_steps + 1):
                if self.render:
                    env.render()

                action = agent.select_action(state)

                full_action = list(action[0:3]) + [0.] * 3 + [action[3]]
                full_action = np.array(full_action)

                obs, reward, _, _ = env.step(full_action)
                episode_steps += 1
                total_numsteps += 1
                episode_reward += reward

                next_objs = self.reset_env(env, get_objs=True)[1]
                state = self.get_state(obs, next_objs)

            save_data['episode'].append(i_episode)
            save_data['reward'].append(episode_reward)
            pickle.dump(save_data, open(f"{save_name}/eval_reward.pkl", 'wb'))
            tqdm.write(f"Saved test data to {save_name}/eval_reward.pkl")

        exit()