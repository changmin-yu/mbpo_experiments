from dm_control import suite
import gym
import numpy as np

class DeepMindControl:

    def __init__(self, name, seed=None):
        domain, task = name.split('_', 1)
        self.domain = domain
        if domain == 'cup':  # Only domain with multiple words.
            domain = 'ball_in_cup'
        if isinstance(domain, str):
            if seed:
                self._env = suite.load(domain, task, task_kwargs={'random': seed})
            else:
                self._env = suite.load(domain, task)
        else:
            assert task is None
            self._env = domain()

    @property
    def observation_space(self):
        init_obs = self._env.reset()
        init_obs = dict(init_obs.observation)
        init_obs = self.obs2arr(init_obs)
        return gym.spaces.Box(-np.inf, np.inf, init_obs.shape, dtype=np.float32)
        # spaces = {}
        # for key, value in self._env.observation_spec().items():
        #     spaces[key] = gym.spaces.Box(-np.inf, np.inf, value.shape, dtype=np.float32)
        # # spaces['image'] = gym.spaces.Box(
        # #     0, 255, self._size + (3,), dtype=np.uint8)
        # return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    def step(self, action):
        time_step = self._env.step(action)
        obs = dict(time_step.observation)
        obs = self.obs2arr(obs)
        # obs['image'] = self.render()
        reward = time_step.reward or 0
        done = time_step.last()
        info = {'discount': np.array(time_step.discount, np.float32)}
        return obs, reward, done, info

    def reset(self):
        time_step = self._env.reset()
        obs = dict(time_step.observation)
        obs = self.obs2arr(obs)
        # obs['image'] = self.render()
        return obs
    
    def obs2arr(self, obs):
        if self.domain == 'walker':
            obs['height'] = np.array([obs['height']])
        return np.concatenate(list(obs.values())).ravel()



#   def render(self, *args, **kwargs):
#     if kwargs.get('mode', 'rgb_array') != 'rgb_array':
#       raise ValueError("Only render mode 'rgb_array' is supported.")
#     return self._env.physics.render(*self._size, camera_id=self._camera)