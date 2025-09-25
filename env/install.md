conda create -n highway-rl-env python=3.9

conda activate highway-rl-env

pip install gym 
pip install gymnasium 
pip install pygame 
pip install highway-env



test code
```
import gymnasium as gym
import highway_env
from highway_env.envs import HighwayEnv
import random

env = gym.make('highway-v0', render_mode='human',
                   config={
                       "controlled_vehicles": 1,
                       "vehicles_count": 10,
                       "vehicles_density": 0.5
                   })


observation_space = env.observation_space
print(observation_space)
action_space = env.action_space
print(action_space)

obs, info = env.reset()
for step in range(200):
    action = random.randint(0,4)

    obs, reward, done, truncated, info = env.step(action)

    if done:
        break

    env.render()
```


test control
```
import gymnasium as gym
import highway_env
from highway_env.envs import HighwayEnv
import random

env = gym.make('highway-v0', render_mode='human',
                   config={
                       "controlled_vehicles": 1,
                       "vehicles_count": 10,
                       "vehicles_density": 0.8
                   })


observation_space = env.observation_space
print(observation_space)
action_space = env.action_space
print(action_space)

obs, info = env.reset()
for step in range(50):
    
    
    #action = random.randint(0,4)
    action = 3
    for obs_i in range(1,5):
        if (obs[obs_i][1]<0.2) and (abs(obs[obs_i][2])<0.2):
            print (obs[obs_i][1])
            action = 4
            break

    obs, reward, done, truncated, info = env.step(action)

    

    if done:
        break

    env.render()


```