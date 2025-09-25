import gymnasium as gym
import highway_env
import random

if __name__ == "__main__":
    env = gym.make('merge-multi-agent-v0', render_mode='human',
                   config={
                    "traffic_density": 0,
                    })
    env.reset()
    print ('obs:',env.observation_space)
    print ('action:',env.action_space)
    for episode in range(10):
        env.reset()
        for step in range(200):
            action = [random.randint(0, 4)]
            next_obs, R, done, truncated, info = env.step(tuple(action))
            print ('next_obs:',next_obs)
            print ('reward :',R)
            print ('done: ',done)
            if done:
                break
    env.close()
