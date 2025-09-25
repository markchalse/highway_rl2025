import gymnasium as gym
import highway_env
import random


if __name__ == "__main__":
    AGENTS_NUM = 3 #自动驾驶车辆数量
    env = gym.make('merge-multi-agent-v0', render_mode='human',
                   config={
                    "traffic_density": 999,
                    "controlled_vehicles":AGENTS_NUM,
                    #"perception_distance":100
                    })
    env.reset()

    print ('obs:',env.observation_space)
    print ('action:',env.action_space)


    for episode in range(10):
        env.reset()
        for step in range(200):
            actions = tuple(random.randint(0, 4) for _ in range(AGENTS_NUM))
            print ('actions:',actions)
            next_obs, R, done, truncated, info = env.step(actions)
            print ('next_obs:',next_obs)
            print ('rewards :',info['agents_rewards'])
            print ('dones: ',info['agents_dones'])
            
            if done:
                break
    env.close()