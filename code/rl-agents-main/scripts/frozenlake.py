import gymnasium as gym
import matplotlib.pyplot as plt
print(gym.envs.registry.values())
# 获取所有环境的名字
env_names = [env_spec.id for env_spec in gym.envs.registry.values()]
# 每行显示10个环境名称
num_per_line = 6
for i in range(0, len(env_names), num_per_line):
    print(", ".join(env_names[i:i+num_per_line]))
exit()
# env= gym.make("FrozenLake-v1", render_mode="rgb_array")
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")
#查看
# print("Observation space:", env.observation_space)
# print("Action space:", env.action_space)
# print("Observation space 类型:", type(env.observation_space))
# print("Action space 类型:", type(env.action_space))
obs, _ = env.reset()
# pic=env.render()
print("初始观测值（状态）:", obs)
# observation, reward, terminated, truncated, info= env.step(env.action_space.sample())
observation, reward, terminated, truncated, info= env.step(2)

print(env.step(0))
# pic=env.render()
plt.imshow(env.render())
# plt.imshow(pic)
plt.show()
env.close()
