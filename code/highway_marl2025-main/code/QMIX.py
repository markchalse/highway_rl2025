#Attention: If the following error occurs, try using the following solutions
#OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
#OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/.
#import os
#os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import gymnasium as gym
import highway_env
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter  # 导入 TensorBoard
from model import QMIX,FlattenObs
# 初始化 TensorBoard
writer = SummaryWriter(log_dir="runs/QMIX_experiment") # 日志保存到 runs/QMIX_experiment 目录


def get_global_state(origin_obses):
    vehicles = []
    for origin_obs in origin_obses:
        vehicles.append(origin_obs[0].tolist())
    vehicles.append([1,0.61765,1.0,0,0])
    sort_index = sorted(range(len(vehicles)),key=lambda i: vehicles[i][1])
    global_state = []
    for obj_i in sort_index:
        for number in vehicles[obj_i]:
            global_state.append(number)
    return global_state


if __name__ == "__main__":
    AGENTS_NUM = 3 #自动驾驶车辆数量
    env = gym.make('merge-multi-agent-v0', render_mode='human',
                   config={
                    "traffic_density": 999,
                    "controlled_vehicles":AGENTS_NUM,
                    "perception_distance":100
                    })
    env.reset()
    #print ('obs:',env.observation_space)
    # init dqn model
    #qmix_model = QMIX(state_size = 25, action_size=5,agent_num=AGENTS_NUM)
    #qmix_model = QMIX(state_size = 25, action_size=5,agent_num=AGENTS_NUM, model_file_path='qmix.pth',use_epsilon=False)
    qmix_model = QMIX(state_size = 25, action_size=5,agent_num=AGENTS_NUM, model_file_path='endpoint/qmix.pth.3-6-16',use_epsilon=False)
    # 训练循环
    for episode in range(800):
        print (f'episode {episode} begin')
        obs = env.reset()[0] #The first observation information extraction
        states=[FlattenObs(obs_i) for obs_i in obs]
        global_s = get_global_state(obs) #qmix
        total_reward = 0
        for step in range(200):
            actions = [qmix_model.get_action(state) for state in states]
            next_obs, R, done, truncated, info = env.step(tuple(actions))
            rewards = info['agents_rewards']
            states_ = [FlattenObs(nextobs_i) for nextobs_i in next_obs]
            global_s_ = get_global_state(next_obs) #qmix
            qmix_step_exp = []
            sort_index = sorted(range(AGENTS_NUM),key=lambda i: states[i][1])
            for i in sort_index:
                qmix_step_exp.append([states[i].tolist(), actions[i], rewards[i], states_[i].tolist(), bool(info['agents_dones'][i]), global_s, global_s_])
            qmix_model.push_experience(qmix_step_exp)
            states = states_
            qmix_model.train()
            total_reward += sum(rewards)
            if done:
                break
            env.render()
        print (f'Episode total reward:{total_reward:.2f}')
        writer.add_scalar("Reward/Episode", total_reward, episode)
        qmix_model.epsilon_change()
        if episode%5==0:
            qmix_model.update_target_model()
            print (f'Epsilon change to {qmix_model.epsilon:.2f}')
            qmix_model.save_model('qmix.pth')
            print (f'Replay buffer size:{len(qmix_model.replay_buffer)}')
            
    writer.close() 
    env.close() 