import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from collections import deque

BATCH_SIZE = 256
GAMMA = 0.99 #折现率
LEARNING_RATE = 0.001 #学习率
EPSILON_START = 1.0 # 初始探索概率
EPSILON_END = 0.05 # 最终探索概率
EPSILON_DECAY = 0.997 #epsilon 衰减率

def FlattenObs(obs):  # (5,5) -> (25)
    obs = np.array(obs)
    return obs.flatten()




class DQNModule(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNModule, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, 1024)
        self.fc6 = nn.Linear(1024, 1024)
        self.fc7 = nn.Linear(1024, 1024)
        self.fc8 = nn.Linear(1024, 256)
        self.fc9 = nn.Linear(256, 64)
        self.fc10 = nn.Linear(64, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = torch.relu(self.fc8(x))
        x = torch.relu(self.fc9(x))
        return self.fc10(x)


class DQNModuleMini(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNModuleMini, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.fc5(x)

class ReplayBuffer:
    def __init__(self, capacity=10000, model_name='dqn'):
        self.buffer = deque(maxlen=capacity)
        self.model_name = model_name

    #def push(self, state, action, reward, next_state, done):
    #    self.buffer.append((state, action, reward, next_state, done))
    def push(self,exp): #exp array for dqn: (state, action, reward, next_state, done)
        self.buffer.append(exp)

    def sample(self, batch_size):
        if self.model_name == 'dqn':
            state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
            return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)
        elif self.model_name == 'vdn':
            batch_samples = random.sample(self.buffer, batch_size)  # 返回 [ [智能体1数据, 智能体2数据, ...], ... ]
            # 按智能体拆分数据
            states, actions, rewards, next_states, dones = [], [], [], [], []
            for sample in batch_samples:  # 遍历每个 batch 样本
                agent_data = list(zip(*sample))  # 拆分成 (states, actions, rewards, next_states, dones)
                states.append(agent_data[0])      # 所有智能体的 state
                actions.append(agent_data[1])     # 所有智能体的 action
                rewards.append(agent_data[2])     # 所有智能体的 reward
                next_states.append(agent_data[3]) # 所有智能体的 next_state
                dones.append(agent_data[4])       # 所有智能体的 done
            return (
                    np.array(states),       # [batch_size, num_agents, state_dim]
                    np.array(actions),      # [batch_size, num_agents]
                    np.array(rewards),      # [batch_size, num_agents]
                    np.array(next_states),  # [batch_size, num_agents, state_dim]
                    np.array(dones)         # [batch_size, num_agents]
                )
        elif self.model_name == 'qmix':
            batch_samples = random.sample(self.buffer, batch_size)  # 返回 [ [智能体1数据, 智能体2数据, ...], ... ]
            # 按智能体拆分数据
            states, actions, rewards, next_states, dones, global_states, next_global_states = [], [], [], [], [], [], []
            for sample in batch_samples:  # 遍历每个 batch 样本
                agent_data = list(zip(*sample))  # 拆分成 (states, actions, rewards, next_states, dones)
                states.append(agent_data[0])      # 所有智能体的 state
                actions.append(agent_data[1])     # 所有智能体的 action
                rewards.append(agent_data[2])     # 所有智能体的 reward
                next_states.append(agent_data[3]) # 所有智能体的 next_state
                dones.append(agent_data[4])       # 所有智能体的 done
                global_states.append(agent_data[5])  # 全局状态
                next_global_states.append(agent_data[6])  # 下一全局状态
            # 转换为 numpy array（形状：[batch_size, num_agents, ...]）
            return (
                np.array(states),       # [batch_size, num_agents, state_dim]
                np.array(actions),      # [batch_size, num_agents]
                np.array(rewards),      # [batch_size, num_agents]
                np.array(next_states),  # [batch_size, num_agents, state_dim]
                np.array(dones),         # [batch_size, num_agents]
                np.array(global_states),
                np.array(next_global_states)
            )

    def __len__(self):
        return len(self.buffer)


class DQN():
    def __init__(self,state_size,action_size,model_file_path='',use_epsilon=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size
        self.model = DQNModule(state_size, action_size).to(self.device)
        self.target_model = DQNModule(state_size, action_size).to(self.device)
        #self.model = DQNModuleMini(state_size, action_size).to(self.device)
        #self.target_model = DQNModuleMini(state_size, action_size).to(self.device)
        self.update_target_model()
        self.replay_buffer = ReplayBuffer(capacity=1000000,model_name='dqn')
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.USE_EPSILON = use_epsilon
        self.epsilon = EPSILON_START
        
        if model_file_path != '' and os.path.exists(model_file_path):
            self.load_model(model_file_path)
    
    def epsilon_change(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def push_experience(self,state, action, reward, state_, done):
        if self.USE_EPSILON:
            self.replay_buffer.push((state, action, reward, state_, done))
        
    def save_model(self, file_path):
        if not self.USE_EPSILON:
            return
        torch.save(self.target_model.state_dict(), file_path)
        print(f"模型已保存到 {file_path}")

    def load_model(self, file_path):
        #print ('************',self.device)
        self.model.load_state_dict(torch.load(file_path,map_location=self.device)) #markchalse 0717
        self.target_model.load_state_dict(torch.load(file_path,map_location=self.device))
        print(f"模型已从 {file_path} 加载")
        
    def get_action(self,state):
        if self.USE_EPSILON and random.random() < self.epsilon: # 随机选择一个动作（探索）
            return random.randint(0, self.action_size - 1)
        # 选择模型认为最优的动作（利用）
        with torch.no_grad():  # 禁用梯度计算
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # 扩展为 2D 张量
            q_values = self.model(state_tensor)
            return q_values.argmax().item()
    
    def train(self):
        if not self.USE_EPSILON:
            return 
        if len(self.replay_buffer) < BATCH_SIZE:
            print ('replay buffer wait...')
            return
        state, action, reward, next_state, done = self.replay_buffer.sample(BATCH_SIZE)
        # 将数据转换为张量并移动到 GPU
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        
        current_q = self.model(state).gather(1, action.unsqueeze(1))
        next_action = self.model(next_state).max(1)[1]  # Double DQN 主网络选择动作
        next_q = self.target_model(next_state).gather(1, next_action.unsqueeze(1)).squeeze(1).detach()
        target_q = reward + (1 - done) * GAMMA * next_q
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
class VDN(DQN):
    def __init__(self, state_size, action_size, model_file_path='', use_epsilon=True):
        super().__init__(state_size, action_size, model_file_path, use_epsilon)
        self.replay_buffer = ReplayBuffer(capacity=1000000,model_name='vdn')
    
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.9) # 添加学习率调度器
    def push_experience(self,exp):
        if self.USE_EPSILON:
            self.replay_buffer.push(exp)
            
    def train(self):
        if not self.USE_EPSILON:
            return 
        if len(self.replay_buffer) < BATCH_SIZE:
            print("Replay buffer not enough samples...")
            return
        # 采样数据（形状：[batch_size, num_agents, ...]）
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)  
        
        states = torch.FloatTensor(states).to(self.device)          # [batch_size, num_agents, state_dim]
        actions = torch.LongTensor(actions).to(self.device)         # [batch_size, num_agents]
        rewards = torch.FloatTensor(rewards).to(self.device)        # [batch_size, num_agents]
        next_states = torch.FloatTensor(next_states).to(self.device)# [batch_size, num_agents, state_dim]
        dones = torch.FloatTensor(dones).to(self.device)            # [batch_size, num_agents]
        batch_size, num_agents = actions.shape
        # 计算当前 Q 值（每个智能体的 Q 值）
        current_qs = []
        for agent_id in range(num_agents):
            agent_states = states[:, agent_id, :]  # [batch_size, state_dim]
            agent_actions = actions[:, agent_id]   # [batch_size]
            agent_q = self.model(agent_states).gather(1, agent_actions.unsqueeze(1))  # [batch_size, 1]
            current_qs.append(agent_q)
        current_q_total = torch.sum(torch.stack(current_qs, dim=0), dim=0)  # 求和得到全局 Q [batch_size, 1]
        # 计算目标 Q 值（Double DQN 风格）
        next_qs = []
        for agent_id in range(num_agents):
            agent_next_states = next_states[:, agent_id, :]  # [batch_size, state_dim]
            # 主网络选择动作
            next_actions = self.model(agent_next_states).max(1)[1]  # [batch_size]
            # 目标网络计算 Q 值
            next_q = self.target_model(agent_next_states).gather(1, next_actions.unsqueeze(1))  # [batch_size, 1]
            next_qs.append(next_q) 
        next_q_total = torch.sum(torch.stack(next_qs, dim=0), dim=0)  # 求和得到全局 next Q [batch_size, 1]
        next_q_total = next_q_total.squeeze(1).detach()  # [batch_size]
        # 计算团队奖励（所有智能体的奖励之和）
        team_rewards = torch.sum(rewards, dim=1)  # [batch_size]
        #print('team_rewards shape:',team_rewards.shape)
        #print ('current_q_total shape:',current_q_total.shape)
        #print ('next_q_total shape:',next_q_total.shape)
        team_dones = torch.any(dones.bool(), dim=1).float()  # 任意智能体 done 则整个 episode done
        #print ('team_dones shape:',team_dones.shape)
        # TD 目标
        target_q = team_rewards + (1 - team_dones) * GAMMA * next_q_total
        # 计算损失（MSE）
        loss = nn.SmoothL1Loss()(current_q_total.squeeze(), target_q) # 使用Huber loss替代MSE
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0) # 梯度裁剪
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
    

class HyperNetwork(nn.Module):
    """超网络，用于生成混合网络的权重"""
    def __init__(self, state_size, hyper_hidden_dim, mixing_output_dim):
        super(HyperNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hyper_hidden_dim)
        self.fc2 = nn.Linear(hyper_hidden_dim, mixing_output_dim)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        return self.fc2(x)

class MixingNetwork(nn.Module):
    """混合网络，将各智能体的Q值混合成联合Q值"""
    def __init__(self, num_agents, state_size, hyper_hidden_dim=16):
        super(MixingNetwork, self).__init__()
        self.num_agents = num_agents
        self.state_size = state_size
        self.hyper_hidden_dim = hyper_hidden_dim
        
        # 超网络用于生成混合网络的权重
        self.hyper_w1 = HyperNetwork(state_size, hyper_hidden_dim, num_agents * hyper_hidden_dim)
        self.hyper_w2 = HyperNetwork(state_size, hyper_hidden_dim, hyper_hidden_dim)
        
        # 超网络用于生成混合网络的偏置
        self.hyper_b1 = HyperNetwork(state_size, hyper_hidden_dim, hyper_hidden_dim)
        self.hyper_b2 = nn.Linear(state_size, 1)
        
    def forward(self, agent_qs, states): #agent_qs: [batch_size, num_agents] 各智能体的Q值  #states: [batch_size, state_size] 全局状态
        batch_size = agent_qs.size(0)
        # 第一层
        w1 = torch.abs(self.hyper_w1(states)).view(-1, self.num_agents, self.hyper_hidden_dim)
        b1 = self.hyper_b1(states).view(-1, 1, self.hyper_hidden_dim)
        # 确保单调性: w1和w2保持非负
        w1 = torch.abs(w1)
        # 计算第一层输出
        hidden = torch.bmm(agent_qs.unsqueeze(1), w1) + b1
        hidden = torch.relu(hidden)
        # 第二层
        w2 = torch.abs(self.hyper_w2(states)).view(-1, self.hyper_hidden_dim, 1)
        b2 = self.hyper_b2(states).view(-1, 1, 1)
        w2 = torch.abs(w2)
        # 计算最终Q值
        y = torch.bmm(hidden, w2) + b2
        return y.squeeze(2)

class QMIX(DQN):
    def __init__(self, state_size, action_size, model_file_path='', use_epsilon=True,agent_num=5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 混合网络
        self.mixing_network = MixingNetwork(num_agents=agent_num, state_size=(agent_num+1)*5).to(self.device)
        self.target_mixing_network = MixingNetwork(num_agents=agent_num, state_size=(agent_num+1)*5).to(self.device)
        super().__init__(state_size, action_size, model_file_path, use_epsilon)
        self.replay_buffer = ReplayBuffer(capacity=1000000,model_name='qmix')
        self.agent_num = agent_num
        
        self.mixer_optimizer = optim.Adam(self.mixing_network.parameters(), lr=LEARNING_RATE)
    
    def push_experience(self,exp):
        if self.USE_EPSILON:
            self.replay_buffer.push(exp)
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
    
    def load_model(self, file_path):
        state_dicts = torch.load(file_path,map_location=self.device)  #markchalse 20250717
        self.model.load_state_dict(state_dicts['agent_model'])
        self.target_model.load_state_dict(state_dicts['agent_model'])
        self.mixing_network.load_state_dict(state_dicts['mixing_network'])
        self.target_mixing_network.load_state_dict(state_dicts['mixing_network'])
        print(f"模型已从 {file_path} 加载")
    
    def save_model(self, file_path):
        if not self.USE_EPSILON:
            return
        state_dicts = {
            'agent_model': self.model.state_dict(),
            'mixing_network': self.mixing_network.state_dict()
        }
        torch.save(state_dicts, file_path)
        print(f"模型已保存到 {file_path}")

    def train(self):
        if not self.USE_EPSILON:
            return
        if len(self.replay_buffer) < BATCH_SIZE:
            print("Replay buffer not enough samples...")
            return

        # 采样数据（与之前相同）
        states, actions, rewards, next_states, dones, global_states, next_global_states = \
            self.replay_buffer.sample(BATCH_SIZE)
        
        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)          # [batch, num_agents, state_dim]
        actions = torch.LongTensor(actions).to(self.device)         # [batch, num_agents]
        rewards = torch.FloatTensor(rewards).to(self.device)        # [batch, num_agents]
        next_states = torch.FloatTensor(next_states).to(self.device)# [batch, num_agents, state_dim]
        dones = torch.FloatTensor(dones).to(self.device)            # [batch, num_agents]       
        global_states = torch.FloatTensor(global_states).to(self.device)  # [batch, global_state_dim]
        # 方案1：取第一个智能体的全局状态（如果全局状态对所有智能体相同）
        global_states = global_states[:, 0, :]  # [256,30]
        next_global_states = torch.FloatTensor(next_global_states).to(self.device)
        next_global_states = next_global_states[:, 0, :]  # [256,30]
        # 计算当前Q值（使用共享网络处理所有智能体的状态）
        batch_size, num_agents = actions.shape
        # 确保正确的reshape方式
        flat_states = states.reshape(-1, states.shape[-1])  # [batch*num_agents, state_dim]
        flat_qs = self.model(flat_states)                   # [batch*num_agents, action_dim]
        flat_actions = actions.reshape(-1, 1)               # [batch*num_agents, 1]
        current_qs = flat_qs.gather(1, flat_actions)        # [batch*num_agents, 1]
        current_qs = current_qs.reshape(batch_size, num_agents)  # [batch, num_agents]
        # 计算联合Q值
        current_joint_q = self.mixing_network(current_qs, global_states)  # [batch, 1]
        # 计算目标Q值（同样使用共享网络）
        flat_next_states = next_states.view(-1, next_states.shape[-1])
        next_qs = self.target_model(flat_next_states)  # [batch*num_agents, action_dim]
        next_actions = self.model(flat_next_states).max(1)[1]  # Double DQN
        next_qs = next_qs.gather(1, next_actions.unsqueeze(1)).view(batch_size, num_agents)
        # 计算目标联合Q值
        next_joint_q = self.target_mixing_network(next_qs, next_global_states)  # [batch, 1]
        # 计算团队奖励和TD目标（与之前相同）
        team_rewards = torch.sum(rewards, dim=1, keepdim=True)
        team_dones = torch.any(dones.bool(), dim=1, keepdim=True).float()
        target_q = team_rewards + (1 - team_dones) * GAMMA * next_joint_q
        # 计算损失
        loss = nn.MSELoss()(current_joint_q, target_q.detach())
        # 反向传播（现在只需更新一个Q网络）
        self.optimizer.zero_grad()
        self.mixer_optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪
        #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        #torch.nn.utils.clip_grad_norm_(self.mixing_network.parameters(), 10.0)
        # 更新参数
        self.optimizer.step()
        self.mixer_optimizer.step()