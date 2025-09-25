import logging
import numpy as np
from functools import partial

from rl_agents.agents.common.factory import safe_deepcopy_env
from rl_agents.agents.tree_search.abstract import Node, AbstractTreeSearchAgent, AbstractPlanner
from rl_agents.agents.tree_search.olop import OLOP

logger = logging.getLogger(__name__)


class MCTSAgent(AbstractTreeSearchAgent):
    """
        An agent that uses Monte Carlo Tree Search to plan a sequence of action in an MDP.
    """
    def make_planner(self):
        prior_policy = MCTSAgent.policy_factory(self.config["prior_policy"])
        rollout_policy = MCTSAgent.policy_factory(self.config["rollout_policy"])
        return MCTS(self.env, prior_policy, rollout_policy, self.config)

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            "budget": 100,
            "horizon": None,
            "prior_policy": {"type": "random_available"},
            "rollout_policy": {"type": "random_available"},
            "env_preprocessors": []
         })
        return config

    @staticmethod
    def policy_factory(policy_config):
        if policy_config["type"] == "random":
            return MCTSAgent.random_policy
        elif policy_config["type"] == "random_available":
            return MCTSAgent.random_available_policy
        elif policy_config["type"] == "preference":
            return partial(MCTSAgent.preference_policy,
                           action_index=policy_config["action"],
                           ratio=policy_config["ratio"])
        else:
            raise ValueError("Unknown policy type")

    @staticmethod
    def random_policy(state, observation):
        """
            Choose actions from a uniform distribution.

        :param state: the environment state
        :param observation: the corresponding observation
        :return: a tuple containing the actions and their probabilities
        """
        actions = np.arange(state.action_space.n)
        probabilities = np.ones((len(actions))) / len(actions)
        return actions, probabilities

    @staticmethod
    def random_available_policy(state, observation):
        """
            Choose actions from a uniform distribution over currently available actions only.

        :param state: the environment state
        :param observation: the corresponding observation
        :return: a tuple containing the actions and their probabilities
        """
        if hasattr(state, 'get_available_actions'):
            available_actions = state.get_available_actions()
        else:
            available_actions = np.arange(state.action_space.n)
        probabilities = np.ones((len(available_actions))) / len(available_actions)
        return available_actions, probabilities

    @staticmethod
    def preference_policy(state, observation, action_index, ratio=2):
        """
            Choose actions with a distribution over currently available actions that favors a preferred action.

            The preferred action probability is higher than others with a given ratio, and the distribution is uniform
            over the non-preferred available actions.
        :param state: the environment state
        :param observation: the corresponding observation
        :param action_index: the label of the preferred action
        :param ratio: the ratio between the preferred action probability and the other available actions probabilities
        :return: a tuple containing the actions and their probabilities
        """
        if hasattr(state, 'get_available_actions'):
            available_actions = state.get_available_actions()
        else:
            available_actions = np.arange(state.action_space.n)
        for i in range(len(available_actions)):
            if available_actions[i] == action_index:
                probabilities = np.ones((len(available_actions))) / (len(available_actions) - 1 + ratio)
                probabilities[i] *= ratio
                return available_actions, probabilities
        return MCTSAgent.random_available_policy(state, observation)


class MCTS(AbstractPlanner):
    """
       An implementation of Monte-Carlo Tree Search, with Upper Confidence Tree exploration.
    """
    def __init__(self, env, prior_policy, rollout_policy, config=None):
        """
            New MCTS instance.

        :param config: the mcts configuration. Use default if None.
        :param prior_policy: the prior policy used when expanding and selecting nodes
        :param rollout_policy: the rollout policy used to estimate the value of a leaf node
        """
        super().__init__(config)
        self.env = env
        self.prior_policy = prior_policy
        self.rollout_policy = rollout_policy
        if not self.config["horizon"]:
            self.config["episodes"], self.config["horizon"] = \
                OLOP.allocation(self.config["budget"], self.config["gamma"])

    @classmethod
    def default_config(cls):
        cfg = super(MCTS, cls).default_config()
        cfg.update({
            "temperature": 2 / (1 - cfg["gamma"]),
            "closed_loop": False
        })
        return cfg

    def reset(self):
        self.root = MCTSNode(parent=None, planner=self)

    def run(self, state, observation):
        """
            Run an iteration of Monte-Carlo Tree Search, starting from a given state

        :param state: the initial environment state
        :param observation: the corresponding observation
        """
        node = self.root  #首先，初始化当前节点为根节点（即起始状态）。
        total_reward = 0  #初始化累计奖励为0。
        depth = 0   #初始化深度为0，表示从根节点开始。
        terminal = False   #表示当前状态是否是终止状态（如达到目标或发生碰撞等）。
        while depth < self.config['horizon'] and node.children and not terminal: #条件：depth < self.config['horizon'] 表示搜索的最大深度，node.children 判断当前节点是否有子节点，not terminal 判断是否达到终止状态。
            action = node.sampling_rule(temperature=self.config['temperature']) #选择（Selection）阶段：
            observation, reward, terminal, truncated, _ = self.step(state, action) #执行选中的动作，更新环境状态，得到奖励、是否终止和是否被截断等信息。
            total_reward += self.config["gamma"] ** depth * reward  #使用折扣因子（gamma）对奖励进行加权并累计到 total_reward 中。gamma 使得后期的奖励对当前决策的影响逐渐减少。
            node_observation = observation if self.config["closed_loop"] else None
            node = node.get_child(action, observation=node_observation)   #根据当前选择的动作，进入下一个子节点。
            depth += 1 #每次执行一个动作后，深度增加。

        if not node.children \
                and depth < self.config['horizon'] \
                and (not terminal or node == self.root):       #扩展（Expansion）阶段： 当达到叶子节点（即 node.children 为空）并且深度尚未超过最大深度时，扩展节点。
            node.expand(self.prior_policy(state, observation))  #调用 expand 方法来扩展当前节点，生成新的子节点。prior_policy 用于决定如何扩展节点。

        if not terminal:      #模拟（Simulation）阶段：如果当前路径没有终止（即不是叶子节点或目标节点），则进入模拟阶段。在此阶段，evaluate() 会运行模拟策略（rollout policy），进行多步随机选择来估计当前路径的未来回报。
            total_reward = self.evaluate(state, observation, total_reward, depth=depth)
        node.update_branch(total_reward)  #回溯（Backpropagation）阶段：在模拟完成后，通过回溯更新路径上所有节点的统计信息（访问次数、价值等）。这会将模拟结果（total_reward）传回路径上的所有节点，更新它们的值（value）。

    def evaluate(self, state, observation, total_reward=0, depth=0):
        """
            Run the rollout policy to yield a sample of the value of being in a given state.

        :param state: the leaf state.
        :param observation: the corresponding observation.
        :param total_reward: the initial total reward accumulated until now
        :param depth: the initial simulation depth
        :return: the total reward of the rollout trajectory
        """
        for h in range(depth, self.config["horizon"]): #从当前的 depth（深度）开始，到预设的最大深度。 horizon 是模拟的最大步数，限制了模拟的时间跨度。
            actions, probabilities = self.rollout_policy(state, observation) #根据当前状态 state 和观察 observation，使用 rollout policy（即模拟策略）来选择可能的动作。
            action = self.np_random.choice(actions, 1, p=np.array(probabilities))[0] #根据 probabilities 中的概率分布随机选择一个动作。
            observation, reward, terminal, truncated, _ = self.step(state, action) #执行所选择的动作 action，更新环境的状态并返回新状态的相关信息。
            total_reward += self.config["gamma"] ** h * reward #将当前获得的奖励（乘以折扣因子）加到累计奖励中，更新 total_reward
            if np.all(terminal) or np.all(truncated):  #检查是否达到终止状态或截断条件。
                break
        return total_reward

    def plan(self, state, observation):
        for i in range(self.config['episodes']):
            if (i+1) % 10 == 0:
                logger.debug('{} / {}'.format(i+1, self.config['episodes']))
            self.run(safe_deepcopy_env(state), observation)
        return self.get_plan()

    def step_planner(self, action):
        if self.config["step_strategy"] == "prior":
            self.step_by_prior(action)
        else:
            super().step_planner(action)

    def step_by_prior(self, action):
        """
            Replace the MCTS tree by its subtree corresponding to the chosen action, but also convert the visit counts
            to prior probabilities and before resetting them.

        :param action: a chosen action from the root node
        """
        self.step_by_subtree(action)
        self.root.convert_visits_to_prior_in_branch()


class MCTSNode(Node):
    K = 1.0
    """ The value function first-order filter gain"""

    def __init__(self, parent, planner, prior=1):
        super(MCTSNode, self).__init__(parent, planner)
        self.value = 0
        self.prior = prior

    def selection_rule(self):
        if not self.children:
            return None
        # Tie best counts by best value
        actions = list(self.children.keys()) ## 获取所有可用的动作（即子节点）
        counts = Node.all_argmax([self.children[a].count for a in actions])
        return actions[max(counts, key=(lambda i: self.children[actions[i]].get_value()))]

    def sampling_rule(self, temperature=None):
        """
            Select an action from the node.
            - if exploration is wanted with some temperature, follow the selection strategy.
            - else, select the action with maximum visit count

        :param temperature: the exploration parameter, positive or zero
        :return: the selected action
        """
        if self.children:
            actions = list(self.children.keys())   # 获取所有可用的动作（即子节点）
            # Randomly tie best candidates with respect to selection strategy
            indexes = [self.children[a].selection_strategy(temperature) for a in actions] # 根据选择策略和温度计算每个动作的得分
            return actions[self.random_argmax(indexes)] # 返回得分最高的动作
        else:
            return None

    def expand(self, actions_distribution):  #扩展节点，在每次遇到没有子节点的节点时，扩展所有可能的后续节点。
        """
            Expand a leaf node by creating a new child for each available action.

        :param actions_distribution: the list of available actions and their prior probabilities
        """
        actions, probabilities = actions_distribution
        for i in range(len(actions)):
            if actions[i] not in self.children:
                self.children[actions[i]] = type(self)(self, self.planner, probabilities[i])

    def update(self, total_reward):
        """
            Update the visit count and value of this node, given a sample of total reward.

        :param total_reward: the total reward obtained through a trajectory passing by this node
        """
        self.count += 1
        self.value += self.K / self.count * (total_reward - self.value)   #模拟完之后，用当前节点到最终结束的一整条轨迹的total_reward值更新当前节点的value值

    def update_branch(self, total_reward):
        """
            Update the whole branch from this node to the root with the total reward of the corresponding trajectory.

        :param total_reward: the total reward obtained through a trajectory passing by this node
        """
        self.update(total_reward)
        if self.parent:
            self.parent.update_branch(total_reward)

    def get_child(self, action, observation=None):
        child = self.children[action]
        if observation is not None:
            if str(observation) not in child.children:
                child.children[str(observation)] = MCTSNode(parent=child, planner=self.planner, prior=0)
            child = child.children[str(observation)]
        return child

    def selection_strategy(self, temperature):
        """
            Select an action according to its value, prior probability and visit count.

        :param temperature: the exploration parameter, positive or zero.
        :return: the selected action with maximum value and exploration bonus.
        """
        if not self.parent:
            return self.get_value()

        # return self.value + temperature * self.prior * np.sqrt(np.log(self.parent.count) / self.count)
        return self.get_value() + temperature * len(self.parent.children) * self.prior/(self.count+1)

    def convert_visits_to_prior_in_branch(self, regularization=0.5):
        """
            For any node in the subtree, convert the distribution of all children visit counts to prior
            probabilities, and reset the visit counts.

        :param regularization: in [0, 1], used to add some probability mass to all children.
                               when 0, the prior is a Boltzmann distribution of visit counts
                               when 1, the prior is a uniform distribution
        """
        self.count = 0
        total_count = sum([(child.count+1) for child in self.children.values()])
        for child in self.children.values():
            child.prior = (1 - regularization)*(child.count+1)/total_count + regularization/len(self.children)
            child.convert_visits_to_prior_in_branch()

    def get_value(self):
        return self.value

