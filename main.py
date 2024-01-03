import numpy as np
import itertools

# 网格世界的常量定义
GRID_ROWS = 3  # 网格行数
GRID_COLS = 4  # 网格列数
START_STATE = (0, 0)  # 起始状态坐标
BLOCKED_STATE = (1, 1)  # 障碍物坐标
TERMINAL_STATES = {(0, 3): 1, (1, 3): -1}  # 终点状态及其对应的奖励
ACTION_PROB = 0.8  # 主动作的执行概率
PERPENDICULAR_PROB = 0.1  # 与主动作垂直方向的执行概率
DEFAULT_REWARD = -0.04  # 默认奖励值
ACTIONS = ['上', '下', '左', '右']  # 定义的动作集


# 定义网格世界环境类
class GridworldEnv:
    def __init__(self, rows, cols, start_state, blocked_state, terminal_states, action_prob, perpendicular_prob, default_reward):
        self.rows = rows
        self.cols = cols
        self.start_state = start_state
        self.blocked_state = blocked_state
        self.terminal_states = terminal_states
        self.action_prob = action_prob
        self.perpendicular_prob = perpendicular_prob
        self.default_reward = default_reward
        # 创建所有可能的状态
        self.states = [(i, j) for i in range(self.rows) for j in range(self.cols) if (i, j) != self.blocked_state]

    # 判断是否为终点状态
    def is_terminal_state(self, state):
        return state in self.terminal_states

    # 根据当前状态和动作计算下一个状态和奖励
    def get_next_state_reward(self, state, action):
        if self.is_terminal_state(state):
            return state, 0

        next_state = list(state)
        if action == 'UP':
            next_state[0] = max(0, next_state[0] - 1)
        elif action == 'DOWN':
            next_state[0] = min(self.rows - 1, next_state[0] + 1)
        elif action == 'LEFT':
            next_state[1] = max(0, next_state[1] - 1)
        elif action == 'RIGHT':
            next_state[1] = min(self.cols - 1, next_state[1] + 1)
        next_state = tuple(next_state)

        if next_state == self.blocked_state:
            next_state = state

        reward = self.terminal_states.get(next_state, self.default_reward)

        return next_state, reward

    # 获取状态转移概率和奖励
    def get_transition_probs_rewards(self, state, action):
        transition_probs = {}

        direct_next_state, direct_reward = self.get_next_state_reward(state, action)
        transition_probs[direct_next_state] = (self.action_prob, direct_reward)

        if action in ['UP', 'DOWN']:
            left_action = 'LEFT'
            right_action = 'RIGHT'
        else:
            left_action = 'UP'
            right_action = 'DOWN'

        left_next_state, left_reward = self.get_next_state_reward(state, left_action)
        right_next_state, right_reward = self.get_next_state_reward(state, right_action)

        transition_probs[left_next_state] = (self.perpendicular_prob, left_reward)
        transition_probs[right_next_state] = (self.perpendicular_prob, right_reward)

        return transition_probs


# 价值迭代算法
def value_iteration(env, theta=0.0001, discount_factor=1.0):
    V = {state: 0 for state in env.states}

    while True:
        delta = 0
        for state in env.states:
            if env.is_terminal_state(state):
                continue

            v = V[state]
            V[state] = max(sum(prob * (reward + discount_factor * V[next_state])
                               for next_state, (prob, reward) in env.get_transition_probs_rewards(state, action).items())
                           for action in ACTIONS)

            delta = max(delta, abs(v - V[state]))

        if delta < theta:
            break

    policy = {state: None for state in env.states}
    for state in env.states:
        if env.is_terminal_state(state):
            continue

        best_action_value = float('-inf')
        for action in ACTIONS:
            action_value = sum(prob * (reward + discount_factor * V[next_state])
                               for next_state, (prob, reward) in env.get_transition_probs_rewards(state, action).items())
            if action_value > best_action_value:
                best_action_value = action_value
                policy[state] = action

    return policy, V


# 策略迭代算法
def policy_iteration(env, theta=0.0001, discount_factor=1.0):
    def one_step_lookahead(state, V, policy_action):
        return sum(prob * (reward + discount_factor * V[next_state])
                   for next_state, (prob, reward) in env.get_transition_probs_rewards(state, policy_action).items())

    policy = {state: np.random.choice(ACTIONS) for state in env.states}
    V = {state: 0 for state in env.states}

    while True:
        while True:
            delta = 0
            for state in env.states:
                v = V[state]
                V[state] = one_step_lookahead(state, V, policy[state])
                delta = max(delta, abs(v - V[state]))
            if delta < theta:
                break

        policy_stable = True
        for state in env.states:
            if env.is_terminal_state(state):
                continue

            old_action = policy[state]
            action_values = {action: one_step_lookahead(state, V, action) for action in ACTIONS}
            best_action = max(action_values, key=action_values.get)

            if old_action != best_action:
                policy_stable = False
            policy[state] = best_action

        if policy_stable:
            break

    return policy, V


# 分析不同参数对最优策略的影响
def analyze_impact(env, reward_values, discount_factors, transition_probabilities):
    analysis_results = {}

    for reward, discount_factor, transition_probability in itertools.product(reward_values, discount_factors, transition_probabilities):
        env.default_reward = reward
        env.action_prob = transition_probability
        env.perpendicular_prob = (1 - transition_probability) / 2

        optimal_policy, _ = value_iteration(env, discount_factor=discount_factor)

        analysis_results[(reward, discount_factor, transition_probability)] = optimal_policy

    return analysis_results

# 初始化环境
env = GridworldEnv(GRID_ROWS, GRID_COLS, START_STATE, BLOCKED_STATE, TERMINAL_STATES, ACTION_PROB, PERPENDICULAR_PROB, DEFAULT_REWARD)

# 测试环境
sample_transition_probs = env.get_transition_probs_rewards(START_STATE, 'UP')
print(sample_transition_probs)

# 执行价值迭代和策略迭代
optimal_policy_vi, V_vi = value_iteration(env)
optimal_policy_pi, V_pi = policy_iteration(env)

print((optimal_policy_vi, V_vi), (optimal_policy_pi, V_pi))

# 定义不同参数
reward_values = [-0.04, -0.02, -0.01]
discount_factors = [0.9, 0.95, 1.0]
transition_probabilities = [0.7, 0.8, 0.9]

# 分析影响
impact_analysis_results = analyze_impact(env, reward_values, discount_factors, transition_probabilities)

print(impact_analysis_results)
