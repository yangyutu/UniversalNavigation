

from Agents.DDPG.DDPG import DDPGAgent
from Env.CustomEnv.StablizerOneD import StablizerOneDContinuous
from utils.netInit import xavier_init
import json
from torch import optim
from copy import deepcopy
from Env.CustomEnv.StablizerOneD import StablizerOneD
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch
from utils.OUNoise import OUNoise
from activeParticleEnv import ActiveParticleEnv

import math
torch.manual_seed(1)


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        self.apply(xavier_init)
    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        value = self.linear3(x)

        return value


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, config):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3_1 = nn.Linear(hidden_size, 1)
        self.linear3_2 = nn.Linear(hidden_size, 1)
        self.apply(xavier_init)
        self.noise = OUNoise(output_size, seed = 1, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.05, decay_period=10000)
        self.noise.reset()

        self.config = config
        self.stepCount = 0
    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        action0 = torch.sigmoid(self.linear3_1(x))
        action1 = torch.tanh(self.linear3_2(x))
        action = torch.cat([action0, action1], dim=1)
        return action

    def getCustomAction(self):

        if self.config['particleType'] == 'FULLCONTROL':
            choice = np.random.randint(0, 3)
            if choice == 0:
                action = np.array([1, 0])
            elif choice == 1:
                action = np.array([1, -1])
            elif choice == 2:
                action = np.array([1, 1])
        elif self.config['particleType'] == 'VANILLASP':
            action = np.array([1])
        elif self.config['particleType'] == 'CIRCLER':
            action = np.array([1])
        elif self.config['particleType'] == 'SLIDER':
            choice = np.random.randint(0, 3)
            if choice == 0:
                action = np.array([1])
            elif choice == 1:
                action = np.array([0])
            elif choice == 2:
                action = np.array([-1])

        return torch.tensor(action, dtype=torch.float32, device=self.config['device']).unsqueeze(0)


    # def select_action(self, state, noiseFlag = False):
    #     self.stepCount += 1
    #     if self.config['customExploreFlag'] and self.stepCount <= self.config['customExploreSteps']:
    #         action = self.getCustomAction()
    #         return action
    #     else:
    #         if noiseFlag:
    #             action = self.forward(state)
    #             action += torch.tensor(self.noise.get_noise(), dtype=torch.float32, device=self.config['device']).unsqueeze(0)
    #     return self.forward(state)

    def select_action(self, state, noiseFlag = False):
        if noiseFlag:
            action = self.forward(state)
            action += torch.tensor(self.noise.get_noise(), dtype=torch.float32, device=config['device']).unsqueeze(0)
            action = torch.clamp(action, -1, 1)
            if action[0][0] < 0.0:
                action[0][0] = 0.0
            return action
        return self.forward(state)

configName = 'config.json'
with open(configName,'r') as f:
    config = json.load(f)

env = ActiveParticleEnv('config.json',1)

N_S = 2
N_A = env.nbActions

netParameter = dict()
netParameter['n_feature'] = N_S
netParameter['n_hidden'] = 100
netParameter['n_output'] = N_A

actorNet = Actor(netParameter['n_feature'],
                                    netParameter['n_hidden'],
                                    netParameter['n_output'], config)

actorTargetNet = deepcopy(actorNet)

criticNet = Critic(netParameter['n_feature'] + N_A,
                                    netParameter['n_hidden'])

criticTargetNet = deepcopy(criticNet)

actorOptimizer = optim.Adam(actorNet.parameters(), lr=config['actorLearningRate'])
criticOptimizer = optim.Adam(criticNet.parameters(), lr=config['criticLearningRate'])

actorNets = {'actor': actorNet, 'target': actorTargetNet}
criticNets = {'critic': criticNet, 'target': criticTargetNet}
optimizers = {'actor': actorOptimizer, 'critic':criticOptimizer}
agent = DDPGAgent(config, actorNets, criticNets, env, optimizers, torch.nn.MSELoss(reduction='mean'), N_A)


plotPolicyFlag = False
N = 100
if plotPolicyFlag:
    phi = 0.0

    xSet = np.linspace(-10,10,N)
    ySet = np.linspace(-10,10,N)
    policy = np.zeros((N, N))

    value = np.zeros((N, N))
    for i, x in enumerate(xSet):
        for j, y in enumerate(ySet):
            # x, y is the target position, (0, 0, 0) is the particle configuration
            distance = np.array([x, y])
            dx = distance[0] * math.cos(phi) + distance[1] * math.sin(phi)
            dy = distance[0] * math.sin(phi) - distance[1] * math.cos(phi)
            angle = math.atan2(dy, dx)
            if math.sqrt(dx ** 2 + dy ** 2) > env.targetClipLength:
                dx = env.targetClipLength * math.cos(angle)
                dy = env.targetClipLength * math.sin(angle)

            state = torch.tensor([dx, dy], dtype=torch.float32, device = config['device']).unsqueeze(0)
            action = agent.actorNet.select_action(state, noiseFlag = False)
            value[i, j] = agent.criticNet.forward(state, action).item()
            action = action.cpu().detach().numpy()
            policy[i, j] = action

    np.savetxt('StabilizerPolicyBeforeTrain.txt', policy, fmt='%+.3f')
    np.savetxt('StabilizerValueBeforeTrain.txt',value, fmt='%+.3f')

agent.train()



def customPolicy(state):
    x = state[0]
    # move towards negative
    if x > 0.1:
        action = 2
    # move towards positive
    elif x < -0.1:
        action = 1
    # do not move
    else:
        action = 0
    return action
# storeMemory = ReplayMemory(100000)
# agent.perform_on_policy(100, customPolicy, storeMemory)
# storeMemory.write_to_text('performPolicyMemory.txt')
# transitions = storeMemory.fetch_all_random()

if plotPolicyFlag:
    phi = 0.0

    xSet = np.linspace(-10,10,N)
    ySet = np.linspace(-10,10,N)
    policy = np.zeros((N, N))

    value = np.zeros((N, N))
    for i, x in enumerate(xSet):
        for j, y in enumerate(ySet):
            # x, y is the target position, (0, 0, 0) is the particle configuration
            distance = np.array([x, y])
            dx = distance[0] * math.cos(phi) + distance[1] * math.sin(phi)
            dy = distance[0] * math.sin(phi) - distance[1] * math.cos(phi)
            angle = math.atan2(dy, dx)
            if math.sqrt(dx ** 2 + dy ** 2) > env.targetClipLength:
                dx = env.targetClipLength * math.cos(angle)
                dy = env.targetClipLength * math.sin(angle)

            state = torch.tensor([dx, dy], dtype=torch.float32, device = config['device']).unsqueeze(0)
            action = agent.actorNet.select_action(state, noiseFlag = False)
            value[i, j] = agent.criticNet.forward(state, action).item()
            action = action.cpu().detach().numpy()
            policy[i, j] = action

    np.savetxt('StabilizerPolicyAfterTrain.txt', policy, fmt='%+.3f')
    np.savetxt('StabilizerValueAfterTrain.txt',value, fmt='%+.3f')