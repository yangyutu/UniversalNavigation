
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
import random
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
    def __init__(self, input_size, hidden_size, output_size, learning_rate=3e-4):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.apply(xavier_init)
        self.noise = OUNoise(output_size, seed = 1, mu=0.0, theta=0.5, max_sigma=0.05, min_sigma=0.001, decay_period=10000)
        self.noise.reset()
    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        action = torch.sigmoid(self.linear3(x))

        return action

    def select_action(self, state, noiseFlag = False):
        if noiseFlag:
            action = self.forward(state)
            action += torch.tensor(self.noise.get_noise(), dtype=torch.float32, device=config['device']).unsqueeze(0)
            action = torch.clamp(action, 0, 1)
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
                                    netParameter['n_output'])

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


checkpoint = torch.load('Log/Finalepoch1000_checkpoint.pt')
agent.actorNet.load_state_dict(checkpoint['actorNet_state_dict'])

config['dynamicInitialStateFlag'] = False
config['dynamicTargetFlag'] = False
config['currentState'] = [15, 15, 0]
config['targetState'] = [10, 15]
config['filetag'] = 'TrajTest/test'

with open('config_test.json', 'w') as f:
    json.dump(config, f)

agent.env = ActiveParticleEnv('config_test.json',1)


nTraj = 20
endStep = 500
recorder = []
for i in range(nTraj):
    print(i)
    state = agent.env.reset()
    agent.env.currentState[2] = random.random() * 2 * np.pi
    done = False
    rewardSum = 0
    stepCount = 0

    while not done:
        action = agent.select_action(agent.actorNet, state, noiseFlag=False)
        nextState, reward, done, info = agent.env.step(action)
        stepCount += 1

        state = nextState
        rewardSum += reward
        if done:
            print("done in step count: {}".format(stepCount))
            break
        if stepCount > endStep:
            break
    print("reward sum = " + str(rewardSum))
