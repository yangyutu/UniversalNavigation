
from Agents.DDPG.DDPG import DDPGAgent

from utils.netInit import xavier_init
import json
from torch import optim
from copy import deepcopy

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        self.linear3_1 = nn.Linear(hidden_size, 1)
        self.linear3_2 = nn.Linear(hidden_size, 1)
        self.apply(xavier_init)
        self.noise = OUNoise(output_size, seed = 1, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.05, decay_period=10000)
        self.noise.reset()
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

    def select_action(self, state, noiseFlag = False):
        if noiseFlag:
            action = self.forward(state)
            action += torch.tensor(self.noise.get_noise(), dtype=torch.float32, device=config['device']).unsqueeze(0)
        return self.forward(state)

configName = 'config.json'
with open(configName,'r') as f:
    config = json.load(f)

env = ActiveParticleEnv('config.json',1)

N_S = 2
N_A = env.nbActions

netParameter = dict()
netParameter['n_feature'] = N_S
netParameter['n_hidden'] = 128
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


checkpoint = torch.load('Log/Finalepoch20000_checkpoint.pt')
agent.actorNet.load_state_dict(checkpoint['actorNet_state_dict'])



plotPolicyFlag = True
N = 100
if plotPolicyFlag:
    phi = 0.0

    xSet = np.linspace(-10,10,N)
    ySet = np.linspace(-10,10,N)
    policyX = np.zeros((N, N))
    policyY = np.zeros((N, N))
    value = np.zeros((N, N))
    for i, x in enumerate(xSet):
        for j, y in enumerate(ySet):
            # x, y is the target position, (0, 0, 0) is the particle configuration
            distance = np.array([x, y])
            dx = distance[0] * math.cos(phi) + distance[1] * math.sin(phi)
            dy = -distance[0] * math.sin(phi) + distance[1] * math.cos(phi)
            angle = math.atan2(dy, dx)
            if math.sqrt(dx ** 2 + dy ** 2) > env.targetClipLength:
                dx = env.targetClipLength * math.cos(angle)
                dy = env.targetClipLength * math.sin(angle)

            state = torch.tensor([dx / agent.env.distanceScale, dy / agent.env.distanceScale], dtype=torch.float32, device = config['device']).unsqueeze(0)
            action = agent.actorNet.select_action(state, noiseFlag = False)
            value[i, j] = agent.criticNet.forward(state, action).item()
            action = action.cpu().detach().numpy()
            policyX[i, j] = action[0, 0]
            policyY[i, j] = action[0, 1]

    np.savetxt('StabilizerPolicyXTest.txt', policyX, fmt='%+.3f')
    np.savetxt('StabilizerPolicyYTest.txt', policyY, fmt='%+.3f')
    np.savetxt('StabilizerValueTest.txt',value, fmt='%+.3f')



config['dynamicInitialStateFlag'] = False
config['dynamicTargetFlag'] = False
config['currentState'] = [15, 15, 0]
config['targetState'] = [10, 15]
config['filetag'] = 'testLong'
config['trajOutputInterval'] = 1
config['trajOutputFlag'] = True
config['customExploreFlag'] = False

with open('config_test.json', 'w') as f:
    json.dump(config, f)

agent.env = ActiveParticleEnv('config_test.json',1)

delta = np.array([[15, 0], [15, 15], [15, -15], [-15, 0], [-15, -15], [-15, 15], [0, -15], [0, 15]])
delta = delta / 1.5
targets = delta + config['currentState'][:2]


nTargets = len(targets)
nTraj = 1
endStep = 100

for j in range(nTargets):
    recorder = []

    for i in range(nTraj):
        print(i)
        agent.env.config['targetState'] = targets[j]
        state = agent.env.reset()

        done = False
        rewardSum = 0
        stepCount = 0
        info = [i, stepCount] + agent.env.currentState.tolist() + agent.env.targetState.tolist() + [0.0 for _ in range(N_A)]
        recorder.append(info)
        for stepCount in range(endStep):
            action = agent.select_action(agent.actorNet, state, noiseFlag=False)
            nextState, reward, done, info = agent.env.step(action)
            info = [i, stepCount] + agent.env.currentState.tolist() + agent.env.targetState.tolist() + action.tolist()
            recorder.append(info)
            state = nextState
            rewardSum += reward
            if done:
                print("done in step count: {}".format(stepCount))
                #break
        print("reward sum = " + str(rewardSum))

    recorderNumpy = np.array(recorder)
    np.savetxt('testTraj_target_'+str(j)+'.txt', recorder, fmt='%.3f')
