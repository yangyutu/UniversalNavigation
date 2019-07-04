

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
from activeParticleEnv import ActiveParticleEscapeEnv

import math
torch.manual_seed(1)


# Convolutional neural network (two convolutional layers)
class CriticConvNet(nn.Module):
    def __init__(self, inputWidth, num_hidden, num_action):
        super(CriticConvNet, self).__init__()

        self.inputShape = (inputWidth, inputWidth)
        # two channels, one for obstacle and one for hazard region
        self.layer1 = nn.Sequential(  # input shape (1, inputWdith, inputWdith)
            nn.Conv2d(2,  # input channel
                      32,  # output channel
                      kernel_size=2,  # filter size
                      stride=1,
                      padding=1),
            # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # inputWdith / 2

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # inputWdith / 2
        # add a fully connected layer
        # width = int(inputWidth / 4) + 1

        self.fc0 = nn.Linear(num_action, 128)
        self.fc1 = nn.Linear(self.featureSize() + 128, num_hidden)
        self.fc2 = nn.Linear(num_hidden, 1)
        self.apply(xavier_init)
    def forward(self, state, action):
        x = state
        xout = self.layer1(x)
        xout = self.layer2(xout)
        xout = xout.reshape(xout.size(0), -1)
        #actionOut = F.relu(self.fc0_action(action))
        yout = F.relu(self.fc0(action))
        out = torch.cat((xout, yout), 1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out

    def featureSize(self):
        return self.layer2(self.layer1(torch.zeros(1, 2, *self.inputShape))).view(1, -1).size(1)

# Convolutional neural network (two convolutional layers)
class ActorConvNet(nn.Module):
    def __init__(self, inputWidth, num_hidden, num_action):
        super(ActorConvNet, self).__init__()

        self.inputShape = (inputWidth, inputWidth)
        self.layer1 = nn.Sequential(  # input shape (1, inputWdith, inputWdith)
            nn.Conv2d(2,  # input channel
                      32,  # output channel
                      kernel_size=2,  # filter size
                      stride=1,
                      padding=1),
            # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # inputWdith / 2

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # inputWdith / 2
        # add a fully connected layer
        # width = int(inputWidth / 4) + 1

        self.fc1 = nn.Linear(self.featureSize(), num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_action)
        self.apply(xavier_init)
        self.noise = OUNoise(num_action, seed=1, mu=0.0, theta=0.15, max_sigma=0.5, min_sigma=0.1, decay_period=1000000)
        self.noise.reset()

    def forward(self, state):
        # two channels, one for obstacle and one for hazard region

        x = state
        xout = self.layer1(x)
        xout = self.layer2(xout)
        xout = xout.reshape(xout.size(0), -1)
        # mask xout for test
        out = F.relu(self.fc1(xout))
        action = torch.tanh(self.fc2(out))
        return action

    def featureSize(self):
        return self.layer2(self.layer1(torch.zeros(1, 2, *self.inputShape))).view(1, -1).size(1)

    def select_action(self, state, noiseFlag = False):
        if noiseFlag:
            action = self.forward(state)
            action += torch.tensor(self.noise.get_noise(), dtype=torch.float32, device=config['device']).unsqueeze(0)
            action = torch.clamp(action, -1, 1)
            return action
        return self.forward(state)

configName = 'config.json'
with open(configName,'r') as f:
    config = json.load(f)

env = ActiveParticleEscapeEnv('config.json',1)

N_S = env.stateDim[0]
N_A = env.nbActions

netParameter = dict()
netParameter['n_feature'] = N_S
netParameter['n_hidden'] = 128
netParameter['n_output'] = N_A

actorNet = ActorConvNet(netParameter['n_feature'],
                                    netParameter['n_hidden'],
                                    netParameter['n_output'])

actorTargetNet = deepcopy(actorNet)

criticNet = CriticConvNet(netParameter['n_feature'] ,
                            netParameter['n_hidden'],
                        netParameter['n_output'])

criticTargetNet = deepcopy(criticNet)

actorOptimizer = optim.Adam(actorNet.parameters(), lr=config['actorLearningRate'])
criticOptimizer = optim.Adam(criticNet.parameters(), lr=config['criticLearningRate'])

actorNets = {'actor': actorNet, 'target': actorTargetNet}
criticNets = {'critic': criticNet, 'target': criticTargetNet}
optimizers = {'actor': actorOptimizer, 'critic':criticOptimizer}
agent = DDPGAgent(config, actorNets, criticNets, env, optimizers, torch.nn.MSELoss(reduction='mean'), N_A)


plotPolicyFlag = True
N = 100
fileName = config['mapName']
mapMat = np.genfromtxt(fileName + '.txt')
fileName = config['hazardMapName']
hazardMat = np.genfromtxt(fileName + '.txt')

if plotPolicyFlag:
    phiIdx = 0
    phi = 0
    policyX = deepcopy(mapMat).astype(np.float)
    policyY = deepcopy(mapMat).astype(np.float)

    value = deepcopy(mapMat)
    for i in range(policyX.shape[0]):
        for j in range(policyX.shape[1]):
            if mapMat[i, j] > 0 or (mapMat[i, j] == 0 and hazardMat[i, j] == 0):
                policyX[i, j] = -2
                value[i, j] = -1
                policyY[i, j] = -2

            else:
                state = agent.env.getSensorInfoFromPos(np.array([i, j, phi]))
                stateTorch = torch.tensor([state], device = config['device'], dtype=torch.float)
                action = agent.actorNet.select_action(stateTorch, noiseFlag=False)
                value[i, j] = agent.criticNet.forward(stateTorch, action).item()
                action = action.cpu().detach().numpy()
                policyX[i, j] = action[0, 0]
                policyY[i, j] = action[0, 1]
    np.savetxt(config['mapName'] + 'PolicyXAnalysisBefore' + 'phiIdx' + str(phiIdx) + '.txt', policyX, fmt='%.3f', delimiter='\t')
    np.savetxt(config['mapName'] + 'PolicyYAnalysisBefore' + 'phiIdx' + str(phiIdx) + '.txt', policyY, fmt='%.3f',
               delimiter='\t')
    np.savetxt(config['mapName'] + 'ValueAnalysisBefore' + 'phiIdx' + str(phiIdx) + '.txt', value, fmt='%.3f', delimiter='\t')

agent.train()




if plotPolicyFlag:
    phiIdx = 0
    phi = 0
    policyX = deepcopy(mapMat).astype(np.float)
    policyY = deepcopy(mapMat).astype(np.float)

    value = deepcopy(mapMat)
    for i in range(policyX.shape[0]):
        for j in range(policyX.shape[1]):
            if mapMat[i, j] > 0 or (mapMat[i, j] == 0 and hazardMat[i, j] == 0):
                policyX[i, j] = -2
                value[i, j] = -1
                policyY[i, j] = -2

            else:
                state = agent.env.getSensorInfoFromPos(np.array([i, j, phi]))
                stateTorch = torch.tensor([state], device=config['device'], dtype=torch.float)
                action = agent.actorNet.select_action(stateTorch, noiseFlag=False)
                value[i, j] = agent.criticNet.forward(stateTorch, action).item()
                action = action.cpu().detach().numpy()
                policyX[i, j] = action[0, 0]
                policyY[i, j] = action[0, 1]
    np.savetxt(config['mapName'] + 'PolicyXAnalysisAfter' + 'phiIdx' + str(phiIdx) + '.txt', policyX, fmt='%.3f',
               delimiter='\t')
    np.savetxt(config['mapName'] + 'PolicyYAnalysisAfter' + 'phiIdx' + str(phiIdx) + '.txt', policyY, fmt='%.3f',
               delimiter='\t')
    np.savetxt(config['mapName'] + 'ValueAnalysisAfter' + 'phiIdx' + str(phiIdx) + '.txt', value, fmt='%.3f',
               delimiter='\t')