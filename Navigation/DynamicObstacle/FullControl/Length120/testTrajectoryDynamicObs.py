
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


# Convolutional neural network (two convolutional layers)
class CriticConvNet(nn.Module):
    def __init__(self, inputWidth, num_hidden, num_action, n_channels):
        super(CriticConvNet, self).__init__()

        self.inputShape = (inputWidth, inputWidth)
        self.n_channels = n_channels
        self.layer1 = nn.Sequential(  # input shape (1, inputWdith, inputWdith)
            nn.Conv2d(self.n_channels,  # input channel
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

        self.fc0 = nn.Linear(2 + num_action, 128)
        self.fc1 = nn.Linear(self.featureSize() + 128, num_hidden)
        self.fc2 = nn.Linear(num_hidden, 1)
        self.apply(xavier_init)
    def forward(self, state, action):
        x = state['sensor']
        y = state['target']
        xout = self.layer1(x)
        xout = self.layer2(xout)
        xout = xout.reshape(xout.size(0), -1)
        # mask xout for test
        #xout.fill_(0)
        yout = F.relu(self.fc0(torch.cat((y, action), 1)))
        #actionOut = F.relu(self.fc0_action(action))
        out = torch.cat((xout, yout), 1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out

    def featureSize(self):
        return self.layer2(self.layer1(torch.zeros(1, self.n_channels, *self.inputShape))).view(1, -1).size(1)

# Convolutional neural network (two convolutional layers)
class ActorConvNet(nn.Module):
    def __init__(self, inputWidth, num_hidden, num_action, n_channels):
        super(ActorConvNet, self).__init__()
        self.n_channels = n_channels
        self.inputShape = (inputWidth, inputWidth)
        self.layer1 = nn.Sequential(  # input shape (1, inputWdith, inputWdith)
            nn.Conv2d(self.n_channels,  # input channel
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

        self.fc0 = nn.Linear(2, 128)
        self.fc1 = nn.Linear(self.featureSize() + 128, num_hidden)
        self.fc2_1 = nn.Linear(num_hidden, 1)
        self.fc2_2 = nn.Linear(num_hidden, 1)
        self.apply(xavier_init)
        self.noise = OUNoise(num_action, seed=1, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.05, decay_period=10000)
        self.noise.reset()

    def forward(self, state):
        x = state['sensor']
        y = state['target']
        xout = self.layer1(x)
        xout = self.layer2(xout)
        xout = xout.reshape(xout.size(0), -1)
        # mask xout for test
        #xout.fill_(0)
        yout = F.relu(self.fc0(y))
        out = torch.cat((xout, yout), 1)
        out = F.relu(self.fc1(out))
        action0 = torch.sigmoid(self.fc2_1(out))
        action1 = torch.tanh(self.fc2_2(out))
        action = torch.cat([action0, action1], dim=1)

        if action is None:
            print("action is None")
        return action

    def featureSize(self):
        return self.layer2(self.layer1(torch.zeros(1, self.n_channels, *self.inputShape))).view(1, -1).size(1)


    def select_action(self, state, noiseFlag = False):
        if noiseFlag:
            action = self.forward(state)
            action += torch.tensor(self.noise.get_noise(), dtype=torch.float32, device=config['device']).unsqueeze(0)
            action = torch.clamp(action, -1, 1)
            return action
        return self.forward(state)


def stateProcessor(state, device = 'cpu'):
    # given a list a dictions like { 'sensor': np.array, 'target': np.array}
    # we want to get a diction like {'sensor': list of torch tensor, 'target': list of torch tensor}
    nonFinalMask = torch.tensor(tuple(map(lambda s: s is not None, state)), device=device, dtype=torch.uint8)

    senorList = [item['sensor'] for item in state if item is not None]
    targetList = [item['target'] for item in state if item is not None]
    nonFinalState = {'sensor': torch.tensor(senorList, dtype=torch.float32, device=device),
              'target': torch.tensor(targetList, dtype=torch.float32, device=device)}
    return nonFinalState, nonFinalMask


configName = 'config.json'
with open(configName,'r') as f:
    config = json.load(f)

env = ActiveParticleEnv('config.json',1)

N_S = env.stateDim[0]
N_A = env.nbActions

netParameter = dict()
netParameter['n_feature'] = N_S
netParameter['n_hidden'] = 128
netParameter['n_output'] = N_A

actorNet = ActorConvNet(netParameter['n_feature'],
                                    netParameter['n_hidden'],
                                    netParameter['n_output'], config['n_channels'])

actorTargetNet = deepcopy(actorNet)

criticNet = CriticConvNet(netParameter['n_feature'] ,
                            netParameter['n_hidden'],
                        netParameter['n_output'], config['n_channels'])

criticTargetNet = deepcopy(criticNet)

actorOptimizer = optim.Adam(actorNet.parameters(), lr=config['actorLearningRate'])
criticOptimizer = optim.Adam(criticNet.parameters(), lr=config['criticLearningRate'])

actorNets = {'actor': actorNet, 'target': actorTargetNet}
criticNets = {'critic': criticNet, 'target': criticTargetNet}
optimizers = {'actor': actorOptimizer, 'critic':criticOptimizer}
agent = DDPGAgent(config, actorNets, criticNets, env, optimizers, torch.nn.MSELoss(reduction='mean'), N_A, stateProcessor=stateProcessor)


checkpoint = torch.load('Log/Epoch5000_checkpoint.pt')
agent.actorNet.load_state_dict(checkpoint['actorNet_state_dict'])


config['dynamicInitialStateFlag'] = False
config['dynamicTargetFlag'] = False
config['currentState'] = [75, 15, 0]
config['targetState'] = [115, 15]
config['filetag'] = 'test'
config['trajOutputFlag'] = True
config['trajOutputInterval'] = 10
config['trapFactor'] = 1.0
with open('config_test.json', 'w') as f:
    json.dump(config, f)

agent.env = ActiveParticleEnv('config_test.json',1)


nTraj = 5
endStep = 200


print("****************************direct transport benchmark ******************")

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
    print(info)
    print("reward sum = " + str(rewardSum))


print("****************************direct transport benchmark ******************")

for i in range(nTraj):
    print(i)
    state = agent.env.reset()
    agent.env.currentState[2] = random.random() * 2 * np.pi
    done = False
    rewardSum = 0
    stepCount = 0

    while not done:
        action = np.array([1.0, 0.0])
        nextState, reward, done, info = agent.env.step(action)
        stepCount += 1

        state = nextState
        rewardSum += reward
        if done:
            print("done in step count: {}".format(stepCount))

            break
        if stepCount > endStep:
            break
    print(info)
    print("reward sum = " + str(rewardSum))