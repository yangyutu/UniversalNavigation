#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 23:55:51 2019

@author: yangyutu123
"""

from activeParticleEnv import ActiveParticleEnv
import numpy as np
import math
import random

#import activeParticleSimulatorPython as model

env = ActiveParticleEnv('config_obs.json',1)

step = 20

state = env.reset()
print(state)

for i in range(step):
    state = env.currentState
    u = math.cos(state[2])*0.1
    v = math.sin(state[2])*0.1   
    w = 1.0
    nextState, reward, action, info = env.step(np.array([w]))
    print(nextState)
    print(info)
    #if i%2 == 0 and i < 10:
    #    env.step(100, np.array([u, v, 1.0]))
    #else:
    #    env.step(100, np.array([u, v, 0]))
        


env = ActiveParticleEnv('config_SP.json',1)

step = 20

state = env.reset()
print(state)

for i in range(step):
    state = env.currentState
    u = math.cos(state[2])*0.1
    v = math.sin(state[2])*0.1   
    w = 1.0
    nextState, reward, action, info = env.step(np.array([w]))
    print(nextState)
    print(info)


env = ActiveParticleEnv('config_FULL.json',1)

step = 20

state = env.reset()
print(state)

for i in range(step):
    state = env.currentState
    u = math.cos(state[2])*0.1
    v = math.sin(state[2])*0.1
    w = 1.0
    nextState, reward, action, info = env.step(np.array([w, w]))
    print(nextState)
    print(info)


env = ActiveParticleEnv('config_CIRCLE.json',1)

step = 100

state = env.reset()
print(state)

for i in range(step):
    state = env.currentState
    u = math.cos(state[2])*0.1
    v = math.sin(state[2])*0.1   
    w = 1.0
    nextState, reward, action, info = env.step(np.array([w]))
    print(nextState)
    print(info)

env = ActiveParticleEnv('config_SLIDER.json',1)

step = 100

state = env.reset()
print(state)

for i in range(step):
    state = env.currentState
    u = math.cos(state[2])*0.1
    v = math.sin(state[2])*0.1
    w = 1.0
    nextState, reward, action, info = env.step(np.array([w]))
    print(nextState)
    print(info)


env = ActiveParticleEnv('config_TWODIM.json',1)

step = 100

state = env.reset()
print(state)

for i in range(step):
    state = env.currentState
    u = np.random.rand() - 0.5
    v = np.random.rand() - 0.5
    w = 1.0
    nextState, reward, action, info = env.step(np.array([u, v]))
    print(nextState)
    print(info)
