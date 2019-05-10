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
        


        
