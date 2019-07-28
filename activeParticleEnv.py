from ActiveParticleSimulatorPython import ActiveParticleSimulatorPython
import numpy as np
import random
import json
import os
from sklearn.metrics.pairwise import euclidean_distances
import math
import sys


class ActiveParticleEnv():
    def __init__(self, configName, randomSeed = 1):

        with open(configName) as f:
            self.config = json.load(f)
        self.randomSeed = randomSeed
        self.model = ActiveParticleSimulatorPython(configName, randomSeed)
        self.read_config()
        self.initilize()

        #self.padding = self.config['']

    def initilize(self):
        if not os.path.exists('Traj'):
            os.makedirs('Traj')
        # import parameter for vector env
        self.viewer = None
        self.steps_beyond_done = None
        self.stepCount = 0

        self.info = {}

        random.seed(self.randomSeed)
        np.random.seed(self.randomSeed)

        self.initObsMat()
        self.constructSensorArrayIndex()
        self.epiCount = -1

    def read_config(self):

        self.receptHalfWidth = self.config['receptHalfWidth']
        self.padding = self.config['obstacleMapPaddingWidth']
        self.receptWidth = 2 * self.receptHalfWidth + 1
        self.targetClipLength = 2 * self.receptHalfWidth
        self.stateDim = (self.receptWidth, self.receptWidth)

        self.sensorArrayWidth = (2*self.receptHalfWidth + 1)


        self.episodeEndStep = 500
        if 'episodeLength' in self.config:
            self.episodeEndStep = self.config['episodeLength']

        self.particleType = self.config['particleType']
        typeList = ['FULLCONTROL','VANILLASP','CIRCLER','SLIDER', 'TWODIM']
        if self.particleType not in typeList:
            sys.exit('particle type not right!')

        if self.particleType == 'FULLCONTROL':
            self.nbActions = 2
        elif self.particleType == 'VANILLASP':
            self.nbActions = 1
        elif self.particleType == 'CIRCLER':
            self.nbActions = 1
        elif self.particleType == 'SLIDER':
            self.nbActions = 1
        elif self.particleType == "TWODIM":
            self.nbActions = 2

        self.startThresh = 1
        self.endThresh = 1
        self.distanceThreshDecay = 10000

        self.targetThreshFlag = False

        if 'targetThreshFlag' in self.config:
            self.targetThreshFlag = self.config['targetThreshFlag']

        if 'target_start_thresh' in self.config:
            self.startThresh = self.config['target_start_thresh']
        if 'target_end_thresh' in self.config:
            self.endThresh = self.config['target_end_thresh']
        if 'distance_thresh_decay' in self.config:
            self.distanceThreshDecay = self.config['distance_thresh_decay']

        self.obstacleFlag = False
        if 'obstacleFlag' in self.config:
            self.obstacleFlag = self.config['obstacleFlag']

        self.dynamicObstacleFlag = False
        if 'dynamicObstacleFlag' in self.config:
            self.dynamicObstacleFlag = self.config['dynamicObstacleFlag']
            self.wallWidth = self.config['wallWidth']
            self.wallLength = self.config['wallLength']
            self.n_channels = 4
            if 'n_channels' in self.config:
                self.n_channels = self.config['n_channels']

        self.nStep = self.config['modelNStep']

        self.distanceScale = 20
        if 'distanceScale' in self.config:
            self.distanceScale = self.config['distanceScale']

        self.actionPenalty = 0.0
        if 'actionPenalty' in self.config:
            self.actionPenalty = self.config['actionPenalty']

        self.obstaclePenalty = 0.0
        if 'obstaclePenalty' in self.config:
            self.obstaclePenalty = self.config['obstaclePenalty']

        self.finishThresh = 1.0
        if 'finishThresh' in self.config:
            self.finishThresh = self.config['finishThresh']

        self.timingFlag = False
        if 'timingFlag' in self.config:
            self.timingFlag = self.config['timingFlag']

            self.timeScale = 100
            self.timeWindowLocation = self.config['timeWindowLocation']
            self.rewardArray = self.config['rewardArray']
            if 'timeScale' in self.config:
                self.timeScale = self.config['timeScale']

            if self.dynamicObstacleFlag:
                raise Exception('timing for free space or dynamic obstacle case is not implemented!')


    def thresh_by_episode(self, step):
        return self.endThresh + (
                self.startThresh - self.endThresh) * math.exp(-1. * step / self.distanceThreshDecay)
    def constructSensorArrayIndex(self):
        x_int = np.arange(-self.receptHalfWidth, self.receptHalfWidth + 1)
        y_int = np.arange(-self.receptHalfWidth, self.receptHalfWidth + 1)
        [Y, X] = np.meshgrid(y_int, x_int)
        self.senorIndex = np.stack((X.reshape(-1), Y.reshape(-1)), axis=1)
    def getSensorInfo(self):
    # sensor information needs to consider orientation information
    # add integer resentation of location
    #    index = self.senorIndex + self.currentState + np.array([self.padding, self.padding])
        phi = self.currentState[2]
    #   phi = (self.stepCount)*math.pi/4.0
    # this is rotation matrix transform from local coordinate system to lab coordinate system
        rotMatrx = np.matrix([[math.cos(phi),  -math.sin(phi)],
                              [math.sin(phi), math.cos(phi)]])
        transIndex = np.matmul(self.senorIndex, rotMatrx.T).astype(np.int)

        i = math.floor(self.currentState[0] + 0.5)
        j = math.floor(self.currentState[1] + 0.5)

        transIndex[:, 0] += self.padding + i
        transIndex[:, 1] += self.padding + j

        # use augumented obstacle matrix to check collision
        self.sensorInfoMat = self.obsMap[transIndex[:, 0], transIndex[:, 1]].reshape(self.receptWidth, -1)

    def getSequenceSensorInfo(self):

        orientFlag = True
        if self.particleType in ['TWODIM']:
            orientFlag = False

        self.sequenceSensorInfoMat = self.model.getObservation(orientFlag)
        self.sequenceSensorInfoMat.shape = (self.n_channels, self.receptWidth, self.receptWidth)


    def getSensorInfoFromPos(self, position):
        phi = position[2]

        rotMatrx = np.matrix([[math.cos(phi),  -math.sin(phi)],
                              [math.sin(phi), math.cos(phi)]])
        transIndex = np.matmul(self.senorIndex, rotMatrx.T).astype(np.int)

        i = math.floor(position[0] + 0.5)
        j = math.floor(position[1] + 0.5)

        transIndex[:, 0] += self.padding + i
        transIndex[:, 1] += self.padding + j

        # use augumented obstacle matrix to check collision
        sensorInfoMat = self.obsMap[transIndex[:, 0], transIndex[:, 1]].reshape(self.receptWidth, -1)

        # use augumented obstacle matrix to check collision
        return np.expand_dims(sensorInfoMat, axis = 0)

    def getExperienceAugmentation(self, state, action, nextState, reward, info):
        if self.timingFlag:
            raise Exception('timing for experience Augmentation case is not implemented!')

        state_Aug, action_Aug, nextState_Aug, reward_Aug = [], [], [], []
        if not self.obstacleFlag:
            if self.particleType == 'FULLCONTROL':
                # state is the position of target in the local frame
                # here uses the mirror relation
                state_Aug.append(np.array([state[0], -state[1]]))
                action_Aug.append(np.array([action[0], -action[1]]))
                if nextState is None:
                    nextState_Aug.append(None)
                else:
                    nextState_Aug.append(np.array([nextState[0], -nextState[1]]))
                reward_Aug.append(reward)
            elif self.particleType == 'SLIDER':
                state_Aug.append(np.array([state[0], -state[1]]))
                action_Aug.append(np.array([-action[0]]))
                if nextState is None:
                    nextState_Aug.append(None)
                else:
                    nextState_Aug.append(np.array([nextState[0], -nextState[1]]))
                reward_Aug.append(reward)
            elif self.particleType == 'VANILLASP':
                state_Aug.append(np.array([state[0], -state[1]]))
                action_Aug.append(np.array([action[0]]))
                if nextState is None:
                    nextState_Aug.append(None)
                else:
                    nextState_Aug.append(np.array([nextState[0], -nextState[1]]))
                reward_Aug.append(reward)
        return state_Aug, action_Aug, nextState_Aug, reward_Aug

    def getHindSightExperience(self, state, action, nextState, info):


        if self.hindSightInfo['obstacle']:
            return None, None, None, None
        else:
            targetNew = self.hindSightInfo['currentState'][0:2]
            distance = targetNew - self.hindSightInfo['previousState'][0:2]
            phi = self.hindSightInfo['previousState'][2]
            # distance will be changed from lab coordinate to local coordinate
            dx = distance[0] * math.cos(phi) + distance[1] * math.sin(phi)
            dy = - distance[0] * math.sin(phi) + distance[1] * math.cos(phi)

            angle = math.atan2(dy, dx)
            if math.sqrt(dx**2 + dy**2) > self.targetClipLength:
                dx = self.targetClipLength * math.cos(angle)
                dy = self.targetClipLength * math.sin(angle)

            if self.obstacleFlag and not self.dynamicObstacleFlag:
                sensorInfoMat = self.getSensorInfoFromPos(self.hindSightInfo['previousState'])
                if not self.timingFlag:
                    stateNew = {'sensor': sensorInfoMat,
                                'target': np.array([dx, dy]) / self.distanceScale}
                else:
                    stateNew = {'sensor': sensorInfoMat,
                                'target': np.array([dx / self.distanceScale, dy / self.distanceScale, state['target'][2]])}
            elif self.obstacleFlag and self.dynamicObstacleFlag:
                stateNew = {'sensor': state['sensor'],
                            'target': np.array([dx, dy]) / self.distanceScale}
            else:
                if not self.timingFlag:
                    stateNew = np.array([dx, dy]) / self.distanceScale
                else:
                    stateNew = np.array([dx / self.distanceScale, dy / self.distanceScale, \
                                             state[2]])
            actionNew = action
            rewardNew = 1.0 + self.actionPenaltyCal(action)
            if self.timingFlag:
                if info['timeStep'] < self.timeWindowLocation[0]:
                    rewardNew = -1.0
                if info['timeStep'] > self.timeWindowLocation[1]:
                    rewardNew = 0.1
            return stateNew, actionNew, None, rewardNew

    def constructSensorArrayIndex(self):
        x_int = np.arange(-self.receptHalfWidth, self.receptHalfWidth + 1)
        y_int = np.arange(-self.receptHalfWidth, self.receptHalfWidth + 1)
        [Y, X] = np.meshgrid(y_int, x_int)
        self.senorIndex = np.stack((X.reshape(-1), Y.reshape(-1)), axis=1)
        # sensormap maps a location (x, y) to to an index. for example (-5, -5) to 0
        # self.sensorMap = {}
        # for i, x in enumerate(x_int):
        #     for j, y in enumerate(y_int):
        #         self.sensorMap[(x, y)] = i * self.receptWidth + j


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
        return action
    def actionPenaltyCal(self, action):
        actionNorm = np.linalg.norm(action, ord=2)
        return -self.actionPenalty * actionNorm ** 2

    def obstaclePenaltyCal(self):

        if self.obstacleFlag and not self.dynamicObstacleFlag:
            i = math.floor(self.currentState[0] + 0.5)
            j = math.floor(self.currentState[1] + 0.5)

            xIdx = self.padding + i
            yIdx = self.padding + j

            if self.obsMap[xIdx, yIdx] > 0:
                return -self.obstaclePenalty, True
            else:
                return 0, False
        if self.obstacleFlag and self.dynamicObstacleFlag:
            trapFlag = self.model.checkDynamicTrap()
            if trapFlag:
                self.info['dynamicTrap'] += 1
                return -self.obstaclePenalty, True
            else:

                return 0, False

    def step(self, action):
        self.hindSightInfo['obstacle'] = False
        self.hindSightInfo['previousState'] = self.currentState.copy()
        reward = 0.0
        #if self.customExploreFlag and self.epiCount < self.customExploreEpisode:
        #    action = self.getCustomAction()
        self.model.step(self.nStep, action)
        if self.obstacleFlag and self.dynamicObstacleFlag:
            self.model.storeDynamicObstacles()
        self.currentState = self.model.getPositions()
        #self.currentState = self.currentState + 2.0 * np.array([action[0], action[1], 0])

        self.hindSightInfo['currentState'] = self.currentState.copy()

        distance = self.targetState - self.currentState[0:2]

        # update step count
        self.stepCount += 1

        done = False

        if self.is_terminal(distance):
            reward = 1.0
            if self.timingFlag:
                if self.stepCount < self.timeWindowLocation[0]:
                    reward = -1.0
                if self.stepCount > self.timeWindowLocation[1]:
                    reward = 1.0

            done = True


        # penalty for actions
        reward += self.actionPenaltyCal(action)

        # update sensor information
        if self.obstacleFlag:
            if not self.dynamicObstacleFlag:
                self.getSensorInfo()
            else:
                self.getSequenceSensorInfo()
            penalty, flag = self.obstaclePenaltyCal()
            reward += penalty
            if flag:
                self.hindSightInfo['obstacle'] = True
                self.currentState = self.hindSightInfo['previousState'].copy()
                #if self.dynamicObstacleFlag:
                #    done = True


        # distance will be changed from lab coordinate to local coordinate
        phi = self.currentState[2]
        dx = distance[0] * math.cos(phi) + distance[1] * math.sin(phi)
        dy = - distance[0] * math.sin(phi) + distance[1] * math.cos(phi)

        angle = math.atan2(dy, dx)
        if math.sqrt(dx**2 + dy**2) > self.targetClipLength:
            dx = self.targetClipLength * math.cos(angle)
            dy = self.targetClipLength * math.sin(angle)

        # recover the global target position after target mapping
        globalTargetX = self.currentState[0] + dx * math.cos(phi) - dy * math.sin(phi)
        globalTargetY = self.currentState[1] + dx * math.sin(phi) + dy * math.cos(phi)

        self.info['previousTarget'] = self.info['currentTarget'].copy()
        self.info['currentState'] = self.currentState.copy()
        self.info['targetState'] = self.targetState.copy()
        self.info['currentTarget'] = np.array([globalTargetX, globalTargetY])
        self.info['currentDistance'] = math.sqrt(dx**2 + dy**2)
        self.info['timeStep'] = self.stepCount
        if self.obstacleFlag:
            if not self.dynamicObstacleFlag:
                if not self.timingFlag:
                    state = {'sensor': np.expand_dims(self.sensorInfoMat, axis=0),
                             'target': np.array([dx , dy]) / self.distanceScale}
                else:
                    state = {'sensor': np.expand_dims(self.sensorInfoMat, axis=0),
                             'target': np.array([dx / self.distanceScale, dy / self.distanceScale, \
                                                 float(self.stepCount) / self.timeScale])}
            else:
                state = {'sensor': self.sequenceSensorInfoMat.copy(),
                         'target': np.array([dx, dy]) / self.distanceScale}
        else:
            if not self.timingFlag:
                state = np.array([dx, dy]) / self.distanceScale
            else:
                state = np.array([dx / self.distanceScale, dy / self.distanceScale, \
                          float(self.stepCount) / self.timeScale])
        return state, reward, done, self.info.copy()

    def is_terminal(self, distance):
        return np.linalg.norm(distance, ord=np.inf) < self.finishThresh

    def reset_helper(self):


        if self.obstacleFlag and self.dynamicObstacleFlag:
            if self.config['dynamicTargetFlag']:
                while True:
                    col = random.randint(2, self.wallWidth - 2)
                    row = random.randint(2, self.wallLength - 2)
                    if not self.model.checkDynamicTrapAround(row, col, 3.0, 10.0):
                        break
                self.targetState = np.array([row, col], dtype=np.int32)

            targetThresh = float('inf')
            if self.targetThreshFlag:
                targetThresh = self.thresh_by_episode(self.epiCount) * max(self.mapMat.shape)
                print('target Thresh', targetThresh)

            if self.config['dynamicInitialStateFlag']:
                while True:

                    col = random.randint(2, self.wallWidth - 2)
                    row = random.randint(2, self.wallLength - 2)
                    distanctVec = np.array([row, col],
                                           dtype=np.float32) - self.targetState
                    distance = np.linalg.norm(distanctVec, ord=np.inf)
                    if not self.model.checkDynamicTrapAround(row, col, 4.0, 10.0) and distance < targetThresh and not self.is_terminal(
                            distanctVec):
                        break
                # set initial state
                print('target distance', distance)
                self.currentState = np.array([row, col, random.random() * 2 * math.pi],
                                             dtype=np.float32)


        else:
            obstacleThresh = -12
            # set target information
            if self.config['dynamicTargetFlag']:
                while True:
                    col = random.randint(0, self.mapMat.shape[1] - 1) + self.padding
                    row = random.randint(0, self.mapMat.shape[0] - 1) + self.padding
                    if np.sum(self.obsMap[row-2:row+3, col-2:col+3]) < obstacleThresh:
                        break
                self.targetState = np.array([row - self.padding, col - self.padding], dtype=np.int32)



            targetThresh = float('inf')
            if self.targetThreshFlag:
                targetThresh = self.thresh_by_episode(self.epiCount) * max(self.mapMat.shape)
                print('target Thresh', targetThresh)


            if self.config['dynamicInitialStateFlag']:
                while True:

                    col = random.randint(0, self.mapMat.shape[1] - 1) + self.padding
                    row = random.randint(0, self.mapMat.shape[0] - 1) + self.padding
                    distanctVec = np.array([row - self.padding, col - self.padding], dtype=np.float32) - self.targetState
                    distance = np.linalg.norm(distanctVec, ord=np.inf)
                    if np.sum(self.obsMap[row-2:row+3, col-2:col+3]) < obstacleThresh and distance < targetThresh and not self.is_terminal(distanctVec):
                        break
                # set initial state
                print('target distance', distance)
                self.currentState = np.array([row - self.padding, col - self.padding, random.random()*2*math.pi], dtype=np.float32)


    def generateTimeStep(self):
        if self.epiCount < self.config['randomEpisode']:
            return random.choice(list(range(self.timeWindowLocation[1])))
        else:
            return 0

    def reset(self):
        self.stepCount = 0
        if self.timingFlag:
            self.stepCount = self.generateTimeStep()

        self.hindSightInfo = {}

        self.info = {}
        self.info['dynamicTrap'] = 0
        self.info['timeStep'] = self.stepCount

        self.info['scaleFactor'] = self.distanceScale
        self.epiCount += 1

        self.currentState = np.array(self.config['currentState'], dtype=np.float32)
        self.targetState = np.array(self.config['targetState'], dtype=np.int32)

        self.model.createInitialState(0.0, 0.0, 0.0)
        self.reset_helper()
        self.model.setInitialState(self.currentState[0], self.currentState[1], self.currentState[2])

        if self.particleType == 'TWODIM': # For TWODIM active particle, orientation does not matter
            self.currentState[2] = 0.0

        # update sensor information
        if self.obstacleFlag:
            if not self.dynamicObstacleFlag:
                self.getSensorInfo()
            else:
                self.getSequenceSensorInfo()

        # pre-run obstacle simulations
        if self.obstacleFlag and self.dynamicObstacleFlag:
            for n in range(self.n_channels):
                self.model.updateDynamicObstacles(self.nStep)
                self.model.storeDynamicObstacles()
        distance = self.targetState - self.currentState[0:2]

        # distance will be change to local coordinate
        phi = self.currentState[2]
        dx = distance[0] * math.cos(phi) + distance[1] * math.sin(phi)
        dy = - distance[0] * math.sin(phi) + distance[1] * math.cos(phi)

        angle = math.atan2(dy, dx)
        if math.sqrt(dx**2 + dy**2) > self.targetClipLength:
            dx = self.targetClipLength * math.cos(angle)
            dy = self.targetClipLength * math.sin(angle)

        globalTargetX = self.currentState[0]+ dx * math.cos(phi) - dy * math.sin(phi)
        globalTargetY = self.currentState[1]+ dx * math.sin(phi) + dy * math.cos(phi)

        self.info['currentTarget'] = np.array([globalTargetX, globalTargetY])

        #angleDistance = math.atan2(distance[1], distance[0]) - self.currentState[2]
        if self.obstacleFlag and not self.dynamicObstacleFlag:
            if not self.timingFlag:
                combinedState = {'sensor': np.expand_dims(self.sensorInfoMat, axis=0),
                                 'target': np.array([dx, dy]) / self.distanceScale}
            else:
                combinedState = {'sensor': np.expand_dims(self.sensorInfoMat, axis=0),
                         'target': np.array([dx / self.distanceScale, dy / self.distanceScale, \
                                             float(self.stepCount) / self.timeScale])}
            return combinedState
        elif self.obstacleFlag and self.dynamicObstacleFlag:
            combinedState = {'sensor': self.sequenceSensorInfoMat.copy(),
                             'target': np.array([dx, dy]) / self.distanceScale}
            return combinedState
        else:
            if not self.timingFlag:
                return np.array([dx , dy]) / self.distanceScale
            else:
                return np.array([dx / self.distanceScale, dy / self.distanceScale, \
                                             float(self.stepCount) / self.timeScale])

    def initObsMat(self):
        fileName = self.config['mapName']
        self.mapMat = np.genfromtxt(fileName + '.txt')
        self.mapShape = self.mapMat.shape
        padW = self.config['obstacleMapPaddingWidth']
        obsMapSizeOne = self.mapMat.shape[0] + 2*padW
        obsMapSizeTwo = self.mapMat.shape[1] + 2*padW
        self.obsMap = np.ones((obsMapSizeOne, obsMapSizeTwo))
        self.obsMap[padW:-padW, padW:-padW] = self.mapMat

        self.obsMap -= 0.5
        self.mapMat -= 0.5
        np.savetxt(self.config['mapName']+'obsMap.txt', self.obsMap, fmt='%.1f', delimiter='\t')


class ActiveParticleEscapeEnv(ActiveParticleEnv):
    def __init__(self, configName, randomSeed=1):
        super(ActiveParticleEscapeEnv, self).__init__(configName, randomSeed)

    def read_config(self):

        super(ActiveParticleEscapeEnv, self).read_config()
        self.escapeFlag = False
        if 'excapeFlag' in self.config:
            self.escapeFlag = self.config['excapeFlag']

        self.hazardPenalty = 0.01
        if 'obstaclePenalty' in self.config:
            self.hazardPenalty = self.config['hazardPenalty']
    def getSensorInfo(self):
        # sensor information needs to consider orientation information
        # add integer resentation of location
        #    index = self.senorIndex + self.currentState + np.array([self.padding, self.padding])
        phi = self.currentState[2]
        #   phi = (self.stepCount)*math.pi/4.0
        # this is rotation matrix transform from local coordinate system to lab coordinate system
        rotMatrx = np.matrix([[math.cos(phi), -math.sin(phi)],
                              [math.sin(phi), math.cos(phi)]])
        transIndex = np.matmul(self.senorIndex, rotMatrx.T).astype(np.int)

        i = math.floor(self.currentState[0] + 0.5)
        j = math.floor(self.currentState[1] + 0.5)

        transIndex[:, 0] += self.padding + i
        transIndex[:, 1] += self.padding + j

        # use augumented obstacle matrix to check collision
        self.sensorInfoMat = np.empty((2, self.receptWidth, self.receptWidth))
        self.sensorInfoMat[0, :, :] = self.obsMap[transIndex[:, 0], transIndex[:, 1]].reshape(self.receptWidth, -1)
        self.sensorInfoMat[1, :, :] = self.hazardMap[transIndex[:, 0], transIndex[:, 1]].reshape(self.receptWidth, -1)

    def getSensorInfoFromPos(self, position):
        phi = position[2]

        rotMatrx = np.matrix([[math.cos(phi), -math.sin(phi)],
                              [math.sin(phi), math.cos(phi)]])
        transIndex = np.matmul(self.senorIndex, rotMatrx.T).astype(np.int)

        i = math.floor(position[0] + 0.5)
        j = math.floor(position[1] + 0.5)

        transIndex[:, 0] += self.padding + i
        transIndex[:, 1] += self.padding + j

        sensorInfoMat = np.empty((2, self.receptWidth, self.receptWidth))
        # use augumented obstacle matrix to check collision
        sensorInfoMat[0, :, :] = self.obsMap[transIndex[:, 0], transIndex[:, 1]].reshape(self.receptWidth, -1)
        sensorInfoMat[1, :, :] = self.hazardMap[transIndex[:, 0], transIndex[:, 1]].reshape(self.receptWidth, -1)

        # use augumented obstacle matrix to check collision
        return sensorInfoMat.copy()

    def getHindSightExperience(self, state, action, nextState, info):

        raise Exception("not implemented")

    def hazardPenaltyCal(self):
        i = math.floor(self.currentState[0] + 0.5)
        j = math.floor(self.currentState[1] + 0.5)

        xIdx = self.padding + i
        yIdx = self.padding + j

        if self.hazardMap[xIdx, yIdx] > 0:
            return -self.hazardPenalty, True
        else:
            return 0, False

    def step(self, action):
        reward = 0.0
        self.model.step(self.nStep, action)
        self.currentState = self.model.getPositions()

        done = False

        penalty, flag = self.hazardPenaltyCal()
        # if not in hazard region, then it escapes
        if not flag:
            reward = 1.0
            done = True
        else:
            reward += penalty

        # penalty for actions
        reward += self.actionPenaltyCal(action)
        self.getSensorInfo()
        # update sensor information
        if self.obstacleFlag:
            penalty, flag = self.obstaclePenaltyCal()
            reward += penalty


        # update step count
        self.stepCount += 1
        state = self.sensorInfoMat.copy()
        return state, reward, done, self.info.copy()

    def reset(self):
        self.stepCount = 0
        self.info = {}
        self.info['scaleFactor'] = self.distanceScale
        self.epiCount += 1

        self.currentState = np.array(self.config['currentState'], dtype=np.float32)

        self.reset_helper()
        if self.particleType == 'TWODIM': # For TWODIM active particle, orientation does not matter
            self.currentState[2] = 0.0

        self.info['currentState'] = self.currentState.copy()
        self.model.createInitialState(self.currentState[0], self.currentState[1], self.currentState[2])
        # update sensor information

        self.getSensorInfo()

        return self.sensorInfoMat.copy()


    def reset_helper(self):

        # because obstacles/hazard regions  are represented by 0.5 and free space is represented by -0.5
        obstacleThresh = -12

        if self.config['dynamicInitialStateFlag']:
            while True:

                col = random.randint(0, self.mapMat.shape[1] - 1) + self.padding
                row = random.randint(0, self.mapMat.shape[0] - 1) + self.padding
                if np.sum(self.obsMap[row - 2:row + 3, col - 2:col + 3]) < obstacleThresh \
                        and np.sum(self.hazardMap[row - 2:row + 3, col - 2:col + 3]) > -obstacleThresh:
                    break
            # set initial state
            self.currentState = np.array([row - self.padding, col - self.padding, random.random() * 2 * math.pi],
                                         dtype=np.float32)

    def initObsMat(self):
        fileName = self.config['mapName']
        self.mapMat = np.genfromtxt(fileName + '.txt')
        padW = self.config['obstacleMapPaddingWidth']
        mapSizeOne = self.mapMat.shape[0] + 2 * padW
        mapSizeTwo = self.mapMat.shape[1] + 2 * padW
        self.obsMap = np.ones((mapSizeOne, mapSizeTwo))
        self.obsMap[padW:-padW, padW:-padW] = self.mapMat

        self.obsMap -= 0.5
        np.savetxt(self.config['mapName'] + 'obsMap.txt', self.obsMap, fmt='%.1f', delimiter='\t')

        fileName = self.config['hazardMapName']
        mapMat = np.genfromtxt(fileName + '.txt')
        padW = self.config['obstacleMapPaddingWidth']
        mapSizeOne = mapMat.shape[0] + 2 * padW
        mapSizeTwo = mapMat.shape[1] + 2 * padW
        self.hazardMap = np.zeros((mapSizeOne, mapSizeTwo)) # zero are good regions
        self.hazardMap[padW:-padW, padW:-padW] = mapMat

        self.hazardMap -= 0.5
        np.savetxt(self.config['mapName'] + 'hazardMap.txt', self.hazardMap, fmt='%.1f', delimiter='\t')

        assert(self.obsMap.shape == self.hazardMap.shape)

class ActiveParticleEnvMultiMap(ActiveParticleEnv):
    def __init__(self, config, seed=1):
        super(ActiveParticleEnvMultiMap, self).__init__(config, seed)

    def initilize(self):
        self.mapNames = self.config['multiMapName'].split(",")
        self.mapProb = self.config['multiMapProb']
        self.readMaze()
        super(ActiveParticleEnvMultiMap, self).initilize()

    def readMaze(self):

        self.mapMatList = []
        self.mapShapeList = []
        for mapName in self.mapNames:
            mapMat = np.genfromtxt(mapName + '.txt')
            mapShape = mapMat.shape
            self.mapMatList.append(mapMat)
            self.mapShapeList.append(mapShape)

        self.mapMat = self.mapMatList[0]
        self.mapShape = self.mapShapeList[0]
        self.numMaps = len(self.mapMatList)

    def reset(self):
        # randomly chosen a map
        mapIdx = np.random.choice(self.numMaps, p=self.mapProb)
        self.mapMat = self.mapMatList[mapIdx]
        self.obsMap = self.obsMatList[mapIdx]

        print("map used:", self.mapNames[mapIdx])
        return super(ActiveParticleEnvMultiMap, self).reset()

    def initObsMat(self):

        padW = self.config['obstacleMapPaddingWidth']

        self.obsMatList = []
        for idx, map in enumerate(self.mapMatList):
            obsMapSizeOne = map.shape[0] + 2*padW
            obsMapSizeTwo = map.shape[1] + 2*padW
            obsMat = np.ones((obsMapSizeOne, obsMapSizeTwo))
            obsMat[padW:-padW, padW:-padW] = map
            self.obsMatList.append(obsMat)
            np.savetxt(self.mapNames[idx]+'obsMap.txt', obsMat, fmt='%d', delimiter='\t')

        self.obsMat = self.obsMatList[0]
