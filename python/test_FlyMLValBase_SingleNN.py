#!/usr/bin/python
#---------general stuff import -----------------
import numpy as np
import math
import numpy as np
import random
##import matplotlib
from collections import namedtuple
from itertools import count
#---------torch stuff import -----------------
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim  as optim
import torch.nn.functional as F
import torchvision.transforms as T
#---------Flight stuff import -----------------
import pyFlight as flight
from pynput import keyboard
import time
#------- Setting GPU --------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
myTime=0
#------- Replay Memory --------------------------------------
#fig = plt.gcf()
#fig.show()
#fig.canvas.draw()
Transition = namedtuple('Transition', ('state', 'action1','next_state', 'reward'))
class ReplayMemory(object):
  def __init__(self, capacity):
    self.capacity = capacity
    self.memory = []
    self.position = 0
  def push(self, *args):
    if len(self.memory) < self.capacity:
      self.memory.append(None)
    self.memory[self.position] = Transition(*args)
    self.position = (self.position + 1) % self.capacity # POSITION??? xyz?
  def sample(self, batch_size):
    return random.sample(self.memory, batch_size)
  def __len__(self):
    return len(self.memory)
#------- Q-network --------------------------------------
class DQN(nn.Module):
  def __init__(self, inpSize, hidSize, numClass):
    super(DQN, self).__init__()
    self.fc1 = nn.Linear(inpSize, hidSize)
    self.relu1 = nn.LeakyReLU()
    self.relu2 = nn.LeakyReLU()
    self.relu3 = nn.LeakyReLU(0.1)
    self.fc2 = nn.Linear(hidSize, numClass)
  def forward(self,x):
   x = x.type(torch.FloatTensor)
   out = self.fc1(x)
   out = self.relu1(out)
   out = self.relu2(out)
   out = self.relu3(out)
   out = self.fc2(out)
   return out

#------- Training Setup--------------------------------------
BATCH_SIZE = 10000
GAMMA = 0.9
EPS_START = 0.99
EPS_END = 0.001
EPS_DECAY = 20000000
TARGET_UPDATE = 1000

policy_net = DQN(26,20,65)#.to(device) 
target_net = DQN(26,20,65)#.to(device) 

target_net.load_state_dict(policy_net.state_dict()) 
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(1000000)

steps_done = 0
eps_threshold = 0
def select_action(state, policy_net):
  global steps_done, eps_threshold
  sample = random.random()
#  eps_threshold = EPS_END + (EPS_START - EPS_END)*math.exp(-steps_done/EPS_DECAY)
  eps_threshold = 0.1*math.exp(-1.0*steps_done/(1.0*EPS_DECAY))
  if sample > eps_threshold:
    with torch.no_grad():
      return policy_net(state).max(-1)[1].view(1,1) 
##      return policy_net(state).min(-1)[1].view(1,1) 
  else:
    return torch.tensor([[random.randrange(65)]], dtype=torch.long)


#------- Set Optimizer --------------------------------------
def optimizer_model():
  global steps_done
  if len(memory) < BATCH_SIZE:
    return
  transitions = memory.sample(BATCH_SIZE)
  batch = Transition(*zip(*transitions))
  non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)))
  non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])
  state_batch = torch.stack(batch.state) ##

  action_batch = torch.stack(batch.action1)

  reward_batch = torch.cat(batch.reward)
  steps_done += 1

  state_action_values = policy_net(state_batch).gather(1, action_batch.squeeze().unsqueeze(1))

  #Compute V(s_{t+1}) for all next states.
  next_state_values = torch.zeros(BATCH_SIZE)
  next_state_values[non_final_mask] = target_net(non_final_next_states).max(-1)[0].detach()
  #Compute the expected Q values
  expected_state_action_values = (next_state_values*GAMMA) + reward_batch
#  
  #Compute Huberr loss
##  loss1 = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
  loss1 = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
  if(myTime % 1000 ==0):
      print(loss1,np.array(list(zip(state_action_values[:].detach().numpy(), expected_state_action_values.unsqueeze(1)[:].detach().numpy())))[:])
  #optimize the model
  optimizer.zero_grad()
  loss1.backward()
  for param in policy_net.parameters():
    param.grad.data.clamp_(-1,1)
#    param.grad.data.clamp_(0,1)

  optimizer.step()

#-- Start Flight Engine ----------------------------------------------------- 
# --- set workdir with configuration of aircraft, textures etc.
flight.loadFromFile("/home/aki/Dropbox/SimpleSimulationEngine/cpp/apps/AeroCombat/data/AeroCraftMainWing1.ini" )

# --- initialize visualization (optional, just for debugging)
#fview = flight.FlightView("/home/aki/Dropbox/SimpleSimulationEngine/")

# --- set initial state of aircraft   setPose( position, velocity, rotMat==[left,up,forward] )
flight.setPose(np.array([0.0,random.uniform(0,200),0.0]), np.array([random.uniform(0,100),random.uniform(-100,100),random.uniform(0,200)]) , np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]))
flight.setTilt( 2, 0.02 ) # set Angle-of-incidence for wing #2 ( = elevator)

# --- initialize targets Spheres (x,y,z,Radius)
targets = np.random.random((30,4));
flight.setTargets( targets )
# ---  initialize C-interface buffers
iPitch=0; iYaw=1; iRoll=2; iThrottle=3; iTrigger=4
controlBuff  = np.zeros(5)                   # python->C [pich,yaw,roll,throttle,trigger] differeces (rate-of-change)
##stateBuff    = Variable(torch.from_numpy(np.zeros(4*3 + 4)))             # C->python [ pos.xyz, vel.xyz, forward.xyz, up.xyz       pich,yaw,roll,throttle  ]
targetHits   = np.zeros( (len(targets),2) )  # C->python if hit [ageMin,ageMax] of projectiles which hit it, if no-hit [+1e+300,-1e+300]


tragetMoveSpeed = 10.0
num_episodes = 10000000
#------ Run ML Loop --------------------------------------

#======================================================================
flight.loadFromFile("/home/aki/Dropbox/SimpleSimulationEngine/cpp/apps/AeroCombat/data/AeroCraftMainWing1.ini" )
for i_episode in range(num_episodes):
  global myTime
  myTime = i_episode
  rewardSum = 0
  dicTensor = torch.tensor([1.0,1.0,1.0,1.0]).double()
  flight.setPose(np.array([0.0,random.uniform(1,49),0.0]), np.array([random.uniform(0,100),random.uniform(-100,100),random.uniform(1,100)]) , np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]))
#  flight.setTilt( 2, 0.02 ) # set Angle-of-incidence for wing #2 ( = elevator)
  flight.setTilt( 0, 0.00 ) # set Angle-of-incidence for wing #2 ( = elevator)
  flight.setTilt( 1, 0.00 ) 
  flight.setTilt( 2, 0.00 ) 
  flight.setTilt( 3, 0.00 ) 
##  stateBuff = Variable(torch.from_numpy(np.zeros(4*3 + 4)))             # C->python [ pos.xyz, vel.xyz, forward.xyz, up.xyz       pich,yaw,roll,throttle  ]
  stateBuff = Variable(torch.from_numpy(np.zeros(8*3 + 4)))             # C->python [ pos.xyz, vel.xyz, forward.xyz, up.xyz       pich,yaw,roll,throttle  ]
  stateBuff = stateBuff.numpy()
  flight.flyAndShootTargets( controlBuff, stateBuff, targetHits, nsub=100, dt=0.001 )
  stateBuff = torch.from_numpy(stateBuff)
  sumRward = 0
  state = []
  iframe = 0
  current = 0.0
  for t in count():
    done=False
    #---------- Start Step ---------------------------------------
    if tragetMoveSpeed > 0.0:
        phi = 0.01*iframe + np.array( range(len(targets)) )
        targets[:,2] += np.cos(phi)*tragetMoveSpeed
    # set controls by keyboard
    controlBuff[:]          =  0.0   # 0.0 means not change - preserve state of controls from previous step
##    myState = torch.cat((torch.tensor([stateBuff[i] for i in (1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27)]), dicTensor.double()))
    myState = torch.tensor([stateBuff[i] for i in (1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27)])
#    myAction = select_action(myState , policy_net)
    myAction = select_action(myState , target_net)
    if   myAction == 0: 
        controlBuff[iPitch]    = +0.0    *10
        controlBuff[iYaw]      = +0.0   *10
        controlBuff[iThrottle] = +0.0  *10
        controlBuff[iRoll]     = +0.0 *10
    elif   myAction == 1: 
        controlBuff[iPitch]    = +0.1 *10
        controlBuff[iYaw]      = +0.0 *10
        controlBuff[iThrottle] = +0.0 *10
        controlBuff[iRoll]     = +0.0 *10
    elif   myAction == 2: 
        controlBuff[iPitch]    = +0.0 *10
        controlBuff[iYaw]      = +0.1 *10
        controlBuff[iThrottle] = +0.0 *10
        controlBuff[iRoll]     = +0.0 *10
    elif   myAction == 3: 
        controlBuff[iPitch]    = +0.0 *10
        controlBuff[iYaw]      = +0.0 *10
        controlBuff[iThrottle] = +0.1 *10
        controlBuff[iRoll]     = +0.0 *10
    elif   myAction == 4: 
        controlBuff[iPitch]    = +0.0 *10
        controlBuff[iYaw]      = +0.0 *10
        controlBuff[iThrottle] = +0.0 *10
        controlBuff[iRoll]     = +0.1 *10
    elif   myAction == 5: 
        controlBuff[iPitch]    = -0.1 *10
        controlBuff[iYaw]      = -0.0 *10
        controlBuff[iThrottle] = -0.0 *10
        controlBuff[iRoll]     = -0.0 *10
    elif   myAction == 6: 
        controlBuff[iPitch]    = -0.0 *10
        controlBuff[iYaw]      = -0.1 *10
        controlBuff[iThrottle] = -0.0 *10
        controlBuff[iRoll]     = -0.0 *10
    elif   myAction == 7: 
        controlBuff[iPitch]    = -0.0 *10
        controlBuff[iYaw]      = -0.0 *10
        controlBuff[iThrottle] = -0.1 *10
        controlBuff[iRoll]     = -0.0 *10
    elif   myAction == 8: 
        controlBuff[iPitch]    = -0.0 *10
        controlBuff[iYaw]      = -0.0 *10
        controlBuff[iThrottle] = -0.0 *10
        controlBuff[iRoll]     = -0.1 *10
    elif   myAction == 9: 
        controlBuff[iPitch]    = +0.1 *10
        controlBuff[iYaw]      = +0.1 *10
        controlBuff[iThrottle] = +0.0 *10
        controlBuff[iRoll]     = +0.0 *10
    elif   myAction == 10: 
        controlBuff[iPitch]    = +0.1 *10
        controlBuff[iYaw]      = +0.0 *10
        controlBuff[iThrottle] = +0.1 *10
        controlBuff[iRoll]     = +0.0 *10
    elif   myAction == 11: 
        controlBuff[iPitch]    = +0.1 *10
        controlBuff[iYaw]      = +0.0 *10
        controlBuff[iThrottle] = +0.0 *10
        controlBuff[iRoll]     = +0.1 *10
    elif   myAction == 12: 
        controlBuff[iPitch]    = +0.0 *10
        controlBuff[iYaw]      = +0.1 *10
        controlBuff[iThrottle] = +0.1 *10
        controlBuff[iRoll]     = +0.0 *10
    elif   myAction == 13: 
        controlBuff[iPitch]    = +0.0 *10
        controlBuff[iYaw]      = +0.1 *10
        controlBuff[iThrottle] = +0.0 *10
        controlBuff[iRoll]     = +0.1 *10
    elif   myAction == 14: 
        controlBuff[iPitch]    = +0.0 *10
        controlBuff[iYaw]      = +0.0 *10
        controlBuff[iThrottle] = +0.1 *10
        controlBuff[iRoll]     = +0.1 *10
    elif   myAction == 15: 
        controlBuff[iPitch]    = -0.1 *10
        controlBuff[iYaw]      = -0.1 *10
        controlBuff[iThrottle] = -0.0 *10
        controlBuff[iRoll]     = -0.0 *10
    elif   myAction == 16: 
        controlBuff[iPitch]    = -0.1 *10
        controlBuff[iYaw]      = -0.0 *10
        controlBuff[iThrottle] = -0.1 *10
        controlBuff[iRoll]     = -0.0 *10
    elif   myAction == 17: 
        controlBuff[iPitch]    = -0.1 *10
        controlBuff[iYaw]      = -0.0 *10
        controlBuff[iThrottle] = -0.0 *10
        controlBuff[iRoll]     = -0.1 *10
    elif   myAction == 18: 
        controlBuff[iPitch]    = -0.0 *10
        controlBuff[iYaw]      = -0.1 *10
        controlBuff[iThrottle] = -0.1 *10
        controlBuff[iRoll]     = -0.0 *10
    elif   myAction == 19: 
        controlBuff[iPitch]    = -0.0 *10
        controlBuff[iYaw]      = -0.1 *10
        controlBuff[iThrottle] = -0.0 *10
        controlBuff[iRoll]     = -0.1 *10
    elif   myAction == 20: 
        controlBuff[iPitch]    = -0.0 *10
        controlBuff[iYaw]      = -0.0 *10
        controlBuff[iThrottle] = -0.1 *10
        controlBuff[iRoll]     = -0.1 *10
    elif   myAction == 21: 
        controlBuff[iPitch]    = +0.1 *10
        controlBuff[iYaw]      = +0.1 *10
        controlBuff[iThrottle] = +0.1 *10
        controlBuff[iRoll]     = +0.0 *10
    elif   myAction == 22: 
        controlBuff[iPitch]    = +0.1 *10
        controlBuff[iYaw]      = +0.1 *10
        controlBuff[iThrottle] = +0.0 *10
        controlBuff[iRoll]     = +0.1 *10
    elif   myAction == 23: 
        controlBuff[iPitch]    = +0.0 *10
        controlBuff[iYaw]      = +0.1 *10
        controlBuff[iThrottle] = +0.1 *10
        controlBuff[iRoll]     = +0.1 *10
    elif   myAction == 24: 
        controlBuff[iPitch]    = +0.1 *10
        controlBuff[iYaw]      = +0.0 *10
        controlBuff[iThrottle] = +0.1 *10
        controlBuff[iRoll]     = +0.1 *10
    elif   myAction == 25: 
        controlBuff[iPitch]    = -0.1 *10
        controlBuff[iYaw]      = -0.1 *10
        controlBuff[iThrottle] = -0.1 *10
        controlBuff[iRoll]     = -0.0 *10
    elif   myAction == 26: 
        controlBuff[iPitch]    = -0.1 *10
        controlBuff[iYaw]      = -0.0 *10
        controlBuff[iThrottle] = -0.1 *10
        controlBuff[iRoll]     = -0.1 *10
    elif   myAction == 27: 
        controlBuff[iPitch]    = -0.1 *10
        controlBuff[iYaw]      = -0.1 *10
        controlBuff[iThrottle] = -0.0 *10
        controlBuff[iRoll]     = -0.1 *10
    elif   myAction == 28: 
        controlBuff[iPitch]    = -0.0 *10
        controlBuff[iYaw]      = -0.1 *10
        controlBuff[iThrottle] = -0.1 *10
        controlBuff[iRoll]     = -0.1 *10
    elif   myAction == 29: 
        controlBuff[iPitch]    = +0.1 *10
        controlBuff[iYaw]      = +0.1 *10
        controlBuff[iThrottle] = +0.1 *10
        controlBuff[iRoll]     = +0.1 *10
    elif   myAction == 30: 
        controlBuff[iPitch]    = -0.1 *10
        controlBuff[iYaw]      = -0.1 *10
        controlBuff[iThrottle] = -0.1 *10
        controlBuff[iRoll]     = -0.1 *10
    elif   myAction == 31: 
        controlBuff[iPitch]    = +0.1 *10
        controlBuff[iYaw]      = -0.1 *10
        controlBuff[iThrottle] = -0.0 *10
        controlBuff[iRoll]     = -0.0 *10
    elif   myAction == 32: 
        controlBuff[iPitch]    = +0.1 *10
        controlBuff[iYaw]      = -0.0 *10
        controlBuff[iThrottle] = -0.1 *10
        controlBuff[iRoll]     = -0.0 *10
    elif   myAction == 33: 
        controlBuff[iPitch]    = +0.1 *10
        controlBuff[iYaw]      = -0.0 *10
        controlBuff[iThrottle] = -0.0 *10
        controlBuff[iRoll]     = -0.1 *10
    elif   myAction == 34: 
        controlBuff[iPitch]    = +0.1 *10
        controlBuff[iYaw]      = -0.1 *10
        controlBuff[iThrottle] = -0.1 *10
        controlBuff[iRoll]     = -0.0 *10
    elif   myAction == 35: 
        controlBuff[iPitch]    = +0.1 *10
        controlBuff[iYaw]      = -0.1 *10
        controlBuff[iThrottle] = -0.0 *10
        controlBuff[iRoll]     = -0.1 *10
    elif   myAction == 36: 
        controlBuff[iPitch]    = +0.1 *10
        controlBuff[iYaw]      = -0.0 *10
        controlBuff[iThrottle] = -0.1 *10
        controlBuff[iRoll]     = -0.1 *10
    elif   myAction == 37: 
        controlBuff[iPitch]    = -0.1 *10
        controlBuff[iYaw]      = +0.1 *10
        controlBuff[iThrottle] = -0.0 *10
        controlBuff[iRoll]     = -0.0 *10
    elif   myAction == 38: 
        controlBuff[iPitch]    = -0.0 *10
        controlBuff[iYaw]      = +0.1 *10
        controlBuff[iThrottle] = -0.1 *10
        controlBuff[iRoll]     = -0.0 *10
    elif   myAction == 39: 
        controlBuff[iPitch]    = -0.0 *10
        controlBuff[iYaw]      = +0.1 *10
        controlBuff[iThrottle] = -0.0 *10
        controlBuff[iRoll]     = -0.1 *10
    elif   myAction == 40: 
        controlBuff[iPitch]    = -0.1 *10
        controlBuff[iYaw]      = +0.1 *10
        controlBuff[iThrottle] = -0.1 *10
        controlBuff[iRoll]     = -0.0 *10
    elif   myAction == 41: 
        controlBuff[iPitch]    = -0.1 *10
        controlBuff[iYaw]      = +0.1 *10
        controlBuff[iThrottle] = -0.0 *10
        controlBuff[iRoll]     = -0.1 *10
    elif   myAction == 42: 
        controlBuff[iPitch]    = -0.0 *10
        controlBuff[iYaw]      = +0.1 *10
        controlBuff[iThrottle] = -0.1 *10
        controlBuff[iRoll]     = -0.1 *10
    elif   myAction == 43: 
        controlBuff[iPitch]    = -0.1 *10
        controlBuff[iYaw]      = -0.0 *10
        controlBuff[iThrottle] = +0.1 *10
        controlBuff[iRoll]     = -0.0 *10
    elif   myAction == 44: 
        controlBuff[iPitch]    = -0.0 *10
        controlBuff[iYaw]      = -0.1 *10
        controlBuff[iThrottle] = +0.1 *10
        controlBuff[iRoll]     = -0.0 *10
    elif   myAction == 45: 
        controlBuff[iPitch]    = -0.0 *10
        controlBuff[iYaw]      = -0.0 *10
        controlBuff[iThrottle] = +0.1 *10
        controlBuff[iRoll]     = -0.1 *10
    elif   myAction == 46: 
        controlBuff[iPitch]    = -0.0 *10
        controlBuff[iYaw]      = -0.1 *10
        controlBuff[iThrottle] = +0.1 *10
        controlBuff[iRoll]     = -0.1 *10
    elif   myAction == 47: 
        controlBuff[iPitch]    = -0.1 *10
        controlBuff[iYaw]      = -0.0 *10
        controlBuff[iThrottle] = +0.1 *10
        controlBuff[iRoll]     = -0.1 *10
    elif   myAction == 48: 
        controlBuff[iPitch]    = -0.1 *10
        controlBuff[iYaw]      = -0.1 *10
        controlBuff[iThrottle] = +0.1 *10
        controlBuff[iRoll]     = -0.0 *10
    elif   myAction == 49: 
        controlBuff[iPitch]    = -0.0 *10
        controlBuff[iYaw]      = -0.0 *10
        controlBuff[iThrottle] = -0.1 *10
        controlBuff[iRoll]     = +0.1 *10
    elif   myAction == 50: 
        controlBuff[iPitch]    = -0.0 *10
        controlBuff[iYaw]      = -0.1 *10
        controlBuff[iThrottle] = -0.0 *10
        controlBuff[iRoll]     = +0.1 *10
    elif   myAction == 51: 
        controlBuff[iPitch]    = -0.1 *10
        controlBuff[iYaw]      = -0.0 *10
        controlBuff[iThrottle] = -0.0 *10
        controlBuff[iRoll]     = +0.1 *10
    elif   myAction == 52: 
        controlBuff[iPitch]    = -0.0 *10
        controlBuff[iYaw]      = -0.1 *10
        controlBuff[iThrottle] = -0.1 *10
        controlBuff[iRoll]     = +0.1 *10
    elif   myAction == 53: 
        controlBuff[iPitch]    = -0.1 *10
        controlBuff[iYaw]      = -0.0 *10
        controlBuff[iThrottle] = -0.1 *10
        controlBuff[iRoll]     = +0.1 *10
    elif   myAction == 54: 
        controlBuff[iPitch]    = -0.1 *10
        controlBuff[iYaw]      = -0.1 *10
        controlBuff[iThrottle] = -0.0 *10
        controlBuff[iRoll]     = +0.1 *10
    elif   myAction == 55: 
        controlBuff[iPitch]    = -0.1 *10
        controlBuff[iYaw]      = -0.1 *10
        controlBuff[iThrottle] = +0.1 *10
        controlBuff[iRoll]     = +0.1 *10
    elif   myAction == 56: 
        controlBuff[iPitch]    = -0.1 *10
        controlBuff[iYaw]      = +0.1 *10
        controlBuff[iThrottle] = -0.1 *10
        controlBuff[iRoll]     = +0.1 *10
    elif   myAction == 57: 
        controlBuff[iPitch]    = -0.1 *10
        controlBuff[iYaw]      = +0.1 *10
        controlBuff[iThrottle] = +0.1 *10
        controlBuff[iRoll]     = -0.1 *10
    elif   myAction == 58: 
        controlBuff[iPitch]    = +0.1 *10
        controlBuff[iYaw]      = -0.1 *10
        controlBuff[iThrottle] = -0.1 *10
        controlBuff[iRoll]     = +0.1 *10
    elif   myAction == 59: 
        controlBuff[iPitch]    = +0.1 *10
        controlBuff[iYaw]      = -0.1 *10
        controlBuff[iThrottle] = +0.1 *10
        controlBuff[iRoll]     = -0.1 *10
    elif   myAction == 60: 
        controlBuff[iPitch]    = +0.1 *10
        controlBuff[iYaw]      = +0.1 *10
        controlBuff[iThrottle] = -0.1 *10
        controlBuff[iRoll]     = -0.1 *10
    elif   myAction == 61: 
        controlBuff[iPitch]    = -0.1 *10
        controlBuff[iYaw]      = +0.1 *10
        controlBuff[iThrottle] = +0.1 *10
        controlBuff[iRoll]     = +0.1 *10
    elif   myAction == 62: 
        controlBuff[iPitch]    = +0.1 *10
        controlBuff[iYaw]      = -0.1 *10
        controlBuff[iThrottle] = +0.1 *10
        controlBuff[iRoll]     = +0.1 *10
    elif   myAction == 63: 
        controlBuff[iPitch]    = +0.1 *10
        controlBuff[iYaw]      = +0.1 *10
        controlBuff[iThrottle] = -0.1 *10
        controlBuff[iRoll]     = +0.1 *10
    elif   myAction == 64: 
        controlBuff[iPitch]    = +0.1 *10
        controlBuff[iYaw]      = +0.1 *10
        controlBuff[iThrottle] = +0.1 *10
        controlBuff[iRoll]     = -0.1 *10


    if(i_episode % 80 == 0):
        print(stateBuff[1], myAction, eps_threshold,policy_net(myState))
##        print(stateBuff)

    dicTensor = torch.tensor([1.0*myAction])
  

    # !!!!! HERE WE CALL THE SIMULATION !!!!!
    stateBuff = stateBuff.numpy()
    flight.flyAndShootTargets( controlBuff, stateBuff, targetHits, nsub=20, dt=0.005 )
    stateBuff = torch.from_numpy(stateBuff)
    
    iframe+=1

    #---------- End Step ---------------------------------------

    reward = 1.0
    rewardSum += reward
    reward = torch.tensor([reward])

    if stateBuff[1] >= 0.0 and stateBuff[1] < 50.0:
      next_state = myState# Must define get state
    else: 
      done=True
#      reward = -1.0

      reward = torch.tensor([reward])
      next_state = None

    if t==10000: 
      print("CONGRATULATIONS!!!!! YOU MADE IT!!!!!!!!!")
      done=True
      next_state = None

    #Store the transition in memory
    memory.push(myState, myAction, next_state, reward)
    #Move to the next state
    state = next_state
    # Perform one step optimization on the target network
    optimizer_model()
    if done:
      if(i_episode % 10 ==0):
        print( "Surive time: ", t, rewardSum)
###      plot_durations()
##      exit()
      break
    # Update the target network
  if i_episode % TARGET_UPDATE == 0:
    torch.save(policy_net.state_dict(), 'testNet')

    target_net.load_state_dict(policy_net.state_dict())
#======================================================================

print ('Mission Complete . . .')
