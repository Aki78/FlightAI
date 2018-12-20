#!/usr/bin/python
#---------general stuff import -----------------
import numpy as np
import math
import numpy as np
import matplotlib
import random
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
#---------torch stuff import -----------------
import torch
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
#------- Replay Memory --------------------------------------
Transition = namedtuple('Transition', ('state', 'action1', 'action2', 'action3','action4', 'next_state', 'reward'))
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
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.fc2 = nn.Linear(hidSize, numClass)
  def forward(self,x):
   out = self.fc1(x)
   out = self.relu1(out)
   out = self.relu2(out)
   out = self.fc2(out)
   return out

#------- Training Setup--------------------------------------
BATCH_SIZE = 64
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

policy_net_pitch = DQN(10,50,32)#.to(device) 
policy_net_yaw   = DQN(10,50,32)#.to(device) 
policy_net_roll  = DQN(10,50,32)#.to(device) 
policy_net_throt = DQN(10,50,32)#.to(device) 

target_net_pitch = DQN(10,50,32)#.to(device) 
target_net_yaw   = DQN(10,50,32)#.to(device) 
target_net_roll  = DQN(10,50,32)#.to(device) 
target_net_trhot = DQN(10,50,32)#.to(device) 

target_net_pitch.load_state_dict(policy_net.state_dict()) 
target_net_yaw.load_state_dict(policy_net.state_dict()) 
target_net_roll.load_state_dict(policy_net.state_dict()) 
target_net_trhot.load_state_dict(policy_net.state_dict()) 

target_net_pitch.eval()
target_net_yaw.eval()
target_net_roll.eval()
target_net_trhot.eval()

optimizer_pitch = optim.RMSprop(policy_net_pitch.parameters())
optimizer_yaw = optim.RMSprop(policy_net_yaw.parameters())
optimizer_roll = optim.RMSprop(policy_net_roll.parameters())
optimizer_trhot = optim.RMSprop(policy_net_throt.parameters())

memory = ReplayMemory(10000)

partial_steps_done = 0

def select_action(state, policy_net):
  global steps_done
  sample = random.random()
  eps_threshold = EPS_END + (EPS_START - EPS_END)*math.exp(-steps_done/EPS_DECAY)
  partial_steps_done += 1
  if sample > eps_threshold:
    with torch.no_grad():
      return policy_net(state).max(1)[1].view(1,1) 
  else:
#    return torch.tensor([[random.randrange(3)]], device=device, dtype=torch.long)
    return torch.tensor([[random.randrange(3)]], dtype=torch.long)

eps_durations = []

#------- Set Optimizer --------------------------------------
def optimizer_model():
  if len(memory) < BATCH_SIZE:
    return
  transitions = memory.sample(BATCH_SIZE)
  batch = Transition(*zip(*transitions))
  non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.uint8)
  non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

  state_batch = torch.cat(batch.state)

  action_batch_pitch = torch.cat(batch.action1)
  action_batch_yaw = torch.cat(batch.action2)
  action_batch_roll = torch.cat(batch.action3)
  action_batch_throt = torch.cat(batch.action4)

  reward_batch = torch.cat(batch.reward)
  # Compute Q(s_t, a), then select the columns of actions taken
  state_action_values_pitch = policy_net(state_batch).gather(1, action_batch_pitch)
  state_action_values_yaw = policy_net(state_batch).gather(1, action_batch_yaw)
  state_action_values_roll = policy_net(state_batch).gather(1, action_batch_roll)
  state_action_values_throt = policy_net(state_batch).gather(1, action_batch_throt)
  #Compute V(s_{t+1}) for all next states.
  next_state_values = torch.zeros(BATCH_SIZE, device=device)
  next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
  #Compute the expected Q values
  expected_state_action_values = (next_state_values*GAMMA) + reward_batch
  #Compute Huberr loss
  loss1 = F.smooth_l1_loss(state_action_values_pitch, expected_state_action_values.unsqueeze(1))
  loss2 = F.smooth_l1_loss(state_action_values_yaw, expected_state_action_values.unsqueeze(1))
  loss3 = F.smooth_l1_loss(state_action_values_roll, expected_state_action_values.unsqueeze(1))
  loss4 = F.smooth_l1_loss(state_action_values_throt, expected_state_action_values.unsqueeze(1))
  #optimize the model
  optimizer_pitch.zero_grad()
  optimizer_yaw.zero_grad()
  optimizer_roll.zero_grad()
  optimizer_trhot.zero_grad()
  loss1.backward()
  loss2.backward()
  loss3.backward()
  loss4.backward()
  for param in policy_net_pitch.parameters():
    param.grad.data.clamp_(-1,1)
    print(param.grad.data.clamp(-1,1))
  for param in policy_net_yaw.parameters():
    param.grad.data.clamp_(-1,1)
    print(param.grad.data.clamp(-1,1))
  for param in policy_net_roll.parameters():
    param.grad.data.clamp_(-1,1)
    print(param.grad.data.clamp(-1,1))
  for param in policy_net_throt.parameters():
    param.grad.data.clamp_(-1,1)
    print(param.grad.data.clamp(-1,1))

  optimizer_pitch.step()
  optimizer_yaw.step()
  optimizer_roll.step()
  optimizer_trhot.step()

#-- Start Flight Engine ----------------------------------------------------- 
# --- set workdir with configuration of aircraft, textures etc.
flight.loadFromFile("/home/aki/Dropbox/SimpleSimulationEngine/cpp/apps/AeroCombat/data/AeroCraftMainWing1.ini" )

# --- initialize visualization (optional, just for debugging)
fview = flight.FlightView("/home/aki/Dropbox/SimpleSimulationEngine/")

# --- set initial state of aircraft   setPose( position, velocity, rotMat==[left,up,forward] )
flight.setPose( np.array([0.0,200.0,0.0]), np.array([0.0,0.0,100.0]) , np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]) )
flight.setTilt( 2, 0.02 ) # set Angle-of-incidence for wing #2 ( = elevator)
#flight.setTilt( 0,  0.1 )
#flight.setTilt( 1, -0.1 )
#flight.setTilt( 3, 0.1 )

# --- initialize targets Spheres (x,y,z,Radius)
targets = np.random.random((30,4));
###targets[:, 0] += -0.5; targets[:, 0] *= 10000;  # x
###targets[:, 2] += -0.5; targets[:, 2] *= 10000;  # z
###targets[:, 1] *= 200;   # y=height
######targets[:, 3] += 50.0   # radius
#print targets
flight.setTargets( targets )

# ---  initialize C-interface buffers
iPitch=0; iYaw=1; iRoll=2; iThrottle=3; iTrigger=4
controlBuff  = np.zeros(5)                   # python->C [pich,yaw,roll,throttle,trigger] differeces (rate-of-change)
stateBuff    = np.zeros(4*3 + 4)             # C->python [ pos.xyz, vel.xyz, forward.xyz, up.xyz       pich,yaw,roll,throttle  ]
targetHits   = np.zeros( (len(targets),2) )  # C->python if hit [ageMin,ageMax] of projectiles which hit it, if no-hit [+1e+300,-1e+300]

iframe = 0

tragetMoveSpeed = 10.0
#------ Run ML Loop --------------------------------------
##while GO_ON: # repeat until pressed key "c"
num_episodes = 1000

controllElev = torch.tensor(random.randrange[3])
controllRudd = torch.tensor(random.randrange[3])
controllAile = torch.tensor(random.randrange[3])
controllEngi = torch.tensor(random.randrange[3])

#======================================================================
for i_episode in range(num_episodes):
  state = []

  for t in count():
    action = select_action(state)
#    _,reward, done, _ =  get_reward()   # env.step(action.item())
    reword = 0.001
    reward = torch.tensor([reward], device=device)
    #Observe new state
###    if not done:
    if state[1] >= 0.0 or state[1] < 500
      next_state = get_state() # Must define get state
    else: 
      reward = -100
      reward = torch.tensor([reward], device=device)
      next_state = None

    #Store the transition in memory
    memory.push(state, action, next_state, reward)
    #Move to the next state
    state = next_state
    # Perform one step optimization on the target network
    optimizer_model()
    if done:
      eps_durations.append(t+1)
###      plot_durations()
      break
    # Update the target network
  if i_episode % TARGET_UPDATE == 0:
    target_net.load_state_dict(policy_net.state_dict())
#======================================================================

    if tragetMoveSpeed > 0.0:
        #targets[:,0] += ( np.random.random(len(targets)) - 0.5 )*tragetDiffuseSpeed;
        #targets[:,2] += ( np.random.random(len(targets)) - 0.5 )*tragetDiffuseSpeed;
        phi = 0.01*iframe + np.array( range(len(targets)) )
##        targets[:,0] += np.sin(phi)*tragetMoveSpeed
        targets[:,2] += np.cos(phi)*tragetMoveSpeed
    #print keyDict
    #print [k for k, v in keyDict.items() if v]
    print("# frame ", iframe)
    # set controls by keyboard

    controlBuff[:]          =  0.0   # 0.0 means not change - preserve state of controls from previous step

    if   controllElev.max == 2: 
        controlBuff[iPitch] = +1.0    # move elevator up,    value +1.0 is maximal rate of change
    elif controllElev.max == 0: 
        controlBuff[iPitch] = -1.0    # move elevator down
    else:
        controlBuff[iPitch] = 0.0

    if   controllAile.max == 2: 
        controlBuff[iRoll]  = +1.0    # move ailerons left
    elif   controllAile.max == 0: 
        controlBuff[iRoll]  = -1.0    # move ailerons right
    else:
        controlBuff[iRoll] = 0.0

    if   controllRudd.max == 0: 
        controlBuff[iYaw]   = -1.0    # move rudder left
    elif   controllRudd.max == 2: 
        controlBuff[iYaw]   = +1.0    # move rudder right
    else:

        controlBuff[iYaw]   = 0.0    # move rudder right
    if controllEngi.max == 2:
        controlBuff[iThrottle] = +1.0  # increase engine power
    elif controllEngi.max == 0:
        controlBuff[iThrottle] = -1.0  # decrease engine power
    else:
        controlBuff[iThrottle] = 0.0

    #elif keyDict[keyboard.Key.space]: 
##    elif keyDict[" "]:
##        #print "python shoot"
##        controlBuff[iTrigger ] = 1.0  # shoot when > 0.5  
##    if not ( keyDict["a"] or keyDict["d"] ):               # if roll keys not active, retract ailerons to neutral position
    if not ( controlAile.max == 1):               # if roll keys not active, retract ailerons to neutral position
        controlBuff[iRoll] = stateBuff[12+iRoll]*(-10.0)    # dRoll/dt =  roll * -10
    #if(niter==0):
    #    controlBuff[iTrigger] = 1
    #flight.fly( poss, vels, rots, nsub=10, dt=0.001 )
    
    # !!!!! HERE WE CALL THE SIMULATION !!!!!
    flight.flyAndShootTargets( controlBuff, stateBuff, targetHits, nsub=10, dt=0.003 )
    #print "control state ", stateBuff[12:]
    print(stateBuff)
    
##    if( stateBuff[1] < 0.0 ):
##        print(" pos.y < 0.0  =>  YOU CRASHED TO GROUND !!!")
##        exit()
    
    # write out which targets where hit by how old projectiles ( age of projectiles is useful for feedback propagation to history ) 
    for i in xrange(len(targetHits)):
        if targetHits[i,0]<1e+200:
            print("hit target ", i," by projectile of age interval" , targetHits[i,:])
    
    # visualize - ( optional, just for debugging )
    fview.draw()
    time.sleep(0.05)  # delay for visualization - may be removed for actual training
    iframe+=1
    #if iframe > 500:
    #    exit()
print ('Mission Complete . . .')
