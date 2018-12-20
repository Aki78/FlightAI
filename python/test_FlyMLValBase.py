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
def printFormatedState(stateBuff):
    print "position:          ", stateBuff[ 0: 3]
    print "velocity:          ", stateBuff[ 3: 6]
    print "forward direction: ", stateBuff[ 6: 9]
    print "up direction:      ", stateBuff[ 9:12]
    print "angular velocity:  ", stateBuff[12:15]
    print "force:             ", stateBuff[15:18]
    print "torque:            ", stateBuff[18:21]
    print "pitch: ",stateBuff[21], " yaw: ",stateBuff[22], " roll : ",stateBuff[23], " throttle: ",stateBuff[24]
#fig = plt.gcf()
#fig.show()
#fig.canvas.draw()
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
    self.relu1 = nn.LeakyReLU()
    self.relu2 = nn.LeakyReLU()
    self.relu3 = nn.LeakyReLU()
#    self.relu4 = nn.LeakyReLU()
#    self.relu5 = nn.LeakyReLU()
#    self.relu6 = nn.LeakyReLU()
#    self.relu7 = nn.LeakyReLU()
#    self.relu8 = nn.LeakyReLU()
#    self.relu9 = nn.LeakyReLU()
    self.fc2 = nn.Linear(hidSize, numClass)
  def forward(self,x):
   x = x.type(torch.FloatTensor)
   out = self.fc1(x)
   out = self.relu1(out)
   out = self.relu2(out)
   out = self.relu3(out)
#   out = self.relu4(out)
#   out = self.relu5(out)
#   out = self.relu6(out)
#   out = self.relu7(out)
#   out = self.relu8(out)
#   out = self.relu9(out)
   out = self.fc2(out)
   return out

#------- Training Setup--------------------------------------
BATCH_SIZE = 10000
GAMMA = 0.9
EPS_START = 0.999
EPS_END = 0.001
EPS_DECAY = 200000
TARGET_UPDATE = 10

policy_net_pitch = DQN(26,10,3)#.to(device) 
policy_net_yaw   = DQN(26,10,3)#.to(device) 
policy_net_roll  = DQN(26,10,3)#.to(device) 
#policy_net_throt = DQN(26,10,3)#.to(device) 

target_net_pitch = DQN(26,10,3)#.to(device) 
target_net_yaw   = DQN(26,10,3)#.to(device) 
target_net_roll  = DQN(26,10,3)#.to(device) 
#target_net_trhot = DQN(30,10,3)#.to(device) 

target_net_pitch.load_state_dict(policy_net_pitch.state_dict()) 
target_net_yaw.load_state_dict(policy_net_yaw.state_dict()) 
target_net_roll.load_state_dict(policy_net_roll.state_dict()) 
#target_net_trhot.load_state_dict(policy_net_throt.state_dict()) 

target_net_pitch.eval()
target_net_yaw.eval()
target_net_roll.eval()
#target_net_trhot.eval()

optimizer_pitch = optim.RMSprop(policy_net_pitch.parameters())
optimizer_yaw = optim.RMSprop(policy_net_yaw.parameters())
optimizer_roll = optim.RMSprop(policy_net_roll.parameters())
#optimizer_trhot = optim.RMSprop(policy_net_throt.parameters())

memory = ReplayMemory(1000000)

steps_done = 0
eps_threshold = 0
def select_action(state, policy_net):
  global steps_done, eps_threshold
  sample = random.random()
#  eps_threshold = EPS_END + (EPS_START - EPS_END)*math.exp(-steps_done/EPS_DECAY)
  eps_threshold =  0.2*math.exp(-1.0*steps_done/(1.0*EPS_DECAY))
  if sample > eps_threshold:
    with torch.no_grad():
      return policy_net(state).max(-1)[1].view(1,1) 
##      return policy_net(state).min(-1)[1].view(1,1) 
  else:
    return torch.tensor([[random.randrange(3)]], dtype=torch.long)


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

  action_batch_pitch = torch.stack(batch.action1)
  action_batch_yaw = torch.stack(batch.action2)
  action_batch_roll = torch.stack(batch.action3)
#  action_batch_throt = torch.stack(batch.action4)

  reward_batch = torch.cat(batch.reward)
  steps_done += 1

  state_action_values_pitch = policy_net_pitch(state_batch).gather(1, action_batch_pitch.squeeze().unsqueeze(1))
  state_action_values_yaw = policy_net_yaw(state_batch).gather(1, action_batch_yaw.squeeze().unsqueeze(1))
  state_action_values_roll = policy_net_roll(state_batch).gather(1, action_batch_roll.squeeze().unsqueeze(1))
#  state_action_values_throt = policy_net_throt(state_batch).gather(1, action_batch_throt.squeeze().unsqueeze(1))

  #Compute V(s_{t+1}) for all next states.
  next_state_values_pitch = torch.zeros(BATCH_SIZE)
  next_state_values_yaw = torch.zeros(BATCH_SIZE)
  next_state_values_roll = torch.zeros(BATCH_SIZE)
#  next_state_values_throt = torch.zeros(BATCH_SIZE)
  next_state_values_pitch[non_final_mask] = target_net_pitch(non_final_next_states).max(-1)[0].detach()
  next_state_values_yaw[non_final_mask] = target_net_yaw(non_final_next_states).max(-1)[0].detach()
  next_state_values_roll[non_final_mask] = target_net_roll(non_final_next_states).max(-1)[0].detach()
#  next_state_values_throt[non_final_mask] = target_net_trhot(non_final_next_states).max(-1)[0].detach()
  #Compute the expected Q values
  expected_state_action_values_pitch = (next_state_values_pitch*GAMMA) + reward_batch
#  
  expected_state_action_values_yaw = (next_state_values_yaw*GAMMA) + reward_batch
  expected_state_action_values_roll = (next_state_values_roll*GAMMA) + reward_batch
#  expected_state_action_values_throt = (next_state_values_throt*GAMMA) + reward_batch
  #Compute Huberr loss
##  loss1 = F.smooth_l1_loss(state_action_values_pitch, expected_state_action_values_pitch.unsqueeze(1))
  loss1 = F.mse_loss(state_action_values_pitch, expected_state_action_values_pitch.unsqueeze(1))
  if(myTime % 200 ==0):
      print(loss1,np.array(list(zip(state_action_values_pitch[:].detach().numpy(), expected_state_action_values_pitch.unsqueeze(1)[:].detach().numpy())))[:])
##  loss2 = F.smooth_l1_loss(state_action_values_yaw, expected_state_action_values_yaw.unsqueeze(1))
  loss2 = F.mse_loss(state_action_values_yaw, expected_state_action_values_yaw.unsqueeze(1))
##  loss3 = F.smooth_l1_loss(state_action_values_roll, expected_state_action_values_roll.unsqueeze(1))
  loss3 = F.mse_loss(state_action_values_roll, expected_state_action_values_roll.unsqueeze(1))
##  loss4 = F.smooth_l1_loss(state_action_values_throt, expected_state_action_values_throt.unsqueeze(1))
#  loss4 = F.mse_loss(state_action_values_throt, expected_state_action_values_throt.unsqueeze(1))
  #optimize the model
  optimizer_pitch.zero_grad()
  optimizer_yaw.zero_grad()
  optimizer_roll.zero_grad()
#  optimizer_trhot.zero_grad()
  loss1.backward()
  loss2.backward()
  loss3.backward()
#  loss4.backward()
  for param in policy_net_pitch.parameters():
    param.grad.data.clamp_(-1,1)
  for param in policy_net_yaw.parameters():
    param.grad.data.clamp_(-1,1)
  for param in policy_net_roll.parameters():
    param.grad.data.clamp_(-1,1)
#  for param in policy_net_throt.parameters():
#    param.grad.data.clamp_(-1,1)

  optimizer_pitch.step()
  optimizer_pitch.step()
  optimizer_pitch.step()
  optimizer_yaw.step()
  optimizer_yaw.step()
  optimizer_yaw.step()
  optimizer_roll.step()
  optimizer_roll.step()
  optimizer_roll.step()
#  optimizer_trhot.step()
#  optimizer_trhot.step()
#  optimizer_trhot.step()
#  optimizer_trhot.step()
#  optimizer_trhot.step()

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
  print(i_episode)
  global myTime
  myTime = i_episode
  rewardSum = 0
  dicTensor = torch.tensor([1.0,1.0,1.0]).double()
  randVal1 = random.uniform(3,47)
  randVal2 = random.uniform(0,100)
  randVal3 = random.uniform(-100,100)
  randVal4 = random.uniform(1,100)
  flight.setPose(np.array([0.0,randVal1 ,0.0]),np.array([randVal2,randVal3,randVal4]) , np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]))
#  flight.setPose(np.array([0.0,random.uniform(0,200),0.0]), np.array([random.uniform(0,100),random.uniform(-100,100),random.uniform(0,200)]) , np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]))

#  flight.setTilt( 2, 0.02 ) # set Angle-of-incidence for wing #2 ( = elevator)
  flight.setTilt( 0, 0.00 ) # set Angle-of-incidence for wing #2 ( = elevator)
  flight.setTilt( 1, 0.00 ) 
  flight.setTilt( 2, 0.00 ) 
  flight.setTilt( 3, 0.00 ) 
  ROCX = 0.0
  ROCY = 0.0
  ROCZ = 0.0
  XBefore=randVal2
  YBefore=randVal3
  ZBefore=randVal4
##  stateBuff = Variable(torch.from_numpy(np.zeros(4*3 + 4)))             # C->python [ pos.xyz, vel.xyz, forward.xyz, up.xyz       pich,yaw,roll,throttle  ]
#  stateBuff = Variable(torch.from_numpy(np.zeros(8*3 + 4)))             # C->python [ pos.xyz, vel.xyz, forward.xyz, up.xyz       pich,yaw,roll,throttle  ]
  stateBuff = Variable(torch.from_numpy(np.zeros(25)))             
  stateBuff = stateBuff.numpy()
  flight.flyAndShootTargets( controlBuff, stateBuff, targetHits, nsub=10, dt=0.001 )
  stateBuff = torch.from_numpy(stateBuff)
  sumRward = 0
  state = []
  iframe = 0
  currentPitch = 0.0
  currentYaw = 0.0
  currentRoll = 0.0
  currentThrot = 0.0
  for t in count():
    XBefore =  stateBuff.numpy()[0] 
    YBefore =  stateBuff.numpy()[1] 
    ZBefore =  stateBuff.numpy()[2] 

    done=False
    #---------- Start Step ---------------------------------------
    if tragetMoveSpeed > 0.0:
        phi = 0.01*iframe + np.array( range(len(targets)) )
        targets[:,2] += np.cos(phi)*tragetMoveSpeed
    # set controls by keyboard
    controlBuff[:]          =  0.0   # 0.0 means not change - preserve state of controls from previous step
#    myState = torch.cat((torch.tensor([stateBuff[i] for i in (1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27)]), dicTensor.double()))
    myState = torch.cat((torch.tensor([stateBuff[i] for i in (1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24)]), dicTensor.double()))
    myState[13:19] = myState[13:19]*0.1
    myState = 0.1*myState
#    print(myState)
#    myState = torch.tensor([stateBuff[i] for i in (1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27)])
    myActionPitch = select_action(myState , policy_net_pitch)
    if   myActionPitch == 2: 
        controlBuff[iPitch] = +1.0    # move elevator up,    value +1.0 is maximal rate of change
    elif myActionPitch == 0: 
        controlBuff[iPitch] = -1.0    # move elevator down
    else:
        controlBuff[iPitch] = 0.0

    myActionYaw = select_action(myState , policy_net_yaw)
    if   myActionYaw == 0: 
        controlBuff[iYaw]   = -1.0    # move rudder left
    elif   myActionYaw == 2: 
        controlBuff[iYaw]   = +1.0    # move rudder right
    else:
        controlBuff[iYaw]   = 0.0    # move rudder right

#    myActionThrot = select_action(myState , policy_net_throt)
#    if myActionThrot == 2:
#        controlBuff[iThrottle] = +1.0  # increase engine power
#    elif myActionThrot == 0:
#        controlBuff[iThrottle] = -1.0  # decrease engine power
#    else:
#        controlBuff[iThrottle] = 0.0
    myActionThrot = 2
    controlBuff[iThrottle] = +1.0

    myActionRoll = select_action(myState , policy_net_roll)
    if   myActionRoll == 2: 
        controlBuff[iRoll]  = +1.0    # move ailerons left
    elif   myActionRoll == 0: 
        controlBuff[iRoll]  = -1.0    # move ailerons right
    else:
        controlBuff[iRoll ] = stateBuff[21+iRoll]*-10.0    # dRoll/dt =  roll * -10
        #controlBuff[iRoll] = 0.0

    if(i_episode % 10 == 0):
        print('pit',stateBuff[1].detach().numpy(), myActionPitch.numpy()[0][0],myActionYaw.numpy()[0][0],myActionRoll.numpy()[0][0], eps_threshold,policy_net_pitch(myState).detach().numpy())
    elif(i_episode % 10 == 1):
        print('yaw',stateBuff[1].detach().numpy(), myActionPitch.numpy()[0][0],myActionYaw.numpy()[0][0],myActionRoll.numpy()[0][0], eps_threshold,policy_net_yaw(myState).detach().numpy())
    elif(i_episode % 10 == 2):
        print('roll',stateBuff[1].detach().numpy(), myActionPitch.numpy()[0][0],myActionYaw.numpy()[0][0],myActionRoll.numpy()[0][0], eps_threshold,policy_net_roll(myState).detach().numpy())
##        print(stateBuff)

##    dicTensor = torch.tensor([1.0*myActionPitch,1.0*myActionYaw, 1.0*myActionRoll, 1.0*myActionThrot])
  

    # !!!!! HERE WE CALL THE SIMULATION !!!!!
    stateBuff = stateBuff.data.numpy()
    flight.flyAndShootTargets( controlBuff, stateBuff, targetHits, nsub=5, dt=0.005 )
    ROCX = XBefore - stateBuff[0]
    ROCY = YBefore - stateBuff[1]
    ROCZ = ZBefore - stateBuff[2]
 #   print(ROCX, ROCY,ROCZ)
    stateBuff = torch.from_numpy(stateBuff)
    
    dicTensor = torch.tensor([ROCX,ROCY,ROCZ])

    iframe+=1

    #---------- End Step ---------------------------------------

    reward =  -np.abs(25 - stateBuff[1].numpy())  - 100.0*ROCY*ROCY 
    print(reward, stateBuff[1].numpy())
    rewardSum += reward
#    reward = torch.tensor([reward])

    if stateBuff[1] >= 0.0 and stateBuff[1] < 50.0:
      next_state = myState# Must define get state
    else: 
      done=True
      reward -= 100.0

#      reward = torch.tensor([reward])
      next_state = None

    reward = torch.tensor([reward])
    if t==10000: 
      print("CONGRATULATIONS!!!!! YOU MADE IT!!!!!!!!!")
      done=True
      next_state = None

    #Store the transition in memory
    memory.push(myState, myActionPitch,myActionYaw,myActionRoll,myActionThrot, next_state, reward)
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
    torch.save(policy_net_pitch.state_dict(), 'testNetPitch')
    torch.save(policy_net_yaw.state_dict(), 'testNetYaw')
    torch.save(policy_net_roll.state_dict(), 'testNetRoll')
#    torch.save(policy_net_throt.state_dict(), 'testNetThrot')

    target_net_pitch.load_state_dict(policy_net_pitch.state_dict())
    target_net_yaw.load_state_dict(policy_net_yaw.state_dict())
    target_net_roll.load_state_dict(policy_net_roll.state_dict())
#    target_net_trhot.load_state_dict(policy_net_throt.state_dict())
#======================================================================

print ('Mission Complete . . .')
