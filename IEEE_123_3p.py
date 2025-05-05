import numpy as np
from numpy import linalg as LA
import gym
import os
import random
import sys
from gym import spaces
from gym.utils import seeding
import copy
import matplotlib.pyplot as plt

from scipy.io import loadmat
import pandapower as pp
import pandapower.networks as pn
import pandas as pd 
import math

from dssdata import SystemClass
from dssdata.pfmodes import run_static_pf
from dssdata.tools import voltages
from dssdata.pfmodes import cfg_tspf

DSS_PATH = "opendss_model/123bus/IEEE123Master.dss"

def create_123bus3p(injection_bus):
    #build the generators
    distSys = SystemClass(path=DSS_PATH, kV=[4.16, 0.48])
    cfg_tspf(distSys,'0.02s')
    injection_bus_dict = dict()
    cmd = []
    for idx in injection_bus:
        v_i = voltages.get_from_buses(distSys,[str(idx)])
        phase_i = v_i['phase'][0]
        injection_bus_dict[str(idx)]=phase_i
        distSys.dss.Circuit.SetActiveBus(str(idx))
        kv_base = distSys.dss.Bus.kVBase()
        for phase in phase_i:
            if phase == 'a':
                cmd.append(f"New Generator.bus{idx}_1 bus1={idx}.1 Phases=1 kv={kv_base} kw=0 kvar=0 pf=1 model=1")
            elif phase == 'b':
                cmd.append(f"New Generator.bus{idx}_2 bus1={idx}.2 Phases=1 kv={kv_base} kw=0 kvar=0 pf=1 model=1")
            elif phase == 'c':
                cmd.append(f"New Generator.bus{idx}_3 bus1={idx}.3 Phases=1 kv={kv_base} kw=0 kvar=0 pf=1 model=1")
    distSys.dsscontent = distSys.dsscontent + cmd
    return distSys, injection_bus_dict

class IEEE123bus3p(gym.Env):
    def __init__(self, distSys, injection_bus_dict, v0=1, vmax=1.05, vmin=0.95):
        self.network =  distSys
        self.obs_dim = 14
        self.action_dim = 14
        self.injection_bus = injection_bus_dict
        self.injection_bus_str = list(injection_bus_dict.keys())
        self.agentnum = len(self.injection_bus)
        
        self.v0 = v0 
        self.vmax = vmax
        self.vmin = vmin
        
        self.state = np.ones((self.agentnum, 3))
    def get_state(self):
        v_pu = voltages.get_from_buses(self.network, self.injection_bus_str)
        state_a = v_pu['v_pu_a'].to_numpy().reshape(-1,1)
        state_b = v_pu['v_pu_b'].to_numpy().reshape(-1,1)
        state_c = v_pu['v_pu_c'].to_numpy().reshape(-1,1)
        self.state = np.hstack([state_a, state_b, state_c]) #shape: number_of_bus*3
        self.state[np.isnan(self.state)]=1.0
        # print(self.state[-1,[1,2]])
        return self.state
    
    def step_Preward(self, action, p_action): 
        
        done = False 
        #safe-ddpg reward
        reward = float(-1.0*LA.norm(p_action,1)-1000*LA.norm(np.clip(self.state-self.vmax, 0, np.inf),2)**2
                       -1000*LA.norm(np.clip(self.vmin-self.state, 0, np.inf),2)**2)
        # local reward
        agent_num = len(self.injection_bus)
        reward_sep = np.zeros(agent_num, )
        #just for ddpg
        p_action = np.array(p_action)
        eta_2 = 300
        eta_1 = 1 #1
        for i in range(agent_num):
            for j in range(3):
                if self.state[i,j]<0.95:
                    reward_sep[i] +=float(-eta_1*LA.norm([p_action[i,j]],1)-eta_2*LA.norm(np.clip([self.state[i,j]-self.vmax], 0, np.inf),2)
                    -eta_2*LA.norm(np.clip([self.vmin-self.state[i,j]], 0, np.inf),2))  
                elif self.state[i,j]>1.05:
                    reward_sep[i] +=float(-eta_1*LA.norm([p_action[i,j]],1)-eta_2*LA.norm(np.clip([self.state[i,j]-self.vmax], 0, np.inf),2)
                    -eta_2*LA.norm(np.clip([self.vmin-self.state[i,j]], 0, np.inf),2))  

            for j in range(3):
                if self.state[i,j]<0.93:
                    reward_sep[i] +=float(-eta_1*LA.norm([p_action[i,j]],1))*10
                elif self.state[i,j]>1.07:
                    reward_sep[i] +=float(-eta_1*LA.norm([p_action[i,j]],1))*10
                           
        action = action * 100 # convert unit to kVar
        for i, idx in enumerate(self.injection_bus_str):
            for phase in self.injection_bus[idx]:
                if phase == 'a':
                    self.network.run_command(f"Generator.bus{idx}_1.kvar={action[i,0]}") 
                elif phase == 'b':
                    self.network.run_command(f"Generator.bus{idx}_2.kvar={action[i,1]}") 
                elif phase == 'c':
                    self.network.run_command(f"Generator.bus{idx}_3.kvar={action[i,2]}") 
        self.network.dss.Solution.Number(1)
        self.network.dss.Solution.Solve()
        self.state=self.get_state()
        
        if(np.min(self.state) > 0.95 and np.max(self.state)< 1.05):
            done = True
            reward_sep += 100
        # if done:
        #     print('successful!')
        return self.state, reward, reward_sep, done
    
    def init_pw(self, idx,a_l,a_u,b_l,b_u,c_l,c_u):
        bus_a_kw = 100*np.random.uniform(a_l, a_u)
        bus_b_kw = 100*np.random.uniform(b_l, b_u)
        bus_c_kw = 100*np.random.uniform(c_l, c_u)
        for phase in self.injection_bus[idx]:
            if phase == 'a':
                self.network.run_command(f"Generator.bus{idx}_1.kw={bus_a_kw}") 
            elif phase == 'b':
                self.network.run_command(f"Generator.bus{idx}_2.kw={bus_b_kw}") 
            elif phase == 'c':
                self.network.run_command(f"Generator.bus{idx}_3.kw={bus_c_kw}") 


    def reset(self, seed=1): #sample different initial volateg conditions during training
        np.random.seed(seed)
        senario = np.random.choice([0,1])
        self.network.init_sys()
        if(senario == 0):
            # Low voltage
            self.init_pw('10',-4,0,0,0,0,0)
            self.init_pw('11',-4,0,0,0,0,0)
            self.init_pw('16',0,0,0,0,-4,0)
            self.init_pw('20',0,2,0,0,0,0)
            self.init_pw('33',-4,0,0,0,0,0)
            self.init_pw('36',-4,2,-3,2,0,0)
            self.init_pw('48',-5,-2,-8,-5,-8,-5)
            self.init_pw('59',0,0,-10,-5,0,0)
            self.init_pw('66',-5,-2,-5,0,-3,0)
            self.init_pw('75',0,0,0,0,-3,0)
            self.init_pw('83',-5,-1,-5,-1,-5,-1)
            self.init_pw('92',0,0,0,0,-5,-1)
            self.init_pw('104',0,0,0,0,-3,0)
            self.init_pw('61',-5,-1,-5,0,-5,0)
        if(senario == 1):
            # High voltage
            self.init_pw('10',5,8,0,0,0,0)
            self.init_pw('11',5,8,0,0,0,0)
            self.init_pw('16',0,0,0,0,5,8)
            self.init_pw('20',1,5,0,0,0,0)
            self.init_pw('33',5,11,0,0,0,0)
            self.init_pw('36',3,5,5,10,3,5)
            self.init_pw('48',3,5,5,10,5,10)
            self.init_pw('59',3,5,5,10,3,5)
            self.init_pw('66',5,11,5,10,5,10)
            self.init_pw('75',5,11,5,10,1,5)
            self.init_pw('83',5,10,5,10,1,5)
            self.init_pw('92',5,10,5,10,5,10)
            self.init_pw('104',1,5,1,5,1,5)
            self.init_pw('61',1,5,1,5,1,5)
        self.network.dss.Solution.Number(1)
        self.network.dss.Solution.Solve()      

        self.state=self.get_state()
        return self.state
    
if __name__ == "__main__":
    injection_bus = np.array([10, 11, 16, 20, 33, 36, 48, 59, 66, 75, 83, 92, 104, 61])
    net, injection_bus_dict = create_123bus3p(injection_bus)    
    env = IEEE123bus3p(net, injection_bus_dict)
    state_list = []
    for i in range(100):
        state = env.reset(i)
        state_list.append(state)
    state_list = np.array(state_list)
    # print(state_list.shape)
    fig, axs = plt.subplots(len(injection_bus), 3, figsize=(3,11))
    for i in range(3):
        for j in range(len(injection_bus)):
            axs[j,i].hist(state_list[:,j,i])
            # axs[j,i].hist(state_list[:,j,i],[1.0,1.05,1.10,1.15,1.20,1.25])
    plt.tight_layout()
    plt.show()
    plt.savefig('hist.png')
