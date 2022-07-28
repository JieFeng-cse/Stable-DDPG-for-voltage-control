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

DSS_PATH = "opendss_model/13bus/IEEE13Nodeckt.dss"

def create_13bus3p(injection_bus):
    #build the generators
    distSys = SystemClass(path=DSS_PATH, kV=[115, 4.16, 0.48])
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

class IEEE13bus3p(gym.Env):
    def __init__(self, distSys, injection_bus_dict, v0=1, vmax=1.05, vmin=0.95):
        self.network =  distSys
        self.obs_dim = 3
        self.action_dim = 3
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
        return self.state
    
    def step_Preward(self, action, p_action): 
        
        done = False 
        #global reward
        reward = float(-1.0*LA.norm(p_action,1)-1000*LA.norm(np.clip(self.state-self.vmax, 0, np.inf),2)**2
                       -1000*LA.norm(np.clip(self.vmin-self.state, 0, np.inf),2)**2)
        # local reward
        agent_num = len(self.injection_bus)
        reward_sep = np.zeros(agent_num, )
        p_action = np.array(p_action)
        for i in range(agent_num):
            for j in range(3):
                if self.state[i,j]<0.95: #ddpg may have really large action, so here we use 50 for \eta_1, for safe-DDPG, we use 1
                    reward_sep[i] +=float(-50.0*LA.norm([p_action[i,j]],1)-1000*LA.norm(np.clip([self.state[i,j]-self.vmax], 0, np.inf),2)**2
                    -1000*LA.norm(np.clip([self.vmin-self.state[i,j]], 0, np.inf),2)**2)  
                elif self.state[i,j]>1.05:
                    reward_sep[i] +=float(-50.0*LA.norm([p_action[i,j]],1)-1000*LA.norm(np.clip([self.state[i,j]-self.vmax], 0, np.inf),2)**2
                    -1000*LA.norm(np.clip([self.vmin-self.state[i,j]], 0, np.inf),2)**2)  
                           
        #safe: -1.0*LA.norm(p_action[i],1) 
        # state-transition dynamics
        action = action * 100 #from kVar to MVar
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
        return self.state, reward, reward_sep, done
    
    def reset_3b(self, seed=1): #only 3 buses are included
        np.random.seed(seed)
        senario = np.random.choice([0,1])
        # senario = 1
        self.network.init_sys()
        if(senario == 0):
            # Low voltage
            bus675_a_kw = -100*np.random.uniform(2, 13)
            bus675_b_kw = -100*np.random.uniform(3, 20)
            bus675_c_kw = -100*np.random.uniform(2, 13)
            self.network.run_command(f"Generator.bus675_1.kw={bus675_a_kw}") 
            self.network.run_command(f"Generator.bus675_2.kw={bus675_b_kw}") 
            self.network.run_command(f"Generator.bus675_3.kw={bus675_c_kw}") 

            bus633_a_kw = -100*np.random.uniform(6, 28)
            bus633_b_kw = -100*np.random.uniform(5, 25)
            bus633_c_kw = -100*np.random.uniform(3, 25)
            self.network.run_command(f"Generator.bus633_1.kw={bus633_a_kw}") 
            self.network.run_command(f"Generator.bus633_2.kw={bus633_b_kw}") 
            self.network.run_command(f"Generator.bus633_3.kw={bus633_c_kw}") 

            bus680_a_kw = -100*np.random.uniform(1.5, 5)
            bus680_b_kw = -100*np.random.uniform(1.5, 8)
            bus680_c_kw = -100*np.random.uniform(1.5, 8)
            self.network.run_command(f"Generator.bus680_1.kw={bus680_a_kw}") 
            self.network.run_command(f"Generator.bus680_2.kw={bus680_b_kw}") 
            self.network.run_command(f"Generator.bus680_3.kw={bus680_c_kw}") 
        if(senario == 1):
            # High voltage
            bus675_a_kw = 100*np.random.uniform(4, 20)
            bus675_b_kw = 100*np.random.uniform(3, 20)
            bus675_c_kw = 100*np.random.uniform(2, 20)
            self.network.run_command(f"Generator.bus675_1.kw={bus675_a_kw}") 
            self.network.run_command(f"Generator.bus675_2.kw={bus675_b_kw}") 
            self.network.run_command(f"Generator.bus675_3.kw={bus675_c_kw}") 

            bus633_a_kw = 100*np.random.uniform(5, 20)
            bus633_b_kw = 100*np.random.uniform(8, 18)
            bus633_c_kw = 100*np.random.uniform(8, 15)
            self.network.run_command(f"Generator.bus633_1.kw={bus633_a_kw}") 
            self.network.run_command(f"Generator.bus633_2.kw={bus633_b_kw}") 
            self.network.run_command(f"Generator.bus633_3.kw={bus633_c_kw}") 

            bus680_a_kw = 100*np.random.uniform(1.5, 5)
            bus680_b_kw = 100*np.random.uniform(4, 7)
            bus680_c_kw = 100*np.random.uniform(3, 10)
            self.network.run_command(f"Generator.bus680_1.kw={bus680_a_kw}") 
            self.network.run_command(f"Generator.bus680_2.kw={bus680_b_kw}") 
            self.network.run_command(f"Generator.bus680_3.kw={bus680_c_kw}") 
        self.network.dss.Solution.Number(1)
        self.network.dss.Solution.Solve()      

        self.state=self.get_state()
        return self.state
    
    def reset(self, seed=1): #sample different initial volateg conditions during training
        np.random.seed(seed)
        senario = np.random.choice([0,1])
        # senario = 1
        self.network.init_sys()
        if(senario == 0):
            # Low voltage
            bus_a_kw = -100*np.random.uniform(2, 4.5)
            bus_b_kw = -100*np.random.uniform(3, 5)
            bus_c_kw = -100*np.random.uniform(2, 4)
            for idx in self.injection_bus_str:
                for phase in self.injection_bus[idx]:
                    if phase == 'a':
                        self.network.run_command(f"Generator.bus{idx}_1.kw={bus_a_kw}") 
                    elif phase == 'b':
                        self.network.run_command(f"Generator.bus{idx}_2.kw={bus_b_kw}") 
                    elif phase == 'c':
                        self.network.run_command(f"Generator.bus{idx}_3.kw={bus_c_kw}") 
        if(senario == 1):
            # High voltage
            bus_a_kw = 100*np.random.uniform(3, 5.5)
            bus_b_kw = 100*np.random.uniform(4.5, 5)
            bus_c_kw = 100*np.random.uniform(4, 5)
            for idx in self.injection_bus_str:
                for phase in self.injection_bus[idx]:
                    if phase == 'a':
                        self.network.run_command(f"Generator.bus{idx}_1.kw={bus_a_kw}") 
                    elif phase == 'b':
                        self.network.run_command(f"Generator.bus{idx}_2.kw={bus_b_kw}") 
                    elif phase == 'c':
                        self.network.run_command(f"Generator.bus{idx}_3.kw={bus_c_kw}") 
        self.network.dss.Solution.Number(1)
        self.network.dss.Solution.Solve()      

        self.state=self.get_state()
        return self.state
    
if __name__ == "__main__":
    # injection_bus = np.array([675,633,680])
    # bus 670 is actually a concentrated point load of the distributed load on line 632 to 671 located at 1/3 the distance from node 632
    injection_bus = np.array([633,634,671,645,646,692,675,611,652,632,680,684])
    net, injection_bus_dict = create_13bus3p(injection_bus)    
    env = IEEE13bus3p(net, injection_bus_dict)
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
    # state = env.reset(0)
    # for i in range(40):
    #     action = np.zeros((len(injection_bus),3))
    #     action += 0.01*i
    #     state, _,_,_ = env.step_Preward(action,action)
    #     state_list.append(state)
    #     # print(env.network.run_command('? Load.675a.kw'))
    # state_list = np.array(state_list)
    # fig, axs = plt.subplots(len(injection_bus), 3, figsize=(9,9))
    # for i in range(3):
    #     for j in range(len(injection_bus)):
    #         axs[j,i].plot(range(40),state_list[:,j,i])
    # plt.show()
