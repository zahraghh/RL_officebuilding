import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import gym
from gym import spaces
from pathlib import Path
import random
# Edit the gaps in LEEM EF
# are we focus on MI? change the E+ results
# Do we have acccess to mroe recent LEEM EFs?
data_path = Path("C:/Users/Zahra/Desktop/RL/Zahra/data")
date_LEEM = '5/23/2015'
# Initialize parameters
gamma = 0.75 # Discount factor
alpha = 0.9 # Learning rate
epsilon = 0.0 # Epsilon-gready
E_WF = 0.00 # Energy Weight factor
P_WF = 0.00 # PPD Weight factor
C_WF = 0.00 # Cost  Weight factor
EF_WF = 1.00 # Emission factor Weight factor
params = [2.073929810495441, 1164.149277247046, 143.33046792990825] # For a gennorm distribution shows: Beta, mean, STD

class Building(gym.Env):
  """Custom Environment that follows gym interface"""
  def __init__(self,data_path,date,_date_LEEM_,_percentag_EF):
      self.E_results = {}
      self._date_LEEM = _date_LEEM_
      self.percentag_EF = _percentag_EF
      energy_TbadLbad_path = data_path / 'OfficeSmall_Meter_TbadLbad.csv'
      energy_TzeroLzero_path = data_path / 'OfficeSmall_Meter_TzeroLzero.csv'
      energy_TzeroLbad_path = data_path / 'OfficeSmall_Meter_TzeroLbad.csv'
      energy_TbadLzero_path = data_path / 'OfficeSmall_Meter_TbadLzero.csv'
      TC_TbadLbad_path = data_path / 'OfficeSmall_TbadLbad.csv'
      TC_TzeroLzero_path = data_path / 'OfficeSmall_TzeroLzero.csv'
      TC_TzeroLbad_path = data_path / 'OfficeSmall_TzeroLbad.csv'
      TC_TbadLzero_path = data_path / 'OfficeSmall_TbadLzero.csv'
      energy_TbadLbad = pd.read_csv(energy_TbadLbad_path)
      energy_TzeroLzero = pd.read_csv(energy_TzeroLzero_path)
      energy_TzeroLbad = pd.read_csv(energy_TzeroLbad_path)
      energy_TbadLzero = pd.read_csv(energy_TbadLzero_path)
      TC_TbadLbad = pd.read_csv(TC_TbadLbad_path)
      TC_TzeroLzero = pd.read_csv(TC_TzeroLzero_path)
      TC_TzeroLbad = pd.read_csv(TC_TzeroLbad_path)
      TC_TbadLzero = pd.read_csv(TC_TbadLzero_path)
        # new data frame with split value columns
      new = energy_TbadLbad["Date/Time"].str.split(" ", n = 0, expand = True)
      energy_TbadLbad["Date"],energy_TbadLbad["Time"]= new[1],new[3]
      hour_min = energy_TbadLbad["Time"].str.split(":", n = 0, expand = True)
      self.hour, self.min = hour_min[0].astype(int), hour_min[1].astype(int)
      self.E_results['Date'] =  energy_TbadLbad["Date"]
      self.E_results['Time'] =  energy_TbadLbad["Time"]
      starting_date = [i for i in range(len( self.E_results['Date'])) if  self.E_results['Date'][i] == date]
      self.starting_date = starting_date[0]
      self.starting_time = self.E_results['Time'][starting_date[0]]
      self.E_results['Time_step'] = energy_TbadLbad["Date"].index.values
      self.E_results['ElectricityNet_TzeroLzero'] = list(energy_TzeroLzero['ElectricityNet:Facility [J](TimeStep)'])
      self.E_results['ElectricityNet_TzeroLbad'] = list(energy_TzeroLbad['ElectricityNet:Facility [J](TimeStep)'])
      self.E_results['ElectricityNet_TbadLzero'] = list(energy_TbadLzero['ElectricityNet:Facility [J](TimeStep)'])
      self.E_results['ElectricityNet_TbadLbad'] = list(energy_TbadLbad['ElectricityNet:Facility [J](TimeStep)'])
      self.E_results['PPD_CORE_ZN_TzeroLzero'] = list(TC_TzeroLzero['CORE_ZN:Zone Thermal Comfort Fanger Model PPD [%](TimeStep)'])
      self.E_results['PPD_CORE_ZN_TzeroLbad'] = list(TC_TzeroLbad['CORE_ZN:Zone Thermal Comfort Fanger Model PPD [%](TimeStep)'])
      self.E_results['PPD_CORE_ZN_TbadLzero'] = list(TC_TbadLzero['CORE_ZN:Zone Thermal Comfort Fanger Model PPD [%](TimeStep)'])
      self.E_results['PPD_CORE_ZN_TbadLbad'] = list(TC_TbadLbad['CORE_ZN:Zone Thermal Comfort Fanger Model PPD [%](TimeStep)'])
      min_action, max_action = 0, 4 # 9 action sets, 96 time a day (evey 15 min) --> 4 for the first try
      low_state, high_state = 0, 144  # time of the day
      self.action_space = spaces.Box(low=np.array(min_action), high=np.array(max_action), dtype=np.float32)
      self.action_list = ['a1','a2','a3','a4']
      self.observation_space = spaces.Box(low=np.array(low_state), high=np.array(high_state), dtype=np.float32)
      self.Q_value = np.full([high_state,max_action], 5.0000)
      self.optimal_policy = np.zeros(high_state)
  def reset(self):
      # Reset the state of the environment to an initial state
      self.observation = 0
      self.date_index = self.starting_date
      self.action = 'a1'
      self.new_action_arg = 0
  def step(self, action,observation,date_index,_LEEM_EF):
      # Execute one time step within the environment
      self.action = self.take_action(action,observation,date_index,_LEEM_EF)
      self.observation += 1
      self.date_index += 1
      return self.action
  def set_action(self, new_action):
      self.action = new_action
  def reward(self, E1, E2, PPD1, PPD2,_LEEM_EF):
      _LEEM_EF_rd = _LEEM_EF*self.percentag_EF + _LEEM_EF
      return (E1-E2)*E_WF/E1 + (PPD1-PPD2)*P_WF/PPD1 + (E1-E2)*C_WF/E1 + (E1-E2)*EF_WF*_LEEM_EF_rd/E1
  def argmax_rd(self,q_values):
      top_value = float("-inf")
      ties = []
      for i in range(len(q_values)):
       # if a value in q_values is greater than the highest value update top and reset ties to zero
       # if a value is equal to top value add the index to ties
          if q_values[i]>top_value:
             top_value = q_values[i]
             ties = []
          if q_values[i]==top_value:
             ties.append(i)
      return np.random.choice(ties)
  def take_action(self, action, observation, date_index,_LEEM_EF):
      # Choose a from s using action selection (e.g. epsilon-greedy)
      rewards_copy = self.max_reward(date_index,_LEEM_EF)
      random_number = np.random.random_sample()
      if random_number >epsilon:
          self.new_action_arg = self.argmax_rd(self.Q_value[observation])
      if random_number <epsilon:
          self.new_action_arg = self.action_list.index(np.random.choice(self.action_list))
      # Compute the temporal difference
      self.reward_action = rewards_copy[self.new_action_arg]
      TD = self.reward_action +  gamma * max(self.Q_value[observation+1,]) - self.Q_value[observation, self.new_action_arg]
      # Update the Q-Value using the Bellman equation
      self.Q_value[observation,self.new_action_arg] += alpha * TD
      new_action = self.action_list[ self.new_action_arg]
      self.optimal_policy[observation] = self.new_action_arg
      return new_action ,self.reward_action
  def max_reward(self, date_index,_LEEM_EF):
      ElectricityNeta1 = self.E_results['ElectricityNet_TzeroLzero'][date_index]
      PPDa1 = self.E_results['PPD_CORE_ZN_TzeroLzero'][date_index]
      ElectricityNeta2 = self.E_results['ElectricityNet_TzeroLbad'][date_index]
      PPDa2 = self.E_results['PPD_CORE_ZN_TzeroLbad'][date_index]
      ElectricityNeta3 = self.E_results['ElectricityNet_TbadLzero'][date_index]
      PPDa3 = self.E_results['PPD_CORE_ZN_TbadLzero'][date_index]
      ElectricityNeta4 = self.E_results['ElectricityNet_TbadLbad'][date_index]
      PPDa4 = self.E_results['PPD_CORE_ZN_TbadLbad'][date_index]
      rewards_copy = [0, self.reward(ElectricityNeta1,ElectricityNeta2,PPDa1,PPDa2,_LEEM_EF), self.reward(ElectricityNeta1,ElectricityNeta3,PPDa1,PPDa3,_LEEM_EF),self.reward(ElectricityNeta1,ElectricityNeta4,PPDa1,PPDa4,_LEEM_EF) ]
      return rewards_copy
  def LEEM_EF_match(self):
      starting_date_Ep = self.starting_date
      hour_Ep, min_Ep = self.hour, self.min
      Time_span = len(self.Q_value)
      LEEM_EF_path = data_path /'EFs'/ '5A_RFCM_DI_MI.csv'
      LEEM_EF = pd.read_csv(LEEM_EF_path)
      LEEM_date_time = LEEM_EF['Time'].str.split(" ", n = 0, expand = True)
      LEEM_EF['Date'], LEEM_EF['Time'] = LEEM_date_time[0],LEEM_date_time[1]
      hour_min_LEEM = LEEM_EF["Time"].str.split(":", n = 0, expand = True)
      hour_LEEM, min_LEEM = hour_min_LEEM[0].astype(int), hour_min_LEEM[1].astype(int)
      LEEM_EF_CO2= LEEM_EF['RL CO2 Emission Rate(lbs/MWh)']
      date_LEEM = [i for i in range(len( LEEM_EF['Date'])) if LEEM_EF['Date'][i] == self._date_LEEM]
      starting_date_LEEM = date_LEEM[0]
      def is_number(s):
          try:
              float(s)
              return True
          except ValueError:
              return False
    #def time_matching():
      starting_time_LEEM = LEEM_EF['Time'][starting_date_LEEM]
      LEEM_EF_match = np.zeros(Time_span)
      for j in range(Time_span):
          for i in range(len(date_LEEM)):
              if is_number(LEEM_EF_CO2[starting_date_LEEM+i]):
                  if hour_LEEM[starting_date_LEEM+i]== hour_Ep[starting_date_Ep+j]:
                      if min_LEEM[starting_date_LEEM+i]== min_Ep[starting_date_Ep+j]:
                          LEEM_EF_match[j] = float(LEEM_EF_CO2[starting_date_LEEM+i])

      missed_indices = np.where(LEEM_EF_match == 0)[0]
      for j in missed_indices:
          if j != Time_span-1:
              if LEEM_EF_match[j+1]!=0 and LEEM_EF_match[j-1]!=0:
                  LEEM_EF_match[j]= (LEEM_EF_match[j+1]+LEEM_EF_match[j-1])/2
              else:
                  LEEM_EF_match[j]=LEEM_EF_match[j-1]
          else:
              LEEM_EF_match[j]=LEEM_EF_match[j-1]
      return LEEM_EF_match.round(3)
  def  cum_reward(self, q_values,date_index,LEEM_EF):
      cum_reward=np.zeros(len(q_values))
      for i in range(len(q_values)):
          cum_reward[i] = max(self.max_reward(i+date_index,LEEM_EF[i]))
      return cum_reward
  def render(self):
      print(self.Q_value)



rd_EF = []
samples =2
for s in range(samples):
    rvs_EF = st.gennorm.rvs(beta=params[0], loc=params[1], scale=params[2])
    rd_EF.append(rvs_EF)
rd_EF = sorted(rd_EF)
rewards_sample = np.zeros(samples)
dict_policy = {}
for s in range(samples):
    percentag_EF  = (rd_EF[s] - params[1])/params[1]
    print("sample: ", s, percentag_EF)
    office_building = Building(data_path, '05/23','5/23/2015', percentag_EF)
    Time_span = len(office_building.Q_value)
    office_building.reset()
    LEEM_EF_rd = office_building.LEEM_EF_match()
    episodes=100   # Update the Q-Value using the Bellman equation
    cum_reward=np.zeros(episodes)
    cum_newaction=np.zeros(episodes)

    for e in range(episodes):
        i=0
        j=0
        office_building.reset()
        while office_building.observation < Time_span-1:
            new_action,rewards_copy = office_building.step(office_building.action,office_building.observation,office_building.date_index,LEEM_EF_rd[office_building.observation])
            office_building.set_action(new_action)
            if new_action=='a4':
                i += 1
            if new_action=='a2':
                j += 1
    #print( office_building.optimal_policy)
    cum_reward = office_building.cum_reward(office_building.Q_value,office_building.date_index,LEEM_EF_rd)
    if e == episodes-1:
        dict_policy[str(rd_EF[s])] = office_building.optimal_policy
        rewards_sample[s] = sum(cum_reward)

print(dict_policy)
for s in range(samples):
    plt.plot(dict_policy.get(str(rd_EF[s])))
    plt.show()
#plt.plot(rewards_sample)
