import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector

class PileEnv():
    def __init__(self, pile_id = 'no id', car_type = 'None', Ta = 0, Td = 3, BC = 24, SOC = 0.2, SOC_T = 0.95, P_max_charge = 6, CE = 0.9, delta_t = 1, price_data = np.zeros(96)):
        self.pile_id  = pile_id
        self.car_type = car_type
        self.Ta    = int(Ta) # T_arrive
        self.Td    = int(Td) # T_depart
        self.Ts    = self.Td  - self.Ta - 1
        self.BC    = BC    # Battery capacity
        self.SOC   = SOC   # Current SOC (%)
        self.SOC_T = SOC_T # Target SOC (%)

        self.SOC_upper = 0.95
        self.SOC_lower = 0.2

        # max charged/discharge power
        self.P_max_charge = P_max_charge

        self.P_distribute = P_max_charge

        self.CE = CE           # Charge efficiency
        self.delta_t = delta_t # charge time step long ( t mins/60 mins )

        self.price_data = np.array(price_data[ self.Ta+1 : self.Td ])

        # === For calculate reward === #
        self.mean_price = 0
        self.mean_price_after_charge = 0
        # ============================ #

    def get_price_data(self):
        if len(self.price_data) <= 0:
            return np.zeros(27)#4+23
        
        elif len(self.price_data) <= 4:
            nor_price = self.price_data/np.max(self.price_data)
            return np.concatenate( ( nor_price, np.zeros( 27 - len(nor_price) ) ) )#.astype(np.float32)
        
        else:
            nor_price = self.price_data/np.max(self.price_data)

            now_15_data = nor_price[:4]

            day_data = np.concatenate( ( nor_price, np.zeros( 96 - len(nor_price) ) ) )#.astype(np.float32)

            future_hour_data = np.mean(day_data[4:].reshape(-1,4),axis=1)

            if len(nor_price)%4 > 0:

                change_index = len(nor_price)//4-1

                change_index_data = np.mean(nor_price[-(len(nor_price)%4):])

                future_hour_data[int(change_index)] = change_index_data
                
            return np.concatenate( ( now_15_data, future_hour_data ) )#.astype(np.float32)  
            
    def show_car_info(self):
        print(f"-- {self.pile_id:8} -- car[ SOC: {self.SOC:.3f} | Target: {self.SOC_T:.3f} | T_stay: {self.Ts:2d} | T_arrive: {self.Ta} | T_depart: {self.Td} | BC: {self.BC}| CE: {self.CE}|P_max_charge: {self.P_max_charge}|Flex:{self.get_flexibility()}]")

    def get_flexibility(self):
        return self.Ts - ( self.SOC_T - self.SOC )*self.BC/( self.P_distribute*self.delta_t )
    
    def get_flexibility_P_d(self,P_d):
        return self.Ts - ( self.SOC_T - self.SOC )*self.BC/( P_d*self.delta_t )

    def get_P_min(self):
        if self.Ts <= 0:
            return 0
        
        flexibility = self.get_flexibility()

        if flexibility >= 1: 
            # able to discharge / supply power to grid
            return max( -1*self.P_max_charge, ( 1 - flexibility )*self.P_distribute, ( self.SOC_lower - self.SOC )*self.BC/self.delta_t )
        else: 
            # need to charge
            # need how much power from grid
            return min( self.P_distribute, ( 1 - flexibility )*self.P_distribute )

    def get_P_max(self): 
        if self.Ts <= 0:
            return 0
        else:
            return min( ( self.SOC_upper - self.SOC )*self.BC/self.delta_t, self.P_max_charge)

    def get_to_request_time(self):
        return ( self.SOC_T - self.SOC )*self.BC/( self.P_distribute*self.delta_t )
        
    def get_to_lower_time(self):
        if self.Ts <= 0:
            return 0
        else:
            return min( ( self.SOC_lower - self.SOC )*self.BC/( self.P_distribute*self.delta_t ), 0)
        
    def get_to_upper_time(self): 
        if self.Ts <= 0:
            return 0
        else:
            return max( ( self.SOC_upper - self.SOC )*self.BC/( self.P_distribute*self.delta_t), 0)
    
    def charge(self, charge_power):
        # charge_power > 0  charge | charge_power < 0 discharge
        self.SOC = self.SOC + charge_power*self.delta_t/self.BC
        self.Ts  = int(self.Ts) - 1

        price_max = np.max(self.price_data)
        self.mean_price = np.mean(self.price_data/price_max)

        # update price data
        self.price_data = np.delete(self.price_data, 0).astype(np.float32)

        if len(self.price_data) <= 1:
            self.mean_price_after_charge = 1
        else:
            self.mean_price_after_charge = np.mean(self.price_data/price_max)

class StationEnv(AECEnv):
    metadata = {
        "render_modes": ["human"],
        "name": "station_v1",
        "is_parallelizable": False,
    }

    def __init__(self, mode = "train", N_piles = 10):
        # self._seed()
        self.mode    = mode

        self.N_piles = N_piles

        self.possible_agents = [f"pile_{n}" for n in range( 1, self.N_piles + 1)]

        self.agents = self.possible_agents[:]

        # For AECEnv to select next agent
        self._agent_selector = agent_selector(self.agents) 

        # -------- station setting ------------ #
        self.pile = { agent : None for agent in self.possible_agents }
        
        self.P_pile = (5+np.random.rand()*5) #(3.5+np.random.rand()*1.5) #(kW)

        self.P_max_station = self.N_piles*self.P_pile
        if self.mode =='test':
            self.P_max_station = 80

        self.car_type_list = [ "Chevy Volt", "Volkswagen E-Golf", "BMW i3", "Tesla Model S"]

        self.car_type_data = {
            "Chevy Volt"        : { 'BC': 16, 'P_rated':  3.3},#(kW)
            "Volkswagen E-Golf" : { 'BC': 24, 'P_rated':  7.2},#(kW)
            "BMW i3"            : { 'BC': 22, 'P_rated':  6.6},#(kW)
            "Tesla Model S"     : { 'BC': 75, 'P_rated': 11.5} #(kW)
        }

        # Price data - 15 mins price ------------
        self.Pgrid_price = np.array([1.9]*9*4+[4.26]*7*4+[6.89]*6*4+[4.26]*2*4)
        
        # stop time step
        self.max_time_step = len(self.Pgrid_price)

        # charge time 
        self.delat_t = 24/self.max_time_step
        
        # calculate cost
        self.total_cost = 0

        # For generating car schedule 
        self.time_probability = {
             0: 0.0,  1: 0.0,  2: 0.0,  3: 0.0,  4: 0.0,  5: 0.0,  6: 0.03,  7: 0.2,  8: 0.4,  9: 0.3, 10: 0.2, 11: 0.2,
            12: 0.3, 13: 0.3, 14: 0.2, 15: 0.2, 16: 0.1, 17: 0.1, 18: 0.08, 19: 0.05, 20: 0.03, 21: 0.0, 22: 0.0, 23: 0.0
        }

        self._for_train_time_have_car = {
             0: 0.0,  1: 0.0,   2: 0.0,  3: 0.0,  4: 0.0,  5: 0.0,  6: 0.03,  7: 0.23,  8: 0.4,  9: 0.7, 10: 0.7, 11: 0.7,
            12: 0.8, 13: 0.85, 14: 0.85, 15: 0.85, 16: 0.8, 17: 0.7, 18: 0.5, 19: 0.4, 20: 0.3, 21: 0.3, 22: 0.0, 23: 0.0
        }

        # ============ Define space ============ #
        self.action_spaces = { agent : spaces.Box( -1, 1, dtype=np.float32) for agent in self.agents}

        # [ T_to_T| T_stay| T_to_L|  T_to_U|   P_t_L|  P_t_U] + price
        self.low  = np.array( [ -20,  0, -20,  0, -3, -3] + [0]*27, dtype=np.float32)
        self.high = np.array( [  20, 96,   0, 20,  3,  3] + [1]*27, dtype=np.float32)

        self.observation_spaces = { agent : spaces.Box(low=self.low, high=self.high, shape=(len(self.low),), dtype=np.float32) for agent in self.agents }

        self.obs_len = len(self.low)
        
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def render(self):
        print("\n==============")
        for agent in self.possible_agents:
            if self.pile[agent] != None:
                self.pile[agent].show_car_info()
        print("==============\n")
    
    def observe(self, agent):
        return self.observations[agent]
    
    def close(self):
        pass

    def save_plot_data(self):
        if self.mode != "train":
            self.plot_data["P_total"].append(sum(self.P_already.values()))
            for agent in self.possible_agents:
                if self.pile[agent] != None:
                    self.plot_data["SOC"][agent].append(self.pile[agent].SOC)
                    self.plot_data["SOC_T"][agent].append(self.pile[agent].SOC_T)
                    self.plot_data["P_action"][agent].append(self.P_plot_action[agent])
                    self.plot_data["P_upper"][agent].append(self.observations[agent][4]) # state : [ T_to_T | T_stay | T_to_L | T_to_U | P_t_L | P_t_U ] + price
                    self.plot_data["P_lower"][agent].append(self.observations[agent][5])
                    self.plot_data["P_use"][agent].append(self.P_already[agent])
                else:
                    self.plot_data["SOC"][agent].append(0)
                    self.plot_data["SOC_T"][agent].append(0)
                    self.plot_data["P_action"][agent].append(0)
                    self.plot_data["P_upper"][agent].append(0)
                    self.plot_data["P_lower"][agent].append(0)
                    self.plot_data["P_use"][agent].append(0)

    def _car_arrive(self, agent):
        if self.T_cur <= 42:
            probability = [0.4,0.3,0.3]
        elif 42 < self.T_cur <= 60:
            probability = [0.0,0.4,0.6]
        else:
            probability = [0.0,0.0,1.0]
        
        if self.mode == "train" and self.select_pile ==  agent:
            probability = [0.0,0.0,1.0]

        depart_time = np.random.choice(['noon','afternoon','night'], 1, p = probability )

        if depart_time == 'noon':
            Td = int(np.clip( np.array( np.random.normal( 44, 5), dtype=np.int8), self.T_cur+5, 64))
        elif depart_time == 'afternoon':
            Td = int(np.clip( np.array( np.random.normal( 72, 5), dtype=np.int8), self.T_cur+5, 80))
        else:
            Td = int(np.clip( np.array( np.random.normal( 90, 3), dtype=np.int8), self.T_cur+5, 95))

        if self.mode == "train":
            car_type    = "random"
            car_BC      = np.random.randint(15,31)
            car_P_rated = car_BC/(3+np.random.rand()*3)
        else:
            car_type    = np.random.choice( self.car_type_list, 1, p = np.ones(len(self.car_type_list))/len(self.car_type_list))[0]
            car_BC      = self.car_type_data[car_type]["BC"]
            car_P_rated = self.car_type_data[car_type]["P_rated"]

        car_CE    = 0.9
        car_SOC_T = np.random.randint(80, 91)/100
        car_SOC   = np.clip( (car_SOC_T - np.random.randint( 3, (Td - self.T_cur -1))*car_P_rated*self.delat_t/car_BC), 0.2, car_SOC_T)

        self.pile[agent] = PileEnv( pile_id = agent, car_type = car_type, Ta = self.T_cur, Td = Td, BC = car_BC, SOC = car_SOC, SOC_T = car_SOC_T,
                                    P_max_charge = car_P_rated, CE = car_CE, delta_t = self.delat_t, price_data = self.execute_Pgrid_price)

    def _for_train_car_already_on_pile(self, agent):
        if self.T_cur <= 42: 
            probability = [0.4,0.3,0.3]
        elif 42 < self.T_cur <= 60:
            probability = [0.0,0.4,0.6]
        else:
            probability = [0.0,0.0,1.0]

        depart_time = np.random.choice(['noon','afternoon','night'], 1, p = probability )

        if depart_time == 'noon':
            Td = int(np.clip( np.array( np.random.normal( 44, 5), dtype=np.int8), self.T_cur+1, 64))
        elif depart_time == 'afternoon':
            Td = int(np.clip( np.array( np.random.normal( 72, 5), dtype=np.int8), self.T_cur+1, 80))
        else:
            Td = int(np.clip( np.array( np.random.normal( 90, 3), dtype=np.int8), self.T_cur+1, 95))

        car_type    = "random"
        car_BC      = np.random.randint(15,31)
        car_P_rated = car_BC/(3+np.random.rand()*3)

        car_CE    = 0.9
        car_SOC_T = np.random.randint(80,95)/100
        if Td - self.T_cur - 1 <= 0:
            car_SOC = car_SOC_T
        else:
            car_SOC = np.clip( (car_SOC_T - (Td - self.T_cur -1)*np.random.rand()*car_P_rated*self.delat_t/car_BC), 0.2, car_SOC_T)

        self.pile[agent] = PileEnv( pile_id = agent, car_type = car_type, Ta = self.T_cur, Td = Td, BC = car_BC, SOC = car_SOC, SOC_T = car_SOC_T,
                                    P_max_charge = car_P_rated, CE = car_CE, delta_t = self.delat_t, price_data = self.execute_Pgrid_price)

    def _reset_time_step_saver(self):
        # when go into a new timestep, reseting the variable
        self.P_request     = { agent : 0 for agent in self.possible_agents }
        self.P_already     = { agent : 0 for agent in self.possible_agents }
        self.P_plot_action = { agent : 0 for agent in self.possible_agents }

    def _train_pre_set(self, select_pile):
        self._car_arrive(select_pile)
        for agent in self.possible_agents:
            if (self.pile[agent] == None) and (np.random.rand() > ( 1 - self.time_probability[self.T_cur//4])):
                self._for_train_car_already_on_pile(agent = agent)

    def _check_env_done(self):
        if self.T_cur >= (self.max_time_step - 1):
            self.env_done = True

    def _get_charge_sequence(self):
        execute_pile_list = []
        self.charge_power_max = {}
        
        for agent in self.possible_agents:
            if (self.pile[agent] != None):
                execute_pile_list.append(agent)

                if self.pile[agent].Ts >= 1:
                    self.charge_power_max[agent] = self.pile[agent].P_max_charge/self.pile[agent].CE
            
        if sum(self.charge_power_max.values()) <= self.P_max_station:
            for agent in self.charge_power_max.keys():
                self.pile[agent].P_distribute = self.pile[agent].P_max_charge
        else:
            self._update_distribute_power( self.charge_power_max.keys())
        
        able_charge_agent_flexibility = {}
        for agent in execute_pile_list:
            if self.pile[agent].Ts < 1:
                able_charge_agent_flexibility[agent] = 100
            else:
                able_charge_agent_flexibility[agent] = self.pile[agent].get_flexibility()

        self.charge_sequence = list( dict( sorted(able_charge_agent_flexibility.items(), key=lambda item: item[1], reverse = True) ).keys())
    
    def _update_distribute_power(self, update_list):
        total_charge_power = 0
        for agent in update_list:
            total_charge_power += self.charge_power_max[agent]
            
        set_P_need_pile = []
        flexibility_enough_pile = []
        remain_power = self.P_max_station - sum(self.P_request.values())
        for agent in update_list:
            pile_distribute_power = max( self.pile[agent].P_max_charge, remain_power*self.pile[agent].CE*(self.charge_power_max[agent]/total_charge_power) )
            
            if self.pile[agent].get_flexibility_P_d(pile_distribute_power) > -0.00001 : # > 0 但因為有浮點誤差
                self.pile[agent].P_distribute = pile_distribute_power
                flexibility_enough_pile.append(agent)
            else:
                P_distribute_need = ( (self.pile[agent].SOC_T - self.pile[agent].SOC)*self.pile[agent].BC )/(self.pile[agent].delta_t*self.pile[agent].Ts)
                self.pile[agent].P_distribute = P_distribute_need
                self.P_request[agent] = P_distribute_need
                set_P_need_pile.append(agent)

        if len(set_P_need_pile) > 1:
            self._update_distribute_power(flexibility_enough_pile)
    
    def _get_charge_request(self):
        self.distribute_power_list = {}
        for agent in self.charge_power_max.keys():
            self.P_request[agent] = max(0,self.pile[agent].get_P_min()/self.pile[agent].CE)
            if self.P_request[agent] == 0:
                self.distribute_power_list[agent] = self.pile[agent].P_distribute/self.pile[agent].CE
    
    def _move_to_next_time_step(self):
        self.save_plot_data()
        self._reset_time_step_saver()
        self._get_charge_sequence()
        self.T_cur = int(self.T_cur) + 1
        
        for agent in self.possible_agents:
            if (self.pile[agent] == None) and (np.random.rand() > ( 1 - self.time_probability[self.T_cur//4])):
                self._car_arrive(agent)
        
    def _move_to_next_iter(self):
        self._move_to_next_time_step()
        self._check_env_done()

        while len(self.charge_sequence)<1:
            if not self.env_done:
                self._move_to_next_time_step()
            else:
                break

        for agent in self.charge_sequence:
            if agent not in self.agents:
                self.rewards[agent] = 0
                self.terminations[agent] = False
                self.truncations[agent]  = False
                self._cumulative_rewards[agent] = 0
                self.infos[agent] = {}
            
        self._get_charge_request()
        self.agents = self.charge_sequence
        self._agent_selector.reinit(self.charge_sequence)
        
    def _get_agent_obs(self, now_agent):
        if self.pile[now_agent] == None:
            return np.zeros(self.obs_len)

        T_stay = self.pile[now_agent].Ts

        # P_already use power / 1 to K-1
        P_already = sum(self.P_already.values())
        # P_request power / K+1 to N
        self.P_request[now_agent] = 0 
        P_request = sum(self.P_request.values())
        
        if now_agent in self.distribute_power_list.keys():
            power_distribute_rate = self.distribute_power_list[now_agent]/sum(self.distribute_power_list.values())
            self.distribute_power_list[now_agent] = 0
        else:
            power_distribute_rate = 1

        if T_stay >= 1 :
            P_min_lower = self.pile[now_agent].get_P_min()
            P_max_upper = self.pile[now_agent].get_P_max()

            P_t_upper = min( P_max_upper, max( 0, ( self.P_max_station - P_already - P_request )* power_distribute_rate)*self.pile[now_agent].CE )

            if P_min_lower >= 0: # need to charge
                P_t_lower = min( P_min_lower, P_t_upper)
            else: # able to discharge
                P_t_lower = max( P_min_lower, (-1*self.P_max_station - P_already - P_request )/self.pile[now_agent].CE )
        else:
            P_t_upper = 0
            P_t_lower = 0

        state_P_t_upper = P_t_upper/self.pile[now_agent].P_distribute
        state_P_t_lower = P_t_lower/self.pile[now_agent].P_distribute

        T_to_request = self.pile[now_agent].get_to_request_time()
        T_to_lower   = self.pile[now_agent].get_to_lower_time()
        T_to_upper   = self.pile[now_agent].get_to_upper_time()
        price_data   = self.pile[now_agent].get_price_data()
        # [ T_to_T| T_stay| T_to_L|  T_to_U|   P_t_L|  P_t_U]
        return np.concatenate( ( [ T_to_request, T_stay, T_to_lower, T_to_upper, state_P_t_lower, state_P_t_upper], price_data ) ).astype(np.float32)
    
    def _seed(self, seed=None):
        if self.mode != 'train':
            np.random.seed(seed)
        self.np_random, seed = seeding.np_random(seed)

    def seed(self,seed):
        np.random.seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        print(f"set multi_env seed:{seed}")

    def reset(self, seed=None, options=None, select_pile = "pile_0", price_data = []):
        ###
        if self.mode == 'train':
            self.P_pile = (5+np.random.rand()*5)
            self.P_max_station = self.N_piles*self.P_pile
        ###
        
        # if seed is not None:
        #     self._seed(seed=seed)
        # ---------------- pettingzoo env need --------------- #
        self.agents              = self.possible_agents[:]
        self.observations        = { agent: np.zeros(self.obs_len) for agent in self.agents }
        self.rewards             = { agent: 0     for agent in self.agents }
        self._cumulative_rewards = { agent: 0     for agent in self.agents }
        self.terminations        = { agent: False for agent in self.agents }
        self.truncations         = { agent: False for agent in self.agents }
        self.infos               = { agent: {}    for agent in self.agents }
        # ----------------------------------------------------- #
        self.step_reward         = { agent : 0    for agent in self.agents }
        self.total_cost = 0
        # ----------------------------------------------------- #
        self.env_done = False
        self.pile = { agent : None for agent in self.agents }
        self._reset_time_step_saver()
        
        # ------------------- setting price ------------------- #
        if self.mode == "train":
            # self.execute_Pgrid_price = []
            # while len(self.execute_Pgrid_price) < 96:
            #     self.execute_Pgrid_price = np.append(self.execute_Pgrid_price, [1+np.random.rand()*6]*np.random.randint(1,26) ).astype(np.float32) #36
            #     # self.execute_Pgrid_price = np.append(self.execute_Pgrid_price, [1+np.random.rand()*6]*np.random.randint(1,41) ).astype(np.float32)
                
            # self.execute_Pgrid_price = np.clip(self.execute_Pgrid_price[0:96] + np.random.normal(0,0.1,96), 1, 7)
            
            self.execute_Pgrid_price = []
            while len(self.execute_Pgrid_price) < 96:
                self.execute_Pgrid_price = np.append(self.execute_Pgrid_price, [1+np.random.rand()*6]*np.random.randint(1,26) ).astype(np.float32)

            self.execute_Pgrid_price = np.clip(self.execute_Pgrid_price[0:96] + np.random.normal(0,0.1,96), 1, 7)

        else:
            if len(price_data)<self.max_time_step:
                self.execute_Pgrid_price = self.Pgrid_price
            else:
                self.execute_Pgrid_price = price_data 

        # ------------------setting start time ------------------ #
        if self.mode == "train":
            self.T_cur = np.random.randint(24,32)
            # self.T_cur = np.random.randint(0,10)
            self.select_pile = select_pile
            self._train_pre_set(select_pile = select_pile)
        else:
            self.select_pile = select_pile # not use at test
            self.T_cur = 6*4

        # ------------------- For plot data ------------------- #
        self.plot_data = {
            "Price"   : self.execute_Pgrid_price,
            "SOC"     : { agent:[0]*self.T_cur for agent in self.possible_agents },
            "SOC_T"   : { agent:[0]*self.T_cur for agent in self.possible_agents },
            "P_action": { agent:[0]*self.T_cur for agent in self.possible_agents },
            "P_upper" : { agent:[0]*self.T_cur for agent in self.possible_agents },
            "P_lower" : { agent:[0]*self.T_cur for agent in self.possible_agents },
            "P_use"   : { agent:[0]*self.T_cur for agent in self.possible_agents },
            "P_max_station" : self.P_max_station,
            "P_total" : [0]*self.T_cur
        }

        # ----------------------------------------------------- #
        # self._move_to_next_time_step()
        self._move_to_next_iter()
        
        for agent in self.possible_agents:
            if self.pile[agent] == None:
                self.observations[agent] = np.zeros(self.obs_len)
                done = True
            elif self.pile[agent].Ts < 1:
                done = True
            else:
                done = False

            self.terminations[agent] = done

        self.agent_selection = self._agent_selector.next()
            
        self._deads_step_first()
        
        self.observations[self.agent_selection] = self._get_agent_obs(self.agent_selection)

    def step(self, action):
        if ( self.terminations[self.agent_selection] or self.truncations[self.agent_selection] ):
            self.pile[self.agent_selection] = None
            self._was_dead_step(action)

            if len(self._agent_selector.agent_order):
                if self.agent_selection not in self._agent_selector.agent_order:
                    
                    self._agent_selector.reinit(self._agent_selector.agent_order)
                    self.agent_selection = self._agent_selector.next()
                    self.observations[self.agent_selection] = self._get_agent_obs(self.agent_selection)
            return
        
        # pile state  [ T_to_T| T_stay| T_to_L|  T_to_U|   P_t_L|  P_t_U]
        now_agent = self.agent_selection
        agent_action = action[0]
        # used power ----------
        P_min = self.observations[now_agent][4]
        P_max = self.observations[now_agent][5]

        # charge used power
        soc_power = ( (agent_action + 1)/2*(P_max - P_min) + P_min )
        charge_power = soc_power*self.pile[now_agent].P_distribute

        if self.mode != "train":
            self.P_plot_action[now_agent] = soc_power
        # charge_soc ----------
        self.pile[now_agent].charge(charge_power)

        if charge_power >= 0:
            used_power = charge_power/self.pile[now_agent].CE
        else:
            used_power = charge_power*self.pile[now_agent].CE
        
        self.P_already[now_agent] = used_power

        self.charge_power_max[now_agent] = 0

        if self._agent_selector.is_last():
            # ----------------- calculate reward for all agent ------------------- #
            self.step_reward = { agent : 0 for agent in self.possible_agents }
            # ----------------- calculate reward for all agent ------------------- #
            for agent in self.agents:
                # pile state  [ T_to_T| T_stay| T_to_L|  T_to_U|   P_t_L|  P_t_U]
                if self.observations[agent][0] >= 0:
                    efficiency_1 = 1/self.pile[agent].CE
                else:
                    efficiency_1 = self.pile[agent].CE

                T_to_target_ = self.pile[agent].get_to_request_time()
                
                if T_to_target_ >= 0:
                    efficiency_2 = 1/self.pile[agent].CE
                else:
                    efficiency_2 = self.pile[agent].CE

                mean_cost = self.observations[agent][0]*self.pile[agent].mean_price*efficiency_1
                
                do_action_cost = (self.observations[agent][6]*self.P_already[agent]/self.pile[agent].P_distribute + T_to_target_*self.pile[agent].mean_price_after_charge*efficiency_2)

                # reward = (mean_cost - do_action_cost)*10
                reward = (mean_cost - do_action_cost)
                
                self.rewards[agent] = reward
                self.step_reward[agent] = reward
                
            self.total_cost += sum(self.P_already.values())*self.execute_Pgrid_price[self.T_cur]/4

            self._check_env_done()
            if not self.env_done:
                # self._move_to_next_time_step()
                self._move_to_next_iter()
            # ----------------- get next state ------------------- #
            for agent in self.possible_agents:
                if self.pile[agent] == None:
                    self.observations[agent] = np.zeros(self.obs_len)
                    done = False
                elif  self.pile[agent].Ts < 1:
                    self.observations[agent] = self._get_agent_obs(agent)
                    done = True
                else:
                    done = False

                self.terminations[agent] = done

            self.truncations  = { agent : ( self.T_cur >= self.max_time_step ) for agent in self.agents }
        else:
            self._clear_rewards()

        if len(self._agent_selector.agent_order):
            self.agent_selection = self._agent_selector.next()

        self._deads_step_first()
        self.observations[self.agent_selection] = self._get_agent_obs(self.agent_selection)
        # self._accumulate_rewards()

if __name__ == "__main__":
    # env = StationEnv(mode = "test", N_piles = 10)
    env = StationEnv(mode = "train", N_piles = 10)
    env.seed(0)

    env.reset( select_pile = np.random.choice(env.possible_agents))
    # print(env.select_pile)
    print(env.execute_Pgrid_price)
    time_step = env.T_cur
    env.render()

    # pile state [ T_to_request, T_stay, T_to_lower, T_to_upper, P_t_lower, P_t_upper]
    print(f"\n----- Time_step : {env.T_cur//4}-{env.T_cur%4}({env.T_cur}) {env.charge_sequence} -----")
    print(f"              [ T_to_T| T_stay| T_to_L|  T_to_U|   P_t_L|  P_t_U]")
    
    print(env.agents)
    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()
        # print(obs)

        if termination or truncation:
            action = None
        else:
            action = env.action_space(agent).sample()
            if action >= 0:
                action = [-1]
            else:
                action = [-1]

        env.step(action)

        # ------------- print data --------------- #
        if action != None:
            action = np.round(action,2)
            
        print(f"{agent:8} - obs[ {str(np.round(obs[0],2)):6}| {str(int(obs[1])):6}| {str(np.round(obs[2],2)):6}|  {str(np.round(obs[3],2)):6}|  {str(np.round(obs[4],2)):6}| {str(np.round(obs[5],2)):6}| {str(np.round(obs[6],2)):6}] - action:{action}")
        if time_step != env.T_cur:
            print('reward_:',env.step_reward)
            env.render()

            print(f"----- Time_step : {env.T_cur//4}-{env.T_cur%4}({env.T_cur}) {env.charge_sequence} -----")
            print(f"              [ T_to_T| T_stay| T_to_L|  T_to_U|   P_t_L|  P_t_U]")
            time_step = env.T_cur
            
    print(env.total_cost)