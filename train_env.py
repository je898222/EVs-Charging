from env.charge_station_env_zero import StationEnv

import numpy  as np
import gymnasium as gym

class TrainSinglePileEnv(gym.Env):
    def __init__(self, N_piles = 10):
        super(TrainSinglePileEnv, self).__init__()
        self.N_piles = N_piles
        self.station_env = StationEnv(mode = "train", N_piles = self.N_piles)
        
        self.observation_space = self.station_env.observation_spaces['pile_1']

        self.action_space = self.station_env.action_spaces['pile_1']

        self.model = None

    def seed(self, seed):
        np.random.seed(seed)
        self.station_env.seed(seed)

    def get_action(self, obs, termination, truncation):
        if termination or truncation:
            action = None
        else:
            action = self.action_space.sample()

        return action

    ###########################################################################
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # reseting environment
        self.pile_id = np.random.choice(self.station_env.possible_agents)
        self.station_env.reset(select_pile = self.pile_id)

        # run other agent until the train pile
        for agent in self.station_env.agent_iter():
            obs, _, termination, truncation, info = self.station_env.last()

            if agent == self.pile_id:
                # print(f"--- stop at {agent} in reset_step ---")
                break
            
            # interact other pile
            action = self.get_action( obs, termination, truncation)
            self.station_env.step(action)

        self.state = obs
        self.time_step = self.station_env.T_cur

        return ( self.state, {} )
    
    ###########################################################################
    def step(self, action):
        # print(action)
        self.reward = 0
        self.station_env.step(action)
        for agent in self.station_env.agent_iter():
            obs, _, termination, truncation, _ = self.station_env.last()
            if agent == self.pile_id:
                break

            # interact other pile
            self.station_env.step(self.get_action( obs=obs, termination=termination, truncation=truncation ))

        self.reward = self.station_env.step_reward[self.pile_id]
        self.state  = obs

        # if (self.station_env.pile[self.pile_id].car_Ts <= 0):
        if (self.station_env.pile[self.pile_id].Ts <= 0):
            done = True
        else:
            done = False

        # print('reward:',self.reward)

        # truncated = False
        return  ( self.state, self.reward, done, False, {} )
    
if __name__ == "__main__":
    ################################
    mode = 0
    ################################
    env = TrainSinglePileEnv(N_piles = 10)

    obs , _ = env.reset()

    print(f"              [ T_to_T| T_stay| T_to_L|  T_to_U|   P_t_L|  P_t_U| price]")

    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _ , _ = env.step(action)
        print(f"action:{action}  reward:{reward}")
        print(f"{env.pile_id:8} - obs[ {str(np.round(obs[0],2)):6}| {str(int(obs[1])):6}| {str(np.round(obs[2],2)):6}|  {str(np.round(obs[3],2)):6}|  {str(np.round(obs[4],2)):6}| {str(np.round(obs[5],2)):6}| {str(np.round(obs[6],2)):6}] - action:{action}")
        print('===========')