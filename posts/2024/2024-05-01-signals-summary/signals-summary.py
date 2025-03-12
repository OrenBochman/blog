'''
  Lewis Signaling Game Model
  --------------------------

  The Lewis signaling game is a model of communication between two agents, a sender and a receiver.
  Nature picks a state, the sender observes the state, chooses a signal, and sends the signal to the receiver who then takes an action based on the signal.
  If the action of the receiver is a match with the state obseved by the sender, agents get a reward of 1, otherwise, they get a reward of 0.
   state, the sender and receiver get a reward of 1, otherwise, they get a reward of 0.
  is a match with the state, the sender and receiver get a reward of 1, otherwise, they get a reward of 0.

'''


import itertools
import functools
from mesa import Agent, Model
from mesa.time import StagedActivation, RandomActivation
from mesa.datacollection import DataCollector

#import random

# agent_roles
r_nature = 'nature'
r_sender = 'sender'
r_receiver = 'receiver'

class HerrnsteinRL():
    '''
                                    The Urn model
     nature            sender                 reciever     reward
                       
    | (0) | --{0}-->  | (0_a)  | --{a}--> | (a_0) | --{0}-->   1   
    |     |           | (0_b)  | --{b}    | (a_1) | --{1}-->   0
    |     |           +--------+    | +-->+-------+
    |     |                         +-|-+  
    | (1) | --{1}-->  | (1_a)  | --{a}+ +>| (b_0) | --{1}-->   1
    |     |           | (1_b)  | --{b}--->| (b_1) | --{0}-->   0
    +-----+           +--------+          +-------+
    
    
    Herrnstein urn algorithm
    ------------------------
    
    1. nature picks a state 
    2. sender  gets the state, chooses a signal by picking a ball in choose_option() from the stat'es urn
    3. receiver gets the action, chooses an action by picking a ball in choose_option()
    4. the balls in the urns are incremented if action == state
    5. repeat
    
    '''
    def __init__(self, options, learning_rate=1.0,verbose=False,name='Herrnstein matching law', balls=None):
        
        # filter options in choose option by input
        self.verbose = verbose
        self.name=name
        self.learning_rate = learning_rate
        self.options = options
        if balls is not None:
          self.balls = balls
        else:
          self.balls = {option: 1.0 for option in self.options}
        if self.verbose:
          print(f'LearningRule.__init__(Options: {options})')
    
    def get_filtered_urn(self, filter):
      ''' filters urn's options by prefix and normalizes the weights
          usege:
          urn=urn.get_filtered_urn(1)
          choice = model.random.choice(list(urn.keys()), p=list(urn.values()))
      '''
      assert type(filter) == int, f"filter must be a int"
      filtered_options = [key for key in self.balls.keys() if key[0] == filter]
      if not filtered_options:
        raise ValueError(f"No options found with filter {filter}")
      if self.verbose:
        print(f"in get_filtered_urn({filter=}) --- filtered_options: {filtered_options=}")
      filtered_balls = {opt: self.balls[opt] for opt in filtered_options}
      if self.verbose:
        print(f"in get_filtered_urn({filter=}) --- filtered_balls: {filtered_balls=}")
      #total = functools.reduce(lambda a,b: a+b, filtered_balls.values())
      total = sum(filtered_balls.values())
      if self.verbose:
        print(f"in get_filtered_urn({filter=}) --- total: {total=}")
      assert total > 0.0, f"total weights is {total=} after {filter=} on {self.balls}"      
      normalized_balls = {option: weight / total for option, weight in filtered_balls.items()}
      if self.verbose:
        print(f"in get_filtered_urn({filter=}) --- returning : {normalized_balls=}")
      return normalized_balls
     
    def choose_option(self,filter,random):
        ''' chooses an option from the urn based on the filter and the random choice
            
            usage:
            urn.choose_option(filter=1,random=model.random)
        '''
       
        urn = self.get_filtered_urn(filter)
        if random:
          options = random.choices(list(urn.keys()), weights=list(urn.values()),k=1)
          option = options[0]
          
          if self.verbose:
            print(f'in HerrnsteinRL.choose_option({filter=}) --- chose {option=} from {urn=}')

          return option
        else:
          raise Exception(f"random must be a random number generator")
        
    def update_weights(self, option, reward):
        old_balls = self.balls[option]
        self.balls[option] += self.learning_rate * reward 
        if self.verbose:
          print(f"Updated weight for option {option}: {old_balls} -> {self.balls[option]}")


class LewisAgent(Agent):
  
    def __init__(self, unique_id, model, game, role, verbose=False):
        super().__init__(unique_id, model)
        self.role = role #( one of nature, sender, receiver)
        self.verbose = verbose
        self.game = game
        self.messages = []
        self.actions = []
        if role == "sender":
          self.urn = HerrnsteinRL(model.states_signals, learning_rate=1.0,verbose=verbose,name='state_signal_weights')
        elif role == "receiver":
          self.urn = HerrnsteinRL(model.signals_actions, learning_rate=1.0,verbose=verbose,name='signal_action_weights')
        else:
          self.urn = None
        
    def step(self):
      # reset agent state before step
      self.messages = []
      self.actions = []

    def gen_state(self)-> None:
        if self.role == r_nature:
          self.current_state = model.random.choice(self.model.states)
          if self.verbose:
                print(f"Nature {self.unique_id} set state {self.current_state}")
                
    @property
    def state(self):
        if self.role == r_nature:
          return self.current_state

    def choose_signal(self, filter):
        if self.role != r_sender:
          throw(f"Only sender can send signals")
        self.option = self.urn.choose_option(filter=filter,random=self.model.random)
        signal = self.option[1] # the prefix is the urn context we want the suffix
        assert type(signal) == int, f"signal {signal=} must be a int"
        self.signal = signal
        if self.verbose:
              print(f"Sender {self.unique_id} got filter {filter} choose option: {self.option} and signaled: {self.signal}")
        return self.signal
          

    def send_signal(self, filter, receiver):
        ''' 
            # Message sending logic:
            1. sender chooses a signal based on the state
            2. sender sends the signal to the receiver
        '''
        if self.role != r_sender:
          raise Exception(f"Only sender can send signals")
         
        assert type(filter) == int, f"filter must be a int"
        assert filter in model.states, f"filter must be a valid state"
        signal = self.choose_signal(filter=filter)
        assert signal is not None, f"signal must be a valid signal"
        if self.verbose:
          print(f"Sender {self.unique_id} chose signal: {signal}")
        receiver.messages.append(signal)
        if self.verbose:
          print(f"Sender {self.unique_id} sends signal: {signal} to receiver {receiver.unique_id}")

    def fuse_actions(self,actions):
        ''' 
            # Message fusion logic:
            1. single message:  if there is only one signal then the action is the action associated with the signal
            2. ordered messages: if there are multiple signals then the action is the number from the string associated with the concatenated signal
               if there are two signals possible per message we concat and covert binary string to number
            3. is the messages are sets we could perform a intersection and take the action associated with the intersection 
               currently this is not implemented
            4. support for recursive signals is currently under research .
        ''' 
        if self.role != r_receiver:
          raise Exception(f"Only receiver can set actions")
        
        if len(actions) == 1: # single action no need to fuse
          return actions[0]
        else:
          # fuse the actions into a binary number
          action = 0
          # if there are multiple signals
          for i in range(len(actions)):
            action += actions[i]*(2**i)
          if self.verbose:
              print(f"Receiver {self.unique_id} fused actions : {self.actions} into action: {action}")
          return action

    def decode_message(self,signal):
        ''' first we need to get the filtered urn for the signal
            and then choose the option based on the urn'''
        if self.role != r_receiver:
          raise Exception(f"Only receiver can decode messages")
        option = self.urn.choose_option(filter=signal,random=self.model.random)
        action = option[1]
        if self.verbose:
              print(f"in decode_message({signal=}) Receiver {self.unique_id} got option: {option} and decoded action: {action}")
        return action

    def set_action(self):
        ''' first we need to use the urn to decode the signals 
            then need to fuse them to get the action '''
        if self.role != r_receiver:
          raise Exception(f"Only receiver can set the action")
        self.actions = []
        for signal in self.messages:
          self.actions.append(self.decode_message(signal))          
        self.action = self.fuse_actions(self.actions)
        # which option to reinforce 
        self.option = (self.messages[0],self.action)
        if self.verbose:
              print(f"Receiver {self.unique_id} received signals: {self.messages} and action: {self.action}")
              
    def set_reward(self,reward):
        if self.role not in [r_receiver,r_sender]:
          raise Exception(f"Only sender and receiver can set rewards")
        self.reward = reward
        if self.verbose:
            print(f"Receiver {self.unique_id} received reward: {self.reward}")
                
    def calc_reward(self,correct_action):
        if self.role != r_receiver:
          raise Exception(f"Only receiver can calculate rewards")
        self.reward = 1 if self.action == correct_action else 0
        
        

class SignalingGame(Model):
  
    # TODO: add support for 
    # 1. bottle necks
    # 2. rename k to state_count
    # 3. state_per_sender = state_count/sender_count 
    # 2. partitioning states by signals => state/sender_count

    def __init__(self, game_count=2, senders_count=1, recievers_count=1, state_count=3,signal_count=3,verbose=True):
        super().__init__()
        self.verbose = verbose
        self.schedule = RandomActivation(self)
        
        
        # Define the states, signals, and actions
        self.states   = [i for i in range(state_count)]
        print(f'{self.states=}')
        self.signals  = [i for i in range(signal_count)]
        print(f'{self.signals=}')
        self.actions  = [i for i in range(state_count)]
        print(f'{self.actions=}')
        
        # e.g., 1 -> 1, 2 -> 2, ...
        self.states_signals =  [(state,signal) for state in self.states for signal in self.signals]
        print(f'{self.states_signals=}')
        self.signals_actions = [(signal,action) for signal in self.signals for action in self.actions] 
        print(f'{self.signals_actions=}')
        
        # Agents

        self.uid=0
        self.senders_count=senders_count
        self.recievers_count=recievers_count

        # Games each game has a nature, senders and receivers
        self.games = []
        # Create games        
        for i in range(game_count):
            game = {
              r_nature: None,
              r_sender: [],
              r_receiver: []
            }
            
            # create nature agent
            game[r_nature] = LewisAgent(self.uid, self, game=i,role = r_nature,verbose=self.verbose)
            self.schedule.add(game[r_nature])
            self.uid += 1
            
            # create sender agents
            for j in range(senders_count):
                sender = LewisAgent(self.uid, self, game=i,role = r_sender,verbose=self.verbose)
                game[r_sender].append(sender)
                self.schedule.add(sender)
                self.uid +=1
                
            # create receiver agents
            for k in range (recievers_count):
                reciever = LewisAgent(self.uid, self, game=i,role = r_receiver,verbose=self.verbose)
                game[r_receiver].append(reciever)
                self.schedule.add(reciever)
                self.uid +=1
                
            self.games.append(game)

            self.total_reward = 0
        

        # Define what data to collect
        self.datacollector = DataCollector(
            model_reporters={"TotalReward": lambda m: m.total_reward},  # A function to call 
            agent_reporters={"Reward": "reward"}  # An agent attribute
        )

    def compute_total_reward(self,model):
        return 
        
    def step(self):
      
        for agent in model.schedule.agents:
            # reset agent state before step
            agent.step()
            
        for game_counter, game in enumerate(self.games):
            if self.verbose:
                print(f"--- Step {model.step_counter} Game {game_counter} ---")
            nature = game[r_nature]
            nature.gen_state()
            state = nature.current_state
            assert type(state) == int, f"state must be a int"
            assert state in model.states, f"state must be a valid state"
            if self.verbose:
                print(f"in model.step() --- game {game_counter} --- Nature {agent.unique_id} set state {state} in game {game_counter}")
            for sender in game[r_sender]:
                for receiver in game[r_receiver]:                    
                    sender.send_signal(filter = state, receiver=receiver)
            for receiver in game[r_receiver]:
                assert receiver.role == r_receiver, f"receiver role must be receiver not {receiver.role}"
                receiver.set_action()
                if self.verbose:
                    print(f"in model.step() --- game {game_counter} --- Receiver {receiver.unique_id} action: {receiver.action}")
                receiver.calc_reward(correct_action=state)
                reward = receiver.reward
                assert type(reward) == int, f"reward must be a int not {type(reward)}"
                assert reward in [0,1], f"reward must be 0 or 1 not {reward}"
                print(f"in model.step() --- game {game_counter} --- Receiver {receiver.unique_id} received reward: {receiver.reward}")
            
            for agent in itertools.chain(game[r_sender],game[r_receiver]):
                agent.set_reward(reward)
                if self.verbose:
                    print(f"in model.step() --- game {game_counter} --- Sender {agent.unique_id} received reward: {reward}")
                agent.urn.update_weights(agent.option, reward)

            #print(f'in model.step() --- game {game_counter}, {self.expected_rewards(game)=}')
                    # Collect data
        
        self.total_reward += sum(agent.reward for agent in self.schedule.agents if agent.role == r_receiver)

        self.datacollector.collect(self)


    def expected_rewards(self,game):
      return 0.25

    def run_model(self, steps):

        """Run the model until the end condition is reached. Overload as
        needed.
        """
        while self.running:
            self.step()
            steps -= 1
            if steps == 0:
                self.running = False



# Running the model
k = 2
state_count= 3  # Number of states, signals, and actions
signal_count= 3
steps = 1000
model = SignalingGame(senders_count=1,recievers_count=1,state_count=state_count,signal_count=signal_count,verbose=False,game_count=2)
model.step_counter = 0
for i in range(steps):
    model.step_counter +=1
    model.step()

import matplotlib.pyplot as plt

# Assuming `model` is your instance of LewisSignalingGame
model.run_model(100)  # Run the model for 100 steps

# Get the reward data
reward_data = model.datacollector.get_model_vars_dataframe()

# Plot the data
plt.figure(figsize=(10, 8))
plt.plot(reward_data['TotalReward'])
plt.xlabel('Step')
plt.ylabel('Total Reward')
plt.title('Total Reward over Time')
plt.grid(True)  # Add gridlines
plt.xlim(left=0)  # Start x-axis from 0
plt.ylim(bottom=0,top=1000)  # Start y-axis from 0
plt.show()
   