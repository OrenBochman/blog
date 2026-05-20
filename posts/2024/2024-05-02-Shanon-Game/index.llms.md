In this note I want to start to explore one of my notions of compositionality of Lewis games with another game. In my thinking currently consider at least three senses for compositionality

1.  Learning a MARL `skill` in a way that agents can repurpose it across new tasks and setting.

2.  Learning a domain specific signaling system that serves the needs of a game framing a lewis signaling game. Serves here means:

    - marl \arg max(Returns \mid \text{given expected actions of other agents})
    - population dynamics \arg max(Progeny \mid \text{given expected actions of other agents})

3.  Learning a structured complex signaling of systems with features more like NLP and less like a bijection.

    1.  Learn **semantics** by break down the state into some structure and express the full state using some aggregation of the parts.
        - e.g. do it using a structured signal, e.g. a template with slots for signals
          - NOT X; AND X,Y; OR X,Y
          - X-big, X-big-er, X-big-est, X-small, X-small-er, X-small-est
          - X-1, X-2, X-3, X-4, X-5, X-6, X-7, X-8, X-9, X-10 for inflections of action-X
          - X-1, X-2, X-3, X-4, X-5, X-6, X-7, X-8, X-9, X-10 for inflections of noun-Y
    2.  Capability to **generalize** learned semantics to new states
        - map a state to a prefix category and a suffix disambiguation
        - the prefix can be a signal in partial equilibrium solution of the lewis signaling game.
        - the might be reuse of an existing signal or a new signal.
    3.  ability to use categories
    4.  ability to learn

4.  using a hieracial model also seems to be an option.

The stating point:

> One can represent a communication protocol as a decision tree.

So in the abstraction one way that a emergent communications might arise is using three modules:

1.  Lewis signaling games to model different coordination tasks like

    1.  the alphabet and or basic signals
    2.  a lexicon built on the alphabet
    3.  a grammar to support composition of lexemes into sentences that can express complex states.
    4.  coordinating of decision rules
        - the sender get a decision trees that map states to a string in the alphabet this is called an encoder.
        - receiver get a decision tree that maps the string to a state space, this is called a decoder The encoder-decoder might just serialize the state or it might do arbitrary complex transformations that include a compression, error detection and correction and even analysis of the agents environment required to pick the best action. However at this point we want a minimal that is restricted to serialization and deserialization, compression using a prefix code and perhaps error detection and correction.
    5.  coordinating grammar rules. We may also need a grammar to support parsing of complex signals. In a simple case we might be processing mathematical expression and need to ensure that the formulas are well formed. In more complex scenarios we may need enforce selectional restrictions and subcategorization frames. In the MVP I imagine a simple rule that allows nested clauses.
    6.  one idea for using lewis games to coordinate the decision rules if it can enumerate all possible trees and let agents pick one. Other more complex scenarios are possible too.

2.  a Shannon game to model the communication of information between agents in which they learn a shared communication protocol with using error detection and correction capabilities.

3.  a Chomsky game to model development of a shared grammar for complex signals.

    - A trivial version is to use concatenation of signals to form new signals.
    - A more powerful version is to use an ordered vector of signals. Using a simple prefix code this allows creation of a powerful morphology.  
    - Probably the minimalist option though is to use a rule that allows nested clauses. If the tree allows recursion we get what Humboldt called the infinite use of finite means.
    - Just having a recursive rule though might over-generate and we might require additional means to restrict the grammar by way of selectional restrictions[^1] and subcategorization frames [^2].

## Shannon Game

Sharon games are about emergence of randomized communication protocols.

A randomized communication protocol is a probability distribution over the set of possible deterministic communication protocols.

We can model any deterministic communication protocol as a pair of decision rees, one for the sender and one for the receiver. The sender’s decision tree maps each possible message to a signal, and the receiver’s decision tree maps each possible signal to a message.

Messages that the sender can send.

The sender samples a message from this distribution and sends it to the receiver. The receiver then uses a decoding function to map the received message back to the original signal. The goal of the game is for the sender and receiver to coordinate on a communication protocol that maximizes their payoff, which is typically based on the accuracy of message transmission and reception.

It is a protocol that uses randomness to encode and decode messages.

This randomness can be used to introduce redundancy in the message, which can help in error detection and correction.

``` python
import numpy as np

class CommunicationAgent:
    def __init__(self, num_strategies):
        self.num_strategies = num_strategies
        self.q_table = np.zeros((num_strategies, num_strategies))
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1
    
    def choose_strategy(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_strategies)
        else:
            return np.argmax(self.q_table.sum(axis=1))
    
    def update_q_values(self, sender_strategy, receiver_strategy, reward):
        max_future_q = np.max(self.q_table[receiver_strategy])
        current_q = self.q_table[sender_strategy, receiver_strategy]
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_future_q - current_q)
        self.q_table[sender_strategy, receiver_strategy] = new_q

# Simulation parameters
num_strategies = 8
num_iterations = 10000

# Initialize agents
alice = CommunicationAgent(num_strategies)
bob = CommunicationAgent(num_strategies)

for _ in range(num_iterations):
    sender_strategy = alice.choose_strategy()
    receiver_strategy = bob.choose_strategy()
    
    # Simulate message transmission and reception with noise
    # This is a placeholder for actual encoding/decoding logic
    success = np.random.rand() < 0.8  # Assume 80% chance of success
    
    reward = 1 if success else -1
    alice.update_q_values(sender_strategy, receiver_strategy, reward)
    bob.update_q_values(receiver_strategy, sender_strategy, reward)

print("Alice's Q-Table:\n", alice.q_table)
print("Bob's Q-Table:\n", bob.q_table)
```

    Alice's Q-Table:
     [[7.15178366 6.71847257 7.1310994  7.14822428 7.13215933 6.84735038
      6.81396269 6.89459472]
     [6.85645207 1.06654029 0.24473592 1.27265155 0.95768168 0.68862805
      1.94200363 1.33717061]
     [6.95050376 1.18477568 0.48241524 0.58790153 0.75648891 1.27801917
      1.15665294 1.21660783]
     [7.30498578 1.54072937 0.         0.         0.71082304 0.72038612
      1.69336523 0.68428729]
     [7.31975542 0.65898277 1.11535764 0.         1.3840257  0.
      0.24160232 0.68254447]
     [6.8931791  1.23249715 0.30461082 0.         0.69379703 1.23282532
      2.33063816 0.48254447]
     [6.97285343 0.48765364 0.68588987 0.35233284 0.71207596 0.6964486
      0.11219172 1.88987428]
     [7.14989089 0.66759575 0.         0.66752801 0.         0.71614712
      1.87169383 0.67007565]]
    Bob's Q-Table:
     [[7.12338754 6.60095964 6.89010818 7.13266896 7.07133169 6.62723482
      6.72302341 6.9400469 ]
     [6.88281297 1.03946698 1.19507183 1.56717324 0.63792784 1.26410653
      0.47528245 0.65430687]
     [7.16462756 0.18645768 0.48228587 0.         1.01206289 0.29427724
      0.68325133 0.        ]
     [7.0783756  1.16890947 0.51403209 0.         0.         0.
      0.29238722 0.68058449]
     [7.05784294 0.87706616 0.74481648 0.72720292 1.32370359 0.67931721
      0.7072117  0.        ]
     [6.93784502 0.69335407 1.31555933 0.72063482 0.         1.25623774
      0.66902232 0.69310786]
     [6.82274117 1.64766764 1.05496204 1.6756027  0.23500081 2.28691924
      0.1        1.89002455]
     [7.05144976 1.32809665 1.26646831 0.75609149 0.7007448  0.50598839
      1.8424545  0.67500976]]

This example illustrates a basic game-theoretic approach where the sender and receiver iteratively learn better strategies for encoding and decoding messages over a noisy channel. The reinforcement learning framework allows both parties to adapt and improve their protocols, enhancing the reliability of communication over time. This model can be extended and refined to include more sophisticated encoding/decoding techniques and more complex noise models.

``` python
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import numpy as np

def hamming_distance(a, b):
    return np.sum(a != b) / len(a)

class Sender(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.protocol = self.random_protocol()
    
    def random_protocol(self):
        # Define a random protocol for encoding (Identity function for now)
        return lambda msg: msg  
    
    def step(self):
        self.model.original_message = np.random.randint(0, 2, self.model.message_length)  # Generate a binary message
        encoded_message = self.protocol(self.model.original_message)
        self.model.sent_message = encoded_message

class Receiver(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.protocol = self.random_protocol()
    
    def random_protocol(self):
        # Define a random protocol for decoding (Identity function for now)
        return lambda msg: msg  
    
    def step(self):
        if self.model.sent_message is None:
            return  # **Avoid processing before sender has sent a message**
        
        # Convert to NumPy array to ensure bitwise operations work
        noisy_message = np.array(self.model.sent_message) ^ np.random.binomial(1, self.model.error_rate, self.model.message_length)
        recovered_message = self.protocol(noisy_message)
        self.model.recovered_message = recovered_message
        self.evaluate_performance()
    
    def evaluate_performance(self):
        original_message = self.model.original_message
        recovered_message = self.model.recovered_message
        distance = hamming_distance(original_message, recovered_message)
        self.model.payoff += self.model.recovery_payoff(distance)
        self.model.payoff += self.model.length_payoff(len(recovered_message))
        self.model.payoff += self.model.early_recovery_payoff(self.model.current_step)

class NoisyChannelModel(Model):
    def __init__(self, message_length=10, error_rate=0.1, max_steps=100):
        self.message_length = message_length
        self.error_rate = error_rate
        self.current_step = 0
        self.max_steps = max_steps
        self.payoff = 0
        self.running = True  # Fix: Initialize running status
        
        self.schedule = RandomActivation(self)
        
        self.original_message = np.random.randint(0, 2, self.message_length)  # Initialize first message
        self.sent_message = None
        self.recovered_message = None
        
        sender = Sender(1, self)
        receiver = Receiver(2, self)
        self.schedule.add(sender)
        self.schedule.add(receiver)
        
        self.datacollector = DataCollector(
            model_reporters={"Payoff": "payoff"}
        )
    
    def recovery_payoff(self, distance):
        return 1 - distance
    
    def length_payoff(self, length):
        return 1 / length if length > 0 else 0  # Avoid division by zero
    
    def early_recovery_payoff(self, step):
        return (self.max_steps - step) / self.max_steps
    
    def step(self):
        self.current_step += 1
        self.schedule.step()
        self.datacollector.collect(self)
        if self.current_step >= self.max_steps:
            self.running = False  # Stop the simulation

# Run the model
model = NoisyChannelModel()
while model.running:
    model.step()

# Retrieve results
results = model.datacollector.get_model_vars_dataframe()
print(results)
```

        Payoff
    0     0.00
    1     2.08
    2     4.15
    3     6.21
    4     8.26
    ..     ...
    95  141.65
    96  142.68
    97  143.50
    98  144.41
    99  145.31

    [100 rows x 1 columns]

    /home/oren/work/blog/.venv/lib/python3.10/site-packages/mesa/agent.py:52: FutureWarning:

    The Mesa Model class was not initialized. In the future, you need to explicitly initialize the Model by calling super().__init__() on initialization.

so this is a variant that uses a noisy channel model to simulate the transmission of messages between a sender and receiver. The agents have protocols for encoding and decoding messages, and the model tracks the performance of the communication system based on the accuracy of message recovery, message length, and early recovery. This example demonstrates how to model and analyze the performance of communication systems in the presence of noise and other challenges.

What we don’t have is a way to pick different protocols or to improve them over time.

I would break this down into a few steps: 1. identify the environmental factors that would encourage the agents to evolve diverse and efficient transmission protocols. a. noisy channels b. limited bandwidth c. limited computational resources d. time constraints e. risks of predation.

2.  allow agents randomly generate candidate protocols and evaluate their performance.

``` python
def random_protocol():
    # Define a random protocol for encoding/decoding
    return lambda msg: np.random.randint(0, 2, len(msg))

# which  would be used as follows

class Sender(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.protocol = random_protocol()
    
    def step(self):
        message = np.random.randint(0, 2, self.model.message_length)
        encoded_message = self.protocol(message)
        self.model.sent_message = encoded_message
```

This could be done by introducing reinforcement learning techniques to allow the agents to adapt and learn better encoding/decoding strategies based on feedback from the environment. This would enable the agents to optimize their protocols for improved communication performance in noisy channels.

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2024,
  author = {Bochman, Oren},
  title = {Shannon {Game}},
  date = {2024-05-02},
  url = {https://orenbochman.github.io/posts/2024/2024-05-02-Shanon-Game/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2024. “Shannon Game.” May 2. <https://orenbochman.github.io/posts/2024/2024-05-02-Shanon-Game/>.

[^1]: restrict sematic roles

[^2]: might restrict phrase elements to morphological categories in the lexicon with suitable features.
