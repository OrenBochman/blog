
# Model design
class WealthAgent(ap.Agent):
    """ An agent with wealth """
    def setup(self):
        self.wealth = 1

    def wealth_transfer(self):
        if self.wealth > 0:
            partner = self.model.agents.random()
            partner.wealth += 1
            self.wealth -= 1

def gini(x):
    """ Calculate Gini Coefficient """
    x = np.array(x)
    mad = np.abs(np.subtract.outer(x, x)).mean()  # Mean absolute difference
    rmad = mad / np.mean(x)  # Relative mean absolute difference
    return 0.5 * rmad




class WealthModel(ap.Model):
    """ A simple model of random wealth transfers """
    def setup(self):
        self.agents = ap.AgentList(self, self.p.agents, WealthAgent)
    def step(self):
        self.agents.wealth_transfer()
    def update(self):
        self.record('Gini Coefficient', gini(self.agents.wealth))
    def end(self):
        self.agents.record('wealth')

