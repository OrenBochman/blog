from shiny import App, render, ui
import agentpy as ap

await micropip.install("scipy")
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
import numpy as np
import seaborn as sns

app_ui = ui.page_fluid(
    ui.input_slider("agents", "Agents", 0, 100, 100),
    ui.input_slider("steps", "Steps", 0, 100, 100),
    ui.input_slider("x", "y", 0, 10, 10),
    ui.input_slider("y", "y", 0, 10, 10),
    ui.output_text_verbatim("txt"),
)

def server(input, output, session):
    @output
    @render.text    
    def txt():
        parameters = {
           'agents': input.agents(),
            'steps': input.steps(),
            'seed': 42,
        }
        model = WealthModel(parameters)
        results = model.run()
        return results.info['completed_steps'] 