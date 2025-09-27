#sugarscape.py


"""
Mesa Sugarscape Simulation with Configurable Rules

A reproduction of the classic Sugarscape model from "Growing Artificial Societies"
by Epstein and Axtell, implemented using the Mesa agent-based modeling framework.

Features:
- Basic movement and sugar collection
- Configurable vision range
- Metabolism and survival mechanics
- Optional reproduction rules
- Optional cultural transmission
- Optional disease mechanics
- Optional pollution rules
- Optional seasonal variations
- Optional combat/conflict rules

Usage:
    python sugarscape.py --help
"""

import argparse
import random
import numpy as np
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any


class SugarAgent(Agent):
    """An agent in the Sugarscape model."""
    
    def __init__(self, unique_id: int, model: 'SugarscapeModel', pos: Tuple[int, int]):
        super().__init__(unique_id, model)
        self.pos = pos
        
        # Basic attributes
        self.sugar = random.randint(5, 25)
        self.metabolism = random.randint(1, 4)
        self.vision = random.randint(1, 6)
        self.max_age = random.randint(60, 100) if model.enable_aging else float('inf')
        self.age = 0
        
        # Cultural attributes (for cultural transmission rule)
        if model.enable_culture:
            self.culture = [random.choice([0, 1]) for _ in range(model.culture_length)]
        
        # Disease attributes
        if model.enable_disease:
            self.diseases = set()
            self.immune_system = random.random()
        
        # Combat attributes
        if model.enable_combat:
            self.tribe = random.choice(['Red', 'Blue'])
        
        # Reproduction attributes
        if model.enable_reproduction:
            self.sex = random.choice(['M', 'F'])
            self.fertility_start = random.randint(12, 15)
            self.fertility_end = random.randint(40, 50) if self.sex == 'F' else random.randint(50, 60)
            self.children = []

    def get_vision_positions(self) -> List[Tuple[int, int]]:
        """Get all positions within agent's vision range."""
        positions = []
        x, y = self.pos
        
        # Check in four cardinal directions
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            for distance in range(1, self.vision + 1):
                new_x = (x + dx * distance) % self.model.width
                new_y = (y + dy * distance) % self.model.height
                positions.append((new_x, new_y))
        
        return positions

    def find_best_sugar_location(self) -> Tuple[int, int]:
        """Find the location with the most sugar within vision range."""
        current_x, current_y = self.pos
        visible_positions = self.get_vision_positions()
        visible_positions.append(self.pos)  # Include current position
        
        best_sugar = -1
        best_positions = []
        
        for pos in visible_positions:
            sugar_level = self.model.sugar_grid[pos[0]][pos[1]]
            
            # If considering pollution, reduce effective sugar
            if self.model.enable_pollution:
                pollution_level = self.model.pollution_grid[pos[0]][pos[1]]
                effective_sugar = max(0, sugar_level - pollution_level)
            else:
                effective_sugar = sugar_level
            
            if effective_sugar > best_sugar:
                best_sugar = effective_sugar
                best_positions = [pos]
            elif effective_sugar == best_sugar:
                best_positions.append(pos)
        
        # If multiple positions have same sugar, choose closest
        if len(best_positions) > 1:
            distances = []
            for pos in best_positions:
                dist = abs(pos[0] - current_x) + abs(pos[1] - current_y)
                distances.append((dist, pos))
            distances.sort()
            return distances[0][1]
        
        return best_positions[0] if best_positions else self.pos

    def move_and_collect(self):
        """Move to best sugar location and collect sugar."""
        new_pos = self.find_best_sugar_location()
        
        # Check if position is occupied (unless combat is enabled)
        if not self.model.enable_combat:
            occupants = self.model.grid.get_cell_list_contents([new_pos])
            if occupants:
                # Find unoccupied position with next best sugar
                visible_positions = self.get_vision_positions()
                for pos in sorted(visible_positions, 
                                key=lambda p: self.model.sugar_grid[p[0]][p[1]], 
                                reverse=True):
                    if not self.model.grid.get_cell_list_contents([pos]):
                        new_pos = pos
                        break
        
        # Move to new position
        self.model.grid.move_agent(self, new_pos)
        
        # Collect sugar
        sugar_collected = self.model.sugar_grid[new_pos[0]][new_pos[1]]
        self.sugar += sugar_collected
        self.model.sugar_grid[new_pos[0]][new_pos[1]] = 0
        
        # Add pollution if enabled
        if self.model.enable_pollution:
            self.model.pollution_grid[new_pos[0]][new_pos[1]] += self.model.pollution_rate

    def reproduce(self):
        """Attempt to reproduce if conditions are met."""
        if not self.model.enable_reproduction:
            return
        
        if not (self.fertility_start <= self.age <= self.fertility_end):
            return
        
        if self.sugar < self.model.reproduction_threshold:
            return
        
        # Find nearby agents of opposite sex
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, radius=1)
        potential_partners = [n for n in neighbors 
                            if isinstance(n, SugarAgent) 
                            and n.sex != self.sex 
                            and n.fertility_start <= n.age <= n.fertility_end
                            and n.sugar >= self.model.reproduction_threshold]
        
        if potential_partners:
            partner = random.choice(potential_partners)
            
            # Create child
            empty_cells = self.model.grid.get_neighborhood(self.pos, moore=True, radius=1)
            empty_cells = [cell for cell in empty_cells 
                         if not self.model.grid.get_cell_list_contents([cell])]
            
            if empty_cells:
                child_pos = random.choice(empty_cells)
                child_id = self.model.next_id()
                child = SugarAgent(child_id, self.model, child_pos)
                
                # Inherit traits from parents
                child.metabolism = random.choice([self.metabolism, partner.metabolism])
                child.vision = random.choice([self.vision, partner.vision])
                child.sugar = (self.sugar + partner.sugar) // 4
                
                # Parents lose sugar
                self.sugar -= child.sugar
                partner.sugar -= child.sugar
                
                # Cultural inheritance
                if self.model.enable_culture:
                    child.culture = []
                    for i in range(self.model.culture_length):
                        child.culture.append(random.choice([self.culture[i], partner.culture[i]]))
                
                self.model.grid.place_agent(child, child_pos)
                self.model.schedule.add(child)
                self.children.append(child)
                partner.children.append(child)

    def cultural_transmission(self):
        """Transmit culture with neighbors."""
        if not self.model.enable_culture:
            return
        
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, radius=1)
        agent_neighbors = [n for n in neighbors if isinstance(n, SugarAgent)]
        
        if agent_neighbors:
            other = random.choice(agent_neighbors)
            
            # Find different cultural traits
            different_indices = [i for i in range(self.model.culture_length) 
                               if self.culture[i] != other.culture[i]]
            
            if different_indices:
                # Probability of transmission based on similarity
                similarity = 1 - len(different_indices) / self.model.culture_length
                if random.random() < similarity:
                    # Transmit one random different trait
                    idx = random.choice(different_indices)
                    if random.random() < 0.5:
                        self.culture[idx] = other.culture[idx]
                    else:
                        other.culture[idx] = self.culture[idx]

    def combat(self):
        """Engage in combat with agents of different tribes."""
        if not self.model.enable_combat:
            return
        
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, radius=1)
        enemies = [n for n in neighbors 
                  if isinstance(n, SugarAgent) and n.tribe != self.tribe]
        
        if enemies:
            enemy = random.choice(enemies)
            
            # Combat based on sugar levels (wealth)
            if self.sugar > enemy.sugar:
                # Winner takes all sugar
                self.sugar += enemy.sugar
                # Remove enemy
                self.model.grid.remove_agent(enemy)
                self.model.schedule.remove(enemy)

    def contract_disease(self):
        """Potentially contract diseases from neighbors."""
        if not self.model.enable_disease:
            return
        
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, radius=1)
        infected_neighbors = [n for n in neighbors 
                            if isinstance(n, SugarAgent) and n.diseases]
        
        for neighbor in infected_neighbors:
            for disease in neighbor.diseases:
                if disease not in self.diseases:
                    # Probability of infection based on immune system
                    if random.random() > self.immune_system:
                        self.diseases.add(disease)

    def step(self):
        """Execute one step of the agent."""
        self.age += 1
        
        # Move and collect sugar
        self.move_and_collect()
        
        # Lose sugar due to metabolism
        self.sugar -= self.metabolism
        
        # Disease effects
        if self.model.enable_disease and self.diseases:
            disease_cost = len(self.diseases) * self.model.disease_cost
            self.sugar -= disease_cost
        
        # Die if no sugar or too old
        if self.sugar <= 0 or self.age >= self.max_age:
            self.model.grid.remove_agent(self)
            self.model.schedule.remove(self)
            return
        
        # Execute optional rules
        if random.random() < 0.1:  # 10% chance per step
            self.reproduce()
        
        if random.random() < 0.3:  # 30% chance per step
            self.cultural_transmission()
        
        self.combat()
        self.contract_disease()


class SugarscapeModel(Model):
    """The Sugarscape model."""
    
    def __init__(self, 
                 width: int = 50,
                 height: int = 50,
                 initial_population: int = 400,
                 enable_reproduction: bool = False,
                 enable_culture: bool = False,
                 enable_disease: bool = False,
                 enable_pollution: bool = False,
                 enable_seasonal: bool = False,
                 enable_combat: bool = False,
                 enable_aging: bool = True,
                 sugar_regrow_rate: int = 1,
                 culture_length: int = 5,
                 reproduction_threshold: int = 50,
                 disease_cost: int = 1,
                 pollution_rate: float = 1.0):
        
        super().__init__()
        self.width = width
        self.height = height
        self.initial_population = initial_population
        self.enable_reproduction = enable_reproduction
        self.enable_culture = enable_culture
        self.enable_disease = enable_disease
        self.enable_pollution = enable_pollution
        self.enable_seasonal = enable_seasonal
        self.enable_combat = enable_combat
        self.enable_aging = enable_aging
        self.sugar_regrow_rate = sugar_regrow_rate
        self.culture_length = culture_length
        self.reproduction_threshold = reproduction_threshold
        self.disease_cost = disease_cost
        self.pollution_rate = pollution_rate
        
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(width, height, True)
        
        # Initialize sugar distribution (two peaks)
        self.sugar_grid = np.zeros((width, height))
        self.max_sugar_grid = np.zeros((width, height))
        self.init_sugar_distribution()
        
        # Initialize pollution grid
        if self.enable_pollution:
            self.pollution_grid = np.zeros((width, height))
        
        # Initialize diseases
        if self.enable_disease:
            self.diseases = ['flu', 'cold', 'fever']
        
        # Create agents
        self._agent_counter = 0
        for _ in range(self.initial_population):
            self.create_agent()
        
        # Data collection
        self.datacollector = DataCollector(
            model_reporters={
                "Population": lambda m: m.schedule.get_agent_count(),
                "Average Sugar": lambda m: np.mean([a.sugar for a in m.schedule.agents]),
                "Average Age": lambda m: np.mean([a.age for a in m.schedule.agents]) if m.schedule.agents else 0,
                "Total Sugar": lambda m: np.sum(m.sugar_grid),
            },
            agent_reporters={
                "Sugar": "sugar",
                "Age": "age",
                "Vision": "vision",
                "Metabolism": "metabolism"
            }
        )
        
        self.datacollector.collect(self)

    def next_id(self) -> int:
        """Get next unique agent ID."""
        self._agent_counter += 1
        return self._agent_counter

    def init_sugar_distribution(self):
        """Initialize sugar distribution with two peaks."""
        center1 = (self.width // 4, self.height // 4)
        center2 = (3 * self.width // 4, 3 * self.height // 4)
        
        for x in range(self.width):
            for y in range(self.height):
                # Distance to nearest peak
                dist1 = np.sqrt((x - center1[0])**2 + (y - center1[1])**2)
                dist2 = np.sqrt((x - center2[0])**2 + (y - center2[1])**2)
                min_dist = min(dist1, dist2)
                
                # Sugar level decreases with distance from peaks
                sugar_level = max(0, 4 - min_dist / 5)
                self.sugar_grid[x][y] = sugar_level
                self.max_sugar_grid[x][y] = sugar_level

    def create_agent(self):
        """Create a new agent at random empty location."""
        empty_cells = []
        for x in range(self.width):
            for y in range(self.height):
                if not self.grid.get_cell_list_contents([(x, y)]):
                    empty_cells.append((x, y))
        
        if empty_cells:
            pos = random.choice(empty_cells)
            agent = SugarAgent(self.next_id(), self, pos)
            
            # Add initial diseases
            if self.enable_disease and random.random() < 0.1:
                agent.diseases.add(random.choice(self.diseases))
            
            self.grid.place_agent(agent, pos)
            self.schedule.add(agent)

    def regrow_sugar(self):
        """Regrow sugar on the landscape."""
        for x in range(self.width):
            for y in range(self.height):
                current_sugar = self.sugar_grid[x][y]
                max_sugar = self.max_sugar_grid[x][y]
                
                # Seasonal variation
                if self.enable_seasonal:
                    seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * self.schedule.steps / 50)
                    max_sugar *= seasonal_factor
                
                if current_sugar < max_sugar:
                    self.sugar_grid[x][y] = min(max_sugar, 
                                               current_sugar + self.sugar_regrow_rate)

    def step(self):
        """Execute one step of the model."""
        self.schedule.step()
        self.regrow_sugar()
        
        # Reduce pollution over time
        if self.enable_pollution:
            self.pollution_grid *= 0.99  # 1% decay per step
        
        self.datacollector.collect(self)


def agent_portrayal(agent):
    """Portrayal function for visualization."""
    if isinstance(agent, SugarAgent):
        portrayal = {"Shape": "circle", "Filled": "true", "r": 0.5}
        
        # Color based on sugar level
        sugar_level = agent.sugar
        if sugar_level > 30:
            portrayal["Color"] = "green"
        elif sugar_level > 15:
            portrayal["Color"] = "yellow"
        else:
            portrayal["Color"] = "red"
        
        # Size based on age
        portrayal["r"] = 0.3 + (agent.age / 100) * 0.4
        
        portrayal["Layer"] = 0
        return portrayal


def run_simulation(args):
    """Run the simulation with given arguments."""
    model = SugarscapeModel(
        width=args.width,
        height=args.height,
        initial_population=args.population,
        enable_reproduction=args.reproduction,
        enable_culture=args.culture,
        enable_disease=args.disease,
        enable_pollution=args.pollution,
        enable_seasonal=args.seasonal,
        enable_combat=args.combat,
        enable_aging=args.aging,
        sugar_regrow_rate=args.regrow_rate,
        culture_length=args.culture_length,
        reproduction_threshold=args.repro_threshold,
        disease_cost=args.disease_cost,
        pollution_rate=args.pollution_rate
    )
    
    print(f"Running Sugarscape simulation for {args.steps} steps...")
    print(f"Rules enabled: ", end="")
    rules = []
    if args.reproduction: rules.append("reproduction")
    if args.culture: rules.append("culture")
    if args.disease: rules.append("disease")
    if args.pollution: rules.append("pollution")
    if args.seasonal: rules.append("seasonal")
    if args.combat: rules.append("combat")
    if args.aging: rules.append("aging")
    print(", ".join(rules) if rules else "none")
    
    for step in range(args.steps):
        model.step()
        if step % 50 == 0:
            print(f"Step {step}: Population = {model.schedule.get_agent_count()}")
        
        if model.schedule.get_agent_count() == 0:
            print("All agents died!")
            break
    
    # Plot results
    model_data = model.datacollector.get_model_vars_dataframe()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    ax1.plot(model_data['Population'])
    ax1.set_title('Population Over Time')
    ax1.set_ylabel('Population')
    
    ax2.plot(model_data['Average Sugar'])
    ax2.set_title('Average Sugar Over Time')
    ax2.set_ylabel('Sugar')
    
    ax3.plot(model_data['Average Age'])
    ax3.set_title('Average Age Over Time')
    ax3.set_ylabel('Age')
    
    ax4.plot(model_data['Total Sugar'])
    ax4.set_title('Total Sugar in Environment')
    ax4.set_ylabel('Total Sugar')
    
    plt.tight_layout()
    plt.savefig('sugarscape_results.png')
    plt.show()
    
    print(f"\nFinal statistics:")
    print(f"Final population: {model.schedule.get_agent_count()}")
    if model.schedule.get_agent_count() > 0:
        print(f"Average sugar: {np.mean([a.sugar for a in model.schedule.agents]):.2f}")
        print(f"Average age: {np.mean([a.age for a in model.schedule.agents]):.2f}")


def run_server(args):
    """Run the visualization server."""
    grid = CanvasGrid(agent_portrayal, args.width, args.height, 500, 500)
    server = ModularServer(
        SugarscapeModel,
        [grid],
        "Sugarscape Model",
        {
            "width": args.width,
            "height": args.height,
            "initial_population": args.population,
            "enable_reproduction": args.reproduction,
            "enable_culture": args.culture,
            "enable_disease": args.disease,
            "enable_pollution": args.pollution,
            "enable_seasonal": args.seasonal,
            "enable_combat": args.combat,
            "enable_aging": args.aging,
            "sugar_regrow_rate": args.regrow_rate,
            "culture_length": args.culture_length,
            "reproduction_threshold": args.repro_threshold,
            "disease_cost": args.disease_cost,
            "pollution_rate": args.pollution_rate
        }
    )
    server.port = 8521
    server.launch()


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Mesa Sugarscape Simulation')
    
    # Basic parameters
    parser.add_argument('--width', type=int, default=50, help='Grid width')
    parser.add_argument('--height', type=int, default=50, help='Grid height')
    parser.add_argument('--population', type=int, default=400, help='Initial population')
    parser.add_argument('--steps', type=int, default=500, help='Number of simulation steps')
    
    # Rule flags
    parser.add_argument('--reproduction', action='store_true', 
                       help='Enable reproduction rule')
    parser.add_argument('--culture', action='store_true', 
                       help='Enable cultural transmission rule')
    parser.add_argument('--disease', action='store_true', 
                       help='Enable disease rule')
    parser.add_argument('--pollution', action='store_true', 
                       help='Enable pollution rule')
    parser.add_argument('--seasonal', action='store_true', 
                       help='Enable seasonal variation')
    parser.add_argument('--combat', action='store_true', 
                       help='Enable combat rule')
    parser.add_argument('--aging', action='store_true', default=True,
                       help='Enable aging (default: True)')
    
    # Rule parameters
    parser.add_argument('--regrow-rate', type=int, default=1, 
                       help='Sugar regrowth rate')
    parser.add_argument('--culture-length', type=int, default=5, 
                       help='Length of cultural vector')
    parser.add_argument('--repro-threshold', type=int, default=50, 
                       help='Sugar threshold for reproduction')
    parser.add_argument('--disease-cost', type=int, default=1, 
                       help='Sugar cost per disease per step')
    parser.add_argument('--pollution-rate', type=float, default=1.0, 
                       help='Pollution production rate')
    
    # Execution mode
    parser.add_argument('--visualize', action='store_true', 
                       help='Run visualization server instead of batch simulation')
    
    args = parser.parse_args()
    
    if args.visualize:
        print("Starting visualization server on http://localhost:8521")
        run_server(args)
    else:
        run_simulation(args)


if __name__ == "__main__":
    main()