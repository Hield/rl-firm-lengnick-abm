import numpy as np
import mesa
from mesa.datacollection import DataCollector

from .firm import Firm, LearningFirm
from .household import Household


def compute_gini(model):
    x = sorted([hh.liquidity for hh in model.households])
    n = len(x)
    # Calculate using the standard formula for Gini coefficient
    b = sum(xi * (n - i) for i, xi in enumerate(x)) / (n * sum(x))
    return 1 + (1 / n) - 2 * b


def count_employed(model):
    return sum([hh.employer is not None for hh in model.households])


def count_vacancies(model):
    return sum([f.open_position for f in model.firms])


def sum_inventory(model) -> int:
    return sum([f.inventories for f in model.firms])


def average_goods_price(model):
    prices = [f.goods_price for f in model.firms]
    return sum(prices) / len(prices)


def average_wage_rate(model):
    wage_rates = [f.wage_rate for f in model.firms]
    return sum(wage_rates) / len(wage_rates) / model.parameters.month_length


def average_unsatisfied_demand(model):
    unsatisfied_demands = [hh.unsatisfied_demand / hh.planned_expenditure if hh.planned_expenditure else 0
                           for hh in model.households]
    return sum(unsatisfied_demands) / len(unsatisfied_demands)


class BaselineModel(mesa.Model):
    '''A baseline economic model based on the paper by M. Lengnick
    '''

    def __init__(self,
                 parameters,
                 seed=None):
        '''Initialize the model.
        '''
        super().__init__(seed=seed)
        self.parameters = parameters
        self.day = 0
        self.burned_in = False

        # Set up data collection
        daily_reporters = {
            'Gini': compute_gini,
            'Inventory': sum_inventory,
        }

        monthly_reporters = {
            'Employed': count_employed,
            'Vacancies': count_vacancies,
            'Inventory': sum_inventory,
            'Price': average_goods_price,
            'Wage': average_wage_rate,
            'Unsatisfied demand': average_unsatisfied_demand
        }

        self.daily_data_collector = DataCollector(daily_reporters)
        self.monthly_data_collector = DataCollector(monthly_reporters)

        # Create firms
        Firm.create_agents(self,
                           self.parameters.n_firms - self.parameters.n_learning_firms,
                           inventories=0,
                           liquidity=self.parameters.firm.initial_liquidity,
                           liquidity_buffer=0,
                           open_position=self.parameters.firm.initial_open_position,
                           goods_price=self.parameters.firm.initial_goods_price,
                           wage_rate=self.parameters.firm.initial_wage_rate)

        # Same starting condition as baseline firms for burn in period
        LearningFirm.create_agents(self,
                                   self.parameters.n_learning_firms,
                                   inventories=0,
                                   liquidity=self.parameters.firm.initial_liquidity,
                                   liquidity_buffer=0,
                                   open_position=self.parameters.firm.initial_open_position,
                                   goods_price=self.parameters.firm.initial_goods_price,
                                   wage_rate=self.parameters.firm.initial_wage_rate)                   

        # Create households
        Household.create_agents(self,
                                self.parameters.n_households,
                                reservation_wage=self.parameters.household.initial_reservation_wage,
                                liquidity=self.parameters.household.initial_liquidity)

        # Create shortcuts AgentSet for performance bonus
        # Firms need to be agents.select instead of agents_by_type to include LearningFirm
        self.firms = self.agents.select(agent_type=Firm)
        self.households = self.agents_by_type[Household]
        self.learning_firms = [] if self.parameters.n_learning_firms == 0 else self.agents_by_type[LearningFirm]

        # Initial conditions of the firms and households
        for household in self.households:
            household.suppliers = self.random.sample(self.firms, self.parameters.household.n_suppliers)

        # Initial employment rate
        for household in self.households:
            if self.random.random() < self.parameters.initial_employment_rate:
                employer = self.random.choice(self.firms)
                household.employer = employer
                employer.employees.append(household)


    def burn_in(self) -> None:
        for _ in range(self.parameters.burn_in_months * self.parameters.month_length):
            self.step(collect_data=False)
        self.burned_in = True


    def step(self, collect_data=True) -> None:
        # Beginning of the month
        if self.day % self.parameters.month_length == 0:
            # Firm actions
            self.firms.do('fire_employee_on_notice')
            self.firms.do('set_wage_rate')
            self.firms.do('manage_workforce')
            self.firms.do('set_goods_price')

            # Household actions
            self.households.shuffle_do('search_for_cheaper_goods')
            self.households.shuffle_do('search_to_satisfy_demands')
            self.households.shuffle_do('find_employment')
            self.households.shuffle_do('budget_consumption')

        # Daily actions
        self.households.shuffle_do('purchase_goods')
        self.firms.shuffle_do('produce')

        # End of the month
        if (self.day + 1) % self.parameters.month_length == 0:
            # Firm actions
            self.firms.shuffle_do('pay_wages')
            self.firms.shuffle_do('build_buffer')
            self.firms.shuffle_do('pay_profits')
            
            # Household actions
            self.households.shuffle_do('adjust_reservation_wage')

            # Collect monthly data
            if collect_data:
                self.monthly_data_collector.collect(self)

            # Reset monthly stats
            self.firms.do('update_monthly_stats')
            self.households.do('update_monthly_stats')

        # Increment the day
        self.day = self.day + 1
        
        # Collect daily data
        if collect_data:
            self.daily_data_collector.collect(self)