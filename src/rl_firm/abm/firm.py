import mesa


class Firm(mesa.Agent):
    '''A baseline firm
    '''


    def __init__(self,
                 model,
                 liquidity: float,
                 liquidity_buffer: float,
                 inventories: float,
                 goods_price: float,
                 wage_rate: float,
                 open_position: bool,
                 this_month_demand: float = 0.0,
                 last_month_demand: float = 0.0):
        super().__init__(model)

        # ATTRIBUTES
        self.liquidity = liquidity
        self.liquidity_buffer = liquidity_buffer
        self.inventories = inventories
        self.goods_price = goods_price
        self.wage_rate = wage_rate
        self.open_position = open_position
        self.employees = []
        self.employee_on_notice = None

        # STATISTICS
        self.months_since_hire_failure = 0
        self.this_month_demand = this_month_demand
        self.last_month_demand = last_month_demand
        self.this_month_revenue = 0.0
        self.last_month_revenue = 0.0
        self.this_month_cost = 0.0
        self.last_month_cost = 0.0
        self.zero_profit_streak = 0

    #------------#
    # Properties #
    #------------#
    @property
    def parameters(self):
        return self.model.parameters.firm


    @property
    def employee_count(self) -> int:
        return len(self.employees)

    
    @property
    def customer_count(self) -> int:
        return len([hh for hh in self.model.households if self.unique_id in hh.supplier_ids])


    @property
    def profit_ranking(self) -> int:
        return sorted(self.model.firms, key=lambda f: f.last_month_profit, reverse=True).index(self) + 1


    @property
    def marginal_cost(self) -> float:
        return self.wage_rate / (self.parameters.lambda_technology * self.model.parameters.month_length)


    @property
    def is_bankrupt(self) -> bool:
        # no money, inventories and no employee
        return not self.liquidity and not self.liquidity_buffer and not self.inventories and not self.employee_count


    @property
    def last_month_profit(self) -> float:
        return self.last_month_revenue - self.last_month_cost


    #--------------------#
    # Internal functions #
    #--------------------#
    def _increase_wage(self) -> None:
        self.wage_rate = self.wage_rate * (1 + self.random.uniform(0, self.parameters.delta_wage_adjustment_rate))


    def _decrease_wage(self) -> None:
        self.wage_rate = self.wage_rate * (1 - self.random.uniform(0, self.parameters.delta_wage_adjustment_rate))


    #-------------#
    # ABM actions #
    #-------------#
    def fire_employee_on_notice(self):
        # Only fire employee if on notice
        if self.employee_on_notice:
            self.let_go(self.employee_on_notice)

    
    def set_wage_rate(self):
        # check if there was a open position last month (not fulfilled)
        if self.open_position:
            self._increase_wage()
        
        # check if should decrease wage
        if self.months_since_hire_failure >= self.parameters.gamma_consecutive_months:
            self._decrease_wage()


    def manage_workforce(self):
        # check if need to hire more
        if self.inventories < self.parameters.lower_phi_demand * self.last_month_demand:
            self.open_position = True
        # or check if need to put someone on notice
        elif self.inventories > self.parameters.upper_phi_demand * self.last_month_demand and self.employees:
            self.employee_on_notice = self.random.choice(self.employees)


    def set_goods_price(self):
        # Only set new price with probs theta_probs_set_new_price
        if self.random.random() >= self.parameters.theta_probs_set_new_price:
            return

        # Consider increasing price if not enough inventories
        if (self.inventories < self.parameters.lower_phi_demand * self.last_month_demand and
            self.goods_price <= self.parameters.upper_phi_marginal_cost * self.marginal_cost):
            self.goods_price = self.goods_price * (1 + self.random.uniform(0, self.parameters.theta_goods_price))

        # Else consider reducing the price
        if (self.inventories > self.parameters.upper_phi_demand * self.last_month_demand and
            self.goods_price >= self.parameters.lower_phi_marginal_cost * self.marginal_cost):
            self.goods_price = self.goods_price * (1 - self.random.uniform(0, self.parameters.theta_goods_price))


    def produce(self):
        self.inventories = self.inventories + self.parameters.lambda_technology * self.employee_count

    
    def pay_wages(self):
        # Bankrupt?
        try:
            # Transfer all the buffer to liquidity since firm build buffer after this anyway
            self.liquidity = self.liquidity + self.liquidity_buffer
            self.liquidity_buffer = 0
            if self.liquidity < self.wage_rate * self.employee_count:
                # Immediate wage cut
                new_wage = self.liquidity / self.employee_count
                # Can't pay if company is too broke
                # Take care of rounding error
                if new_wage < 1e-3:
                    return
                self.wage_rate = new_wage

            for employee in self.employees:
                # Take care of rounding errors
                salary = self.wage_rate if self.liquidity > self.wage_rate else self.liquidity
                self.liquidity = self.liquidity - salary
                employee.liquidity = employee.liquidity + salary
                self.this_month_cost = self.this_month_cost + salary
        except:
            print('Liquidity', self.liquidity)
            print('Buffer', self.liquidity_buffer)
            raise


    def build_buffer(self):
        ideal_buffer = self.parameters.chi_buffer * self.wage_rate * self.employee_count

        # Reorganize
        self.liquidity = self.liquidity + self.liquidity_buffer
        self.liquidity_buffer = 0

        if self.liquidity < ideal_buffer:
            ideal_buffer = self.liquidity

        self.liquidity = self.liquidity - ideal_buffer
        self.liquidity_buffer = self.liquidity_buffer + ideal_buffer


    def pay_profits(self):
        total_household_liquidity = sum([household.liquidity for household in self.model.households])
        total_payout = self.liquidity
        # Payout
        for household in self.model.households:
            try:
                payout = total_payout * (household.liquidity / total_household_liquidity)
                payout = payout if self.liquidity > payout else self.liquidity # Rounding errors
                household.liquidity = household.liquidity + payout
                self.liquidity = self.liquidity - payout
            except:
                print(payout)
                print(household.liquidity)
                raise


    def update_monthly_stats(self):
        self.last_month_demand = self.this_month_demand
        self.this_month_demand = 0
        
        # Check if unfilled position
        if self.open_position:
            self.months_since_hire_failure = 0
        else:
            self.months_since_hire_failure = self.months_since_hire_failure + 1

        # Stats management
        self.last_month_revenue = self.this_month_revenue
        self.last_month_cost = self.this_month_cost
        self.this_month_revenue = 0
        self.this_month_cost = 0

        if self.last_month_profit <= 0:
            self.zero_profit_streak = self.zero_profit_streak + 1
        else:
            self.zero_profit_streak = 0


    #-------------------#
    # Utility functions #
    #-------------------#
    def hire(self, employee):
        '''Hire a household'''
        self.open_position = False
        self.employees.append(employee)
        employee.employer = self


    def let_go(self, employee):
        '''Let go a household'''
        if self.employee_on_notice is employee:
            self.employee_on_notice = None

        self.employees.remove(employee)
        employee.employer = None


    def sell(self, amount, cost):
        ''' Selling utility function to be used in household agent actions'''
        self.liquidity = self.liquidity + cost
        self.inventories = self.inventories - amount
        self.this_month_revenue = self.this_month_revenue + cost


    def register_demand(self, demand):
        '''Register directed demand from households, used in household's purchasing action'''
        self.this_month_demand = self.this_month_demand + demand


class LearningFirm(Firm):
    '''A reinforcement learned firm optimizing for profit
    '''
    def apply_action(self, action):
        '''action is a dict with 3 keys: wage_adjustment, price_adjustment and hr_action
           Just store the action to be used in other functions
        '''
        self.action = action
        

    # Overriding all action that needs to be learned
    # Use the baseline strategy for the burn in period to give every firms an equal start
    def set_wage_rate(self):
        if not self.model.burned_in:
            super().set_wage_rate()
            return

        self.wage_rate = self.wage_rate * (1 + float(self.action['wage_adjustment']))

        
    def manage_workforce(self):
        if not self.model.burned_in:
            super().manage_workforce()
            return

        if self.action['hr_action'] == 0:
            if self.employees:
                self.employee_on_notice = self.random.choice(self.employees)
        elif self.action['hr_action'] == 1:
            pass # do nothing
        elif self.action['hr_action'] == 2:
            self.open_position = True
        else:
            raise NotImplementedError(f'Action {self.action["hr_action"]} not implemented')

    def set_goods_price(self):
        if not self.model.burned_in:
            super().set_goods_price()
            return

        # Keep calvo rigidity
        if self.random.random() >= self.parameters.theta_probs_set_new_price:
            return
        self.goods_price = self.goods_price * (1 + float(self.action['price_adjustment']))
        