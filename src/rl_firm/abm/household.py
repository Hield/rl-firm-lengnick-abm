import mesa


class Household(mesa.Agent):
    '''A household

    Reservation wage: omega_h
    Liquidity: m_h
    '''

    def __init__(self,
                 model,
                 reservation_wage: float,
                 liquidity: float,
                 suppliers: list = None,
                 employer: list = None):
        '''Create a new agent'''
        super().__init__(model)

        # ATTRIBUTES
        self.reservation_wage = reservation_wage
        self.liquidity = liquidity
        self.employer = employer
        self.suppliers = suppliers or []
        self.planned_expenditure = 0

        # STATISTICS
        self.last_month_constrained_suppliers = {}
        self.constrained_suppliers = {}
        self.unsatisfied_demand = 0

    @property
    def parameters(self):
        return self.model.parameters.household
    
    @property
    def supplier_ids(self):
        return [supplier.unique_id for supplier in self.suppliers]

    def search_for_cheaper_goods(self):
        if self.random.random() >= self.parameters.psi_price:
            return
        # Only try to find new supplier with psi_price * 100%
        evaluating_supplier = self.random.choice(self.suppliers)
        potential_firms = self.model.firms.select(lambda f: f not in self.suppliers)
        
        # Adding a scouting mechanism to avoid zombie firms
        #if self.random.random() <= 0.05:
        #    comparing_supplier = self.random.choice(potential_firms)
        #else:
            # Minimum 1 employee for weight calculation to avoid 0 total weighs
        comparing_supplier = self.random.choices(potential_firms, weights=[firm.employee_count + 1 for firm in potential_firms], k=1)[0]

        if 1 - comparing_supplier.goods_price / evaluating_supplier.goods_price >= self.parameters.xi_price_percentage:
            self.suppliers.remove(evaluating_supplier)
            # Remove from the constrained list if switching
            self.constrained_suppliers.pop(evaluating_supplier.unique_id, None)
            self.suppliers.append(comparing_supplier)
        
    
    def search_to_satisfy_demands(self):
        if not self.constrained_suppliers or self.random.random() >= self.parameters.psi_quant:
            return
        # Try to find new firms to satisfy demand if constrained
        bad_suppliers = self.last_month_constrained_suppliers.items()
        bad_supplier_id = self.random.choices([firm_id for firm_id, _ in bad_suppliers],
                                               weights=[float(constraint) for _, constraint in bad_suppliers],
                                               k=1)[0]
        potential_suppliers = self.model.firms.select(lambda f: f.unique_id not in self.supplier_ids)
        replacing_supplier = self.random.choice(potential_suppliers)
        og_suppliers = self.suppliers
        self.suppliers = [supplier for supplier in self.suppliers if supplier.unique_id != bad_supplier_id]
        self.suppliers.append(replacing_supplier)
        
        
    def find_employment(self):
        # Do nothing if satisfied and random > pi
        if self.employer and self.employer.wage_rate >= self.reservation_wage and self.random.random() > self.parameters.pi_explore_odd:
            return
        # Try more if unemployed
        tries = self.parameters.beta_firm_count if not self.employer else 1

        if self.employer:
            potential_employers = self.random.sample(self.model.firms.select(lambda f: f.unique_id != self.employer.unique_id), k=tries)
        else:
            potential_employers = self.random.sample(self.model.firms, k=tries)

        for potential_employer in potential_employers:
            # Do nothing if potential employer isn't hiring
            if not potential_employer.open_position:
                continue
            
            # If unemployed, do nothing if potential employer doesn't pay minimum wage
            if not self.employer and potential_employer.wage_rate < self.reservation_wage:
                continue
                
            # If employed instead, do nothing if potential employer doesn't offer better wage
            if self.employer and potential_employer.wage_rate <= self.employer.wage_rate:
                continue

            # Quit old jobs
            if self.employer:
                self.employer.let_go(self)

            # Accept new position
            potential_employer.hire(self)
            return
            
    def budget_consumption(self):
        average_goods_price = sum([supplier.goods_price for supplier in self.suppliers]) / len(self.suppliers)
        try:
            self.planned_expenditure = min(
                (self.liquidity / average_goods_price) ** self.parameters.alpha_budget_exponent,
                self.liquidity / average_goods_price
            )
        except:
            print(self.liquidity)
            print(average_goods_price)
            print((self.liquidity / average_goods_price) ** self.parameters.alpha_budget_exponent)
            print(self.liquidity / average_goods_price)
            raise

    def purchase_goods(self):
        planned_daily_demand = self.planned_expenditure / self.model.parameters.month_length
        # If planned daily_demand is 0, just return
        if not planned_daily_demand:
            return
        daily_demand = planned_daily_demand
        visited_suppliers = []
        while daily_demand / planned_daily_demand > 0.05 and len(visited_suppliers) < len(self.suppliers):
            chosen_supplier = self.random.choice([supplier for supplier in self.suppliers if supplier not in visited_suppliers])
            visited_suppliers.append(chosen_supplier)

            # If doesn't have enough liquidity, adjust demand (to effective demand)
            if self.liquidity < chosen_supplier.goods_price * daily_demand:
                daily_demand = self.liquidity / chosen_supplier.goods_price
            # If firm can satisfied
            if chosen_supplier.inventories >= daily_demand:
                transaction_amount = daily_demand
            # Else blacklist
            else:
                transaction_amount = chosen_supplier.inventories
                if chosen_supplier.unique_id not in self.constrained_suppliers:
                    self.constrained_suppliers[chosen_supplier.unique_id] = 0
                self.constrained_suppliers[chosen_supplier.unique_id] = self.constrained_suppliers[chosen_supplier.unique_id] + (daily_demand - chosen_supplier.inventories)

            # Register demand for the supplier first before purchasing and modifying daily_demand
            chosen_supplier.register_demand(daily_demand)

            # Transaction time
            transaction_cost = chosen_supplier.goods_price * transaction_amount
            if transaction_cost > self.liquidity:
                # Take care of rare rounding error
                transaction_cost = self.liquidity
            self.liquidity = self.liquidity - transaction_cost
            daily_demand = daily_demand - transaction_amount

            chosen_supplier.sell(transaction_amount, transaction_cost)

        # Add the reset daily demand to unsatisfied stats
        self.unsatisfied_demand = self.unsatisfied_demand + max(0, daily_demand)

    
    def adjust_reservation_wage(self):
        if not self.employer:
            self.reservation_wage = self.reservation_wage * 0.9
            return

        if self.employer.wage_rate > self.reservation_wage:
            self.reservation_wage = self.employer.wage_rate


    def update_monthly_stats(self):
        self.last_month_constrained_suppliers = self.constrained_suppliers
        self.constrained_suppliers = {}
        self.unsatisfied_demand = 0