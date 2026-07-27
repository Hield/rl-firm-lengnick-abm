from dataclasses import dataclass


@dataclass
class FirmParameters:
    '''Parameters for the firms'''
    gamma_consecutive_months: int
    delta_wage_adjustment_rate: float
    upper_phi_demand: float
    lower_phi_demand: float
    upper_phi_marginal_cost: float
    lower_phi_marginal_cost : float
    theta_goods_price: float
    theta_probs_set_new_price: float
    lambda_technology: float
    chi_buffer : float

    # Initial values
    initial_open_position: bool
    initial_liquidity: float
    initial_goods_price: float
    initial_wage_rate: float

@dataclass
class HouseholdParameters:
    '''Parameters for the households'''
    xi_price_percentage: float
    beta_firm_count: int
    pi_explore_odd: float
    alpha_budget_exponent: float
    n_suppliers: int
    psi_price: float
    psi_quant: float

    # Initial values
    initial_reservation_wage: float
    initial_liquidity: float
    
@dataclass
class ModelParameters:
    '''Parameters for the baseline model'''
    month_length: int
    initial_employment_rate: float
    n_firms: int
    n_learning_firms: int
    n_households: int
    burn_in_months: int
    
    firm: FirmParameters
    household: HouseholdParameters