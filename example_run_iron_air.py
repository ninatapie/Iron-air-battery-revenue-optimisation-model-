
import sizing_bidding_strategy_iron_air as sizing_bidding_strategy
import df_filter
import pandas as pd



### Inputs
    
# chg / dchg efficiency in percentage (multiply together to get total roundtrip)
storage_chg_eff = 0.73 # 69-73% (Form Energy, 2024)
storage_dchg_eff = 0.62 # 58-62% (Form Energy, 2024) 

# Transmission efficiency (cables and inverter DC/AC)
AC_DC_eff = 1.0 # set to 1.0 since Form Energy's figures already include this loss in charing efficiency
DC_AC_eff = 1.0 # set to 1.0 since Form Energy's figures already include this loss in charing efficiency
DC_DC_eff = 1.0 # not used in AC-coupled mode (coupling[0]=True) - irrelevant to battery-grid conversion; set to 1.0 for clarity
AC_AC_eff = 1.0 # not used in AC-coupled mode (coupling[0]=True) - irrelevant to battery-grid conversion; set to 1.0 for clarity

# Marginal Cost of chg / dchg in £ per MWh d/chg
MC_chg = 0
MC_dchg = 0
MC_solar = 0
MC_wind = 0

# Discharge duration and depth-of-discharge (DoD)
discharge_duration = 100
depth_of_discharge = 1.0 # iron-air batteries tolerate ~100% depth-of-discharge (Weinrich et al., 2019)

# Self-discharge of the storage system per day
self_discharge = 0.015 # midpoint to 1-2% per day (Narayanan et al., 2012) 

# Degradation rate -> Reduction of the DoD per MWh
Lt_cal = 20 # 15-20 years enclosure lifetime (Form Energy, 2024)
Lt_cycle = 2000 # Narayanan et al. (2012) and McKerracher et al. (2015)

# RES and grid connection capacities in MW 
storage_capacity = 100.0 # kept the same as it aligns with the size of announced Form Energy projects
solar_capacity = 0.0 # not relevant for my project
wind_capacity = 0.0 # not relevant for my project
grid_connection_capacity = 100.0 # kept the same - it matches storage capacity so grid connection is not a limiting factor (just the batteries performance)

# Upper bound of the storage system power capacity - not relevant for my project
capacity_upper_bound = 3*max(solar_capacity, wind_capacity)

# Interest rate and project duration for the annualisation
interest_rate = 0.116  # 11.6% for merchant and CM scenarios; 10.10% for cap and floor (CEPA, 2025) 
project_duration = 20  

# Daily cycle limit
daily_cycle_lim = 2  

# Adjustment for price time period. Default is time periods are set to 1/2 hour periods.
time_adjust = 0.5

# Storage Capital and Operational Costs in £/MW and £/MW/yr for each discharge duration
storage_capex_cost = {100: 1118100} # £/MW at 100h duration, lowerbound of Form Energy's (2024) $15-20/kWh, converted at $1=£0.7454 (July 2026 mid-market rate)
storage_opex_cost = {100: 11181} # £/MW/yr, lowerbound of Form Energy's (2024) $15-20/kW-yr, same conversion
storage_replacement_cost = {100: 0} # set to 0 since Lt_cal = project_duration (20 years) — no replacement needed, project ends at battery end-of-life. Revert this if Lt_cal and project_duration are ever set differently.

# Grid connection cost
grid_connection_cost = 0 # Form Energy's (2024) capex figure includes grid interconnection costs; set to 0 to avoid double counting

# Capital and operational cost of the RES in £/MW and in £/MW/yr
solar_opex_cost = 15400 # not relevant for my project
wind_opex_cost = 25400 # not relevant for my project

solar_capex_cost = 746440 # not relevant for my project
wind_capex_cost = 1342960 # not relevant for my project

# Capacity Market 
capacity_market_available = False

capacity_market_price = {'T4_low': 8.40, 'T4_medium': 27.10, 'T4_high': 65.00} # £/kW/yr, GB T-4 auction clearing prices (2021/22, 2029/30, 2027/28)

de_rating_factor = 0.9216 # NESO de-rating factor, 12h band (current methodology, 2029/30 auction) — applied consistently across all CM price scenarios to isolate price sensitivity specifically

capacity_market_scenario = 'T4_high'  # options: 'T4_low', 'T4_medium', 'T4_high' - to manually change depending on the scenario tested

CM_contract_length = 15  # years — GB Capacity Market maximum agreement length for high-capex storage (≥£350/kW de-rated), Modo Energy (2026)

CM_price = (capacity_market_price[capacity_market_scenario] * 1000) if capacity_market_available else 0 
CM_de_rating = de_rating_factor if capacity_market_available else 0

# Cap and Floor 
cap_and_floor_available = False 

CF_regime_duration = 25  # years, Ofgem default LDES cap-and-floor regime length (2025)

floor_return = 0.0447   # Ofgem (June, 2026)
cap_return = 0.0748     # Ofgem (June, 2026)
soft_cap_share = 0.30   # operator retains 30% of revenue above cap; 70% returned to consumers (Ofgem, 2025)

### Scenario Variables
# Adjust them depending on the scenario you want to test.

charge_from_grid = True # Possibility to charge from the grid on the WS or BM markets.
BM_available = False # Possibility to participate in BM services.
DC_available = False # Possibility to participate in Dynamic Containment services.

coupling = [True, False, False] # Type of coupling: AC_coupled, DC_coupled, DC_AC_coupled

optimise_RES = False # The RES capacity is also optimised

generation_type = ['solar'] # add 'solar' and/or 'wind'

RES_fixed_price = 0 # Fixed Price of the renewable PPA. If 0, the assets operate as Full Merchants

assets_coordinated = False # The assets coordinate their operations (Full Hybrid Project)

# Period of time considered in the optimisation. The model is developed to consider only one year at the moment. 
start_date = '01-01-2022' # dd-mm-yyyy 
end_date = '31-12-2022' # dd-mm-yyyy 

# Country studied
country = 'UK'

# Historical data - capacity factors
df_solar_data = df_filter.filter_production(start_date, end_date, 'Data/Production/' + country + '_solar_2022.csv')
df_wind_data = df_filter.filter_production(start_date, end_date, 'Data/Production/' + country + '_wind_2022.csv')

# Historical data - prices
df_master_data = df_filter.BM_filter_price(start_date,end_date, 'Data/Price/' + country + '_PRICE_2022_2025.csv', 'UK')[:-1]
df_DC_data = df_filter.DC_filter_price(start_date, end_date, 'Data/Price/DC_price_2021_2023.csv')


# Example of a code to find the optimal battery dispatch

output = []
output = sizing_bidding_strategy.Hybrid_solver(
        storage_capacity,
        df_solar_data,
        df_wind_data,
        df_master_data, 
        df_DC_data,
        start_date, 
        end_date, 
        storage_chg_eff, 
        storage_dchg_eff,
        coupling,
        DC_DC_eff,
        DC_AC_eff,
        AC_DC_eff,
        AC_AC_eff,
        MC_chg,
        MC_dchg,
        MC_solar,
        MC_wind, 
        grid_connection_capacity, 
        discharge_duration,
        depth_of_discharge,
        self_discharge,
        Lt_cal,
        Lt_cycle,
        solar_capacity,
        wind_capacity,
        capacity_upper_bound, 
        time_adjust,
        daily_cycle_lim,
        interest_rate,
        project_duration,
        storage_capex_cost[discharge_duration],
        storage_opex_cost[discharge_duration],
        storage_replacement_cost[discharge_duration],
        solar_capex_cost,
        solar_opex_cost,
        wind_capex_cost,
        wind_opex_cost,
        grid_connection_cost,
        'test_details',
        generation_type,
        charge_from_grid,
        BM_available,
        DC_available,
        optimise_RES,
        RES_fixed_price,
        assets_coordinated,
        CM_price,
        CM_de_rating,
        CM_contract_length,
        capacity_market_available,
        cap_and_floor_available,
        floor_return,
        cap_return,
        soft_cap_share,
        CF_regime_duration)

results_filename = f"test_summary_{discharge_duration}.csv"
df = pd.DataFrame(output.items(), columns=["variable", "value"])
df.to_csv(results_filename, index=False)

print("Saved")
