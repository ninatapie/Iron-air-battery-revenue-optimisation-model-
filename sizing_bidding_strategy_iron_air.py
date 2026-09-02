### This file contains the bidding optimisation algorithm. 

#import json
import pandas as pd
import os
import time
import numpy as np
import numpy_financial as npf
from collections import defaultdict # used for the daily cycle total and constraint

""" 1. Import the solver """
from ortools.linear_solver import pywraplp

def Hybrid_solver (
                 power_capacity,
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
                 storage_capex_cost,
                 storage_opex_cost,
                 storage_replacement_cost,
                 solar_capex_cost,
                 solar_opex_cost,
                 wind_capex_cost,
                 wind_opex_cost,
                 grid_connection_cost,
                 result_file_name,
                 generation_type,
                 charge_grid,
                 charge_BM,
                 charge_DC,
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
                 CF_regime_duration):
    
    if coupling[1] and (charge_BM or charge_DC):
        print("DC coupling can't participate in ancillary services")
        return 

    start_time = time.perf_counter()
    
    # Convert dataframes into list of dicts
    list_master_data = df_master_data.to_dict(orient= 'records')
    list_DC_data = df_DC_data.to_dict(orient='records')

    """ 2. Create the mip solver with the GLOP backend."""
    solver = pywraplp.Solver.CreateSolver('GLOP')
    solver.SetSolverSpecificParametersAsString('solution_feasibility_tolerance: 1e-4')
    
    """ 3. Define the constants and input data """

    # Create a list of the wholesale electricity prices
    WS_price = [item['WS_price'] for item in list_master_data]
    if RES_fixed_price != 0:
        RES_price = [RES_fixed_price for i in range(len(list_master_data))]
    else:
        RES_price = WS_price
    
    # Create a list of the BM volumes and prices
    BM_price = [item['BM_price'] for item in list_master_data] if charge_BM and all('BM_price' in item for item in list_master_data) else [0 for i in range(len(list_master_data))] 
    
    if charge_BM:
        if all('BM_price' in item for item in list_master_data):
            BM_imbal_vol = [item['BM_imbal_vol'] for item in list_master_data]
        else:
            BM_imbal_vol = [1000000 for i in range(len(df_master_data))] 
    else:
        BM_imbal_vol = [0 for i in range(len(list_master_data))]

    # Create a list of the DC volumes and prices
    DC_low_price = [item['DCL Clearing Price'] for item in list_DC_data] if charge_DC else [0 for i in range(len(list_master_data))]
    DC_low_vol = [item['DCL Cleared Volume'] for item in list_DC_data] if charge_DC else [0 for i in range(len(list_master_data))]
    DC_high_price = [item['DCH Clearing Price'] for item in list_DC_data] if charge_DC else [0 for i in range(len(list_master_data))]
    DC_high_vol = [item['DCH Cleared Volume'] for item in list_DC_data] if charge_DC else [0 for i in range(len(list_master_data))]


    # Generates the part of the energy reserved for DC that will be activated
    epsilon_low = {}
    epsilon_high = {}
    for i in range(len(list_master_data)):
        epsilon_low[i] = 0.1
        epsilon_high[i] = 0.1

    """ 4. Define the decision variables """

    # Initialise the power and energy capacity of the storage system and the RES systems
    energy_capacity = power_capacity * discharge_duration

    # Initialise the renewable capacities depending on the type of renewable used in generation_type and the scenario variable optimise_RES
    if 'solar' in generation_type:
        if optimise_RES:
            solar_capacity = solver.NumVar(0,capacity_upper_bound, 'solar_capacity')
        if not('wind' in generation_type):
            wind_capacity = 0
        
    if 'wind' in generation_type:
        if optimise_RES:
            wind_capacity = solver.NumVar(0,capacity_upper_bound, 'wind_capacity')
        if not('solar' in generation_type):
            solar_capacity = 0
        
    if len(generation_type) == 0:
        solar_capacity = 0
        wind_capacity = 0

    # Definition of the transmission efficiencies between the different units
    # Defers as a function of the coupling type (AC, DC, DC-AC)
    eff_solar_to_grid = 0
    eff_solar_to_storage = 0
    eff_wind_to_grid = 0
    eff_wind_to_storage =0
    if coupling[0]:
        if 'solar' in generation_type:
            eff_solar_to_grid = DC_AC_eff
            eff_solar_to_storage = DC_AC_eff * AC_DC_eff
        if 'wind' in generation_type:
            eff_wind_to_grid = AC_AC_eff
            eff_wind_to_storage = AC_AC_eff * AC_DC_eff
        eff_grid_to_storage = AC_DC_eff
        eff_storage_to_grid = DC_AC_eff
    elif (coupling[1] or coupling[2]):
        if 'solar' in generation_type:
            eff_solar_to_grid = DC_DC_eff * DC_AC_eff
            eff_solar_to_storage = DC_DC_eff * DC_DC_eff
        if 'wind' in generation_type:
            eff_wind_to_grid = AC_DC_eff * DC_AC_eff
            eff_wind_to_storage = AC_DC_eff * DC_DC_eff
        eff_grid_to_storage = AC_DC_eff * DC_DC_eff
        eff_storage_to_grid = DC_DC_eff * DC_AC_eff

    # Initialise energy level at start of each period - this is a decision variable even though it is not directly changed by the solver
    E = {}
    for i in range(len(list_master_data)):
        E[i] = solver.NumVar(0, energy_capacity, 'E[%i]' % i)
    
    # Define the gross variables (WS charge/discharge, BM charge/discharge, DC low/high volume cleared)
    WS_dchg = {}
    BM_dchg = {}
    WS_chg = {}
    BM_chg = {}
    DC_high_cleared = {}
    DC_low_cleared = {}

    for i in range(len(list_master_data)):
        WS_dchg[i] = solver.NumVar(0, energy_capacity, 'WS_dchg[%i]' % i) # MWh
        BM_dchg[i] = solver.NumVar(0, energy_capacity, 'BM_dchg[%i]' % i) if charge_BM else solver.NumVar(0, 0, 'BM_dchg[%i]' % i)
        WS_chg[i] = solver.NumVar(0, energy_capacity, 'WS_chg[%i]' % i) if charge_grid else solver.NumVar(0, 0, 'WS_chg[%i]' % i)
        BM_chg[i] = solver.NumVar(0, energy_capacity, 'BM_chg[%i]' % i) if (charge_BM and charge_grid) else solver.NumVar(0, 0, 'BM_chg[%i]' % i)
        DC_high_cleared[i] = solver.NumVar(0, energy_capacity, 'DC_high_cleared[%i]' % i) if (charge_DC and charge_grid) else solver.NumVar(0, 0, 'DC_high_cleared[%i]' % i)
        DC_low_cleared[i] = solver.NumVar(0, energy_capacity, 'DC_low_cleared[%i]' % i) if charge_DC else solver.NumVar(0, 0, 'DC_low_cleared[%i]' % i)
        
    # Define the gross amount of energy charging to the storage system and going to the grid from the RES systems
    # Values set to 0 if the generation type is not considered 
    solar_chg = {}
    solar_to_grid = {}
    wind_chg = {}
    wind_to_grid = {}

    for i in range(len(list_master_data)):
        solar_chg[i] = solver.NumVar(0, energy_capacity, 'solar_chg[%i]' % i) if 'solar' in generation_type else solver.NumVar(0, 0, 'solar_chg[%i]' % i)
        wind_chg[i] = solver.NumVar(0, energy_capacity, 'wind_chg[%i]' % i) if 'wind' in generation_type else solver.NumVar(0, 0, 'wind_chg[%i]' % i)

        if assets_coordinated:
            solar_to_grid[i] = solver.NumVar(0, capacity_upper_bound/2, 'solar_to_grid[%i]' % i) if 'solar' in generation_type else solver.NumVar(0, 0, 'solar_to_grid[%i]' % i)
            wind_to_grid[i] = solver.NumVar(0, capacity_upper_bound/2, 'wind_to_grid[%i]' % i) if 'wind' in generation_type else solver.NumVar(0, 0, 'wind_to_grid[%i]' % i)
        
        else:
            solar_to_grid[i] = (time_adjust * min(grid_connection_capacity/eff_solar_to_grid, solar_capacity * df_solar_data['electricity'][i]) if RES_price[i] > 0 else 0) if 'solar' in generation_type else 0
            wind_to_grid[i] = (time_adjust * min(grid_connection_capacity/eff_wind_to_grid - (eff_solar_to_grid/eff_wind_to_grid) * solar_to_grid[i], wind_capacity * df_wind_data['electricity'][i]) if RES_price[i] > 0 else 0) if 'wind' in generation_type else 0

    """ 6. Define the constraints """

    # DC constraints: Takes into account the volume available from the system operator
    if charge_DC:
        for i in range(0, len(list_master_data), 8):
            # Use the first index of the block as the base index
            base_index = i
            
            # Ensure consistency within each 4-hour block
            for j in range(1, 8):
                if i + j < len(list_master_data):
                    solver.Add(DC_low_cleared[i + j] == DC_low_cleared[base_index])
                    solver.Add(DC_high_cleared[i + j] == DC_high_cleared[base_index])


        for i in range(len(list_master_data)):
            if DC_low_vol[i] > 0 :
                solver.Add(DC_low_cleared[i] <= DC_low_vol[i]) 
            elif DC_high_vol[i] > 0 :
                solver.Add(DC_high_cleared[i] <= DC_high_vol[i])
            else:
                solver.Add(DC_high_cleared[i] == 0)
                solver.Add(DC_low_cleared[i] == 0)
    
    ## Balancing mechanisms constraints: Takes into account the volume available from the system operator
    if charge_BM:
        for i in range(len(list_master_data)):
            if BM_imbal_vol[i] < 0 :
                # If BM_imbal_vol (NIV) is negative, system is long, ESO is selling energy. Only chg available (up to NIV), dchg not available
                solver.Add(BM_chg[i] <= - BM_imbal_vol[i]) # NOTE: BM_imbal_vol is negative, BM_chg is always positive. To compare must have - in front of BM_imbal_vol
                solver.Add(BM_dchg[i] == 0)
            elif BM_imbal_vol[i] > 0 :
                # If BM_imbal_vol (NIV) is positive, system is short, ESO is buying energy. Only dchg is available (up to NIV), chg not available.
                solver.Add(BM_chg[i] == 0)
                solver.Add(BM_dchg[i]/storage_dchg_eff/eff_storage_to_grid <= BM_imbal_vol[i])
            elif BM_imbal_vol[i] == 0: # If BM_imbal_vol (NIV) is zero, no BM activity required, d/chg must both be zero.
                solver.Add(BM_chg[i] == 0)
                solver.Add(BM_dchg[i] == 0)
            else:
                print ('BM_imbal_vol data missing or not in int format')
                print ('Period index: ', i)
  
    ## Power constraints

    # Power available from the solar panels
    if 'solar' in generation_type:
        for i in range(len(list_master_data)):
            solver.Add( (solar_chg[i] + solar_to_grid[i] <= solar_capacity * df_solar_data['electricity'][i] * time_adjust) )

    # Power available from the wind turbines
    if 'wind' in generation_type:
        for i in range(len(list_master_data)):
            solver.Add( (wind_chg[i] + wind_to_grid[i] <= wind_capacity * df_wind_data['electricity'][i] * time_adjust) )

    # The total energy received or transferred to the grid in each period is limited by grid capacity - this restricts X netting between WS, BM and DC.
    for i in range(len(list_master_data)):
        solver.Add( (solar_to_grid[i] * eff_solar_to_grid + (WS_dchg[i] + BM_dchg[i]) * storage_dchg_eff * eff_storage_to_grid + wind_to_grid[i] * eff_wind_to_grid + WS_chg[i] + BM_chg[i] + DC_low_cleared[i] + DC_high_cleared[i] <= grid_connection_capacity * time_adjust) )

    # The total energy discharged or charged in each period is limited by total storage power capacity - this restricts X netting between WS, BM and DC.
    for i in range(len(list_master_data)):
        solver.Add( ((solar_chg[i] * eff_solar_to_storage + wind_chg[i] * eff_wind_to_storage) * storage_chg_eff + WS_dchg[i] + BM_dchg[i] + DC_low_cleared[i]/storage_dchg_eff/eff_storage_to_grid + (WS_chg[i] + BM_chg[i] + DC_high_cleared[i]) * eff_grid_to_storage * storage_chg_eff) <= (power_capacity * time_adjust))
    
    ## Energy Constraints
    
    # The total WS, BM and DC quantity discharged in each period is limited by available energy stored (at start of that period)
    for i in range(len(list_master_data)):
        solver.Add(WS_dchg[i] + BM_dchg[i] + DC_low_cleared[i]/storage_dchg_eff/eff_storage_to_grid <= (1 - self_discharge/24*time_adjust) * E[i])

    # Battery degradation (Reduction of its state of health)
    if energy_capacity != 0:
        aging_cal = [0 for i in range (len(list_master_data))]
        aging_cycle = {}
        for i in range(len(list_master_data)):
            aging_cycle[i] = solver.NumVar(0, 1, 'aging_cycle[%i]' % i)
        
        solver.Add(aging_cycle[0] == 0)
        
        for i in range(1, len(list_master_data)):
            aging_cal[i] =  i * time_adjust/(Lt_cal*8760)
            solver.Add(aging_cycle[i] == aging_cycle[i-1] + 0.5 * (solar_chg[i-1] * eff_solar_to_storage + wind_chg[i-1] * eff_wind_to_storage + (WS_chg[i-1] + BM_chg[i-1] + epsilon_high[i-1] * DC_high_cleared[i-1]) * eff_grid_to_storage * storage_chg_eff + WS_dchg[i-1] + BM_dchg[i-1] + epsilon_low[i-1] * DC_low_cleared[i-1]/storage_dchg_eff/eff_storage_to_grid)/(Lt_cycle*energy_capacity))


        state_of_health = {}
        for i in range(len(list_master_data)):
            state_of_health[i] = solver.NumVar(0, 1, 'state_of_health[%i]' % i)

        solver.Add(state_of_health[0] == 1)

        for i in range(1, len(list_master_data)):
            solver.Add(state_of_health[i] == 1 - 0.2 * (aging_cal[i] + aging_cycle[i]))

        # Assuming the battery starts empty
        solver.Add(E[0] == 0)
        
        # The energy level at each period is limited by the storage capacity, the depth-of-discharge and the state-of-health
        for i in range(len(list_master_data)):
            solver.Add(E[i] <= energy_capacity * state_of_health[i] * depth_of_discharge)
    
    # Daily cycle constraints
    # Define dictionary to store the decision variables for each date
    daily_d_chg = defaultdict(list) # Creates an empty dict of special type (defaultdict), that creates a list for whatever key you 
    # Populate the dictionaries with the decision variables for each date
    for i, item in enumerate(df_master_data['datetime']):
        item = pd.to_datetime(item, format='%d/%m/%Y %H:%M:%S')
        date = item.date()
        daily_d_chg[date].append( (solar_chg[i] * eff_solar_to_storage + wind_chg[i] * eff_wind_to_storage) * storage_chg_eff  + WS_dchg[i] + BM_dchg[i] + (WS_chg[i] + BM_chg[i] + DC_high_cleared[i]) * eff_grid_to_storage * storage_chg_eff + DC_low_cleared[i]/eff_storage_to_grid/storage_dchg_eff)
        
    
    # Convert number of daily cycles to daily_max_d_chg (which is total MWh d/chg'd). This equals total d/chg'ing / (energy capacity / 2)
    daily_max_d_chg =  daily_cycle_lim  * (energy_capacity * 2)
    
    for date in daily_d_chg:
        solver.Add(solver.Sum(daily_d_chg[date]) <= daily_max_d_chg)  # Daily charge limit
    

    # The energy level at start of each period is the previous energy level 
    # plus the total RES amount + WS, BM and DC charged minus the total WS and BM discharged
    # It also considers a self-discharge 
    for i in range(1, len(list_master_data)):  # start from 1 because E[0] = chg[0]
        solver.Add(E[i] == (1 - self_discharge/24*time_adjust) * E[i-1] + (solar_chg[i-1] * eff_solar_to_storage + wind_chg[i-1] * eff_wind_to_storage) * storage_chg_eff + (WS_chg[i-1] + BM_chg[i-1] + epsilon_high[i-1] * DC_high_cleared[i-1]) * eff_grid_to_storage * storage_chg_eff - WS_dchg[i-1] - BM_dchg[i-1] - epsilon_low[i-1] * DC_low_cleared[i-1]/storage_dchg_eff/eff_storage_to_grid)

    """ Define the objective function """
    
    objective_terms = []
    annuity_factor = (((1 + interest_rate)**project_duration) * interest_rate)/((1 + interest_rate)**project_duration - 1)

    for i in range(len(list_master_data)):
        # WS, BM and DC discharge revenue net of MC and efficiency + RES production to the grid revenue 
        # minus WS, BM, DC charge including MC and efficiency losses
        
        if assets_coordinated:
            term = (WS_dchg[i] * WS_price[i] + BM_dchg[i] * BM_price[i]) * eff_storage_to_grid * storage_dchg_eff - (WS_dchg[i] * MC_dchg) - (BM_dchg[i] * MC_dchg) + (solar_to_grid[i] * eff_solar_to_grid + wind_to_grid[i] * eff_wind_to_grid) * RES_price[i] - ((solar_to_grid[i] + solar_chg[i]) * MC_solar) - ((wind_to_grid[i] + wind_chg[i]) * MC_wind) - (solar_chg[i] * storage_chg_eff * eff_solar_to_storage * MC_chg) - (wind_chg[i] * storage_chg_eff * eff_wind_to_storage * MC_chg) - (WS_chg[i] * WS_price[i]) - (WS_chg[i] * storage_chg_eff * eff_grid_to_storage * MC_chg) - (BM_chg[i] * BM_price[i]) - (BM_chg[i] * storage_chg_eff * eff_grid_to_storage * MC_chg) + (DC_high_cleared[i] * DC_high_price[i] + DC_low_cleared[i] * DC_low_price[i]) - (epsilon_high[i] * DC_high_cleared[i]*storage_chg_eff*eff_grid_to_storage * MC_chg) - (epsilon_low[i] * DC_low_cleared[i]/storage_dchg_eff/eff_storage_to_grid * MC_dchg) 
        else:
            term = (WS_dchg[i] * WS_price[i] + BM_dchg[i] * BM_price[i]) * eff_storage_to_grid * storage_dchg_eff - (WS_dchg[i] * MC_dchg) - (BM_dchg[i] * MC_dchg) - (solar_chg[i] * storage_chg_eff * eff_solar_to_storage * MC_chg) - (wind_chg[i] * storage_chg_eff * eff_wind_to_storage * MC_chg) - (WS_chg[i] * WS_price[i]) - (WS_chg[i] * storage_chg_eff * eff_grid_to_storage * MC_chg) - (BM_chg[i] * BM_price[i]) - (BM_chg[i] * storage_chg_eff * eff_grid_to_storage * MC_chg) + (DC_high_cleared[i] * DC_high_price[i] + DC_low_cleared[i] * DC_low_price[i]) - (epsilon_high[i] * DC_high_cleared[i]*storage_chg_eff*eff_grid_to_storage * MC_chg) - (epsilon_low[i] * DC_low_cleared[i]/storage_dchg_eff/eff_storage_to_grid * MC_dchg) #- max(0,min(grid_connection_capacity * RES_price[i] * time_adjust, (eff_solar_to_grid * solar_capacity * df_solar_data['electricity'][i] + eff_wind_to_grid * wind_capacity * df_wind_data['electricity'][i]) * RES_price[i]*time_adjust))

        objective_terms.append(term)
    
    # Objective function is the annualised NPV (annual profit - annualised investment cost). Storage costs are not considered as the storage capacity is fixed within the bidding strategy.
    if assets_coordinated:
        solver.Maximize(solver.Sum(objective_terms) - (solar_capex_cost * annuity_factor + solar_opex_cost) * solar_capacity - (wind_capex_cost * annuity_factor + wind_opex_cost) * wind_capacity - grid_connection_cost * grid_connection_capacity * annuity_factor)
    else:
        if optimise_RES:
            solver.Maximize(solver.Sum(objective_terms) - power_capacity/(power_capacity + solar_capacity + wind_capacity) * grid_connection_capacity * grid_connection_cost * annuity_factor)
        elif (solar_capacity + wind_capacity) > 0:
            solver.Maximize(solver.Sum(objective_terms) - power_capacity/(power_capacity + solar_capacity + wind_capacity) * grid_connection_capacity * grid_connection_cost * annuity_factor)
        else:
            solver.Maximize(solver.Sum(objective_terms))

    # Solve the problem
    status = solver.Solve()
    
    """ Save output to .csv file and save """
    
    if status == pywraplp.Solver.OPTIMAL:
        
        print ('Optimal Bidding Solution found')

        # Creating lists to store the solution values (gross and nt
        energy_level = []

        charged_WS = []
        charged_BM = []
        discharged_WS = []
        discharged_BM = []
        solar_power_to_storage = []
        solar_power_to_grid = []
        net_solar_power_to_grid = []
        wind_power_to_storage = []
        wind_power_to_grid = []
        net_wind_power_to_grid = []
        net_charged_WS = []
        net_charged_BM = []
        net_discharged_WS = []
        net_discharged_BM = []
        storage_dchg_WS_revenues = []
        storage_dchg_BM_revenues = []
        storage_chg_WS_costs = []
        storage_chg_BM_costs = []
        solar_to_grid_revenues = []
        wind_to_grid_revenues = []
        objective_values = []
        solar_power_list = []
        wind_power_list = []
        optimised_DC_high_cleared = []
        optimised_DC_low_cleared = []
        optimised_DC_high_used = []
        optimised_DC_low_used = []
        storage_DC_high_revenues = []
        storage_DC_low_revenues = []
        storage_RES_costs = []
        RES_to_grid_revenues = []
        RES_to_storage = []
    
        for i in range(len(list_master_data)):
            # Append solution values to the lists
            objective_values.append(objective_terms[i].solution_value())
            energy_level.append(E[i].solution_value())
            discharged_WS.append(WS_dchg[i].solution_value())
            solar_power_to_storage.append(solar_chg[i].solution_value())
            solar_power_to_grid.append((solar_to_grid[i].solution_value() if assets_coordinated else solar_to_grid[i]) if 'solar' in generation_type else 0)
            net_solar_power_to_grid.append((solar_to_grid[i].solution_value() * eff_solar_to_grid if assets_coordinated else solar_to_grid[i] * eff_solar_to_grid) if 'solar' in generation_type else 0)
            wind_power_to_storage.append(wind_chg[i].solution_value())
            wind_power_to_grid.append((wind_to_grid[i].solution_value() if assets_coordinated else wind_to_grid[i]) if 'wind' in generation_type else 0)
            net_wind_power_to_grid.append((wind_to_grid[i].solution_value() * eff_wind_to_grid if assets_coordinated else wind_to_grid[i] * eff_wind_to_grid) if 'wind' in generation_type else 0)
            net_discharged_WS.append(WS_dchg[i].solution_value() * eff_storage_to_grid * storage_dchg_eff)
            storage_dchg_WS_revenues.append((WS_dchg[i] * WS_price[i]).solution_value() * eff_storage_to_grid * storage_dchg_eff)
            storage_chg_WS_costs.append((WS_chg[i] * WS_price[i]).solution_value())
            solar_to_grid_revenues.append(((solar_to_grid[i] * RES_price[i]).solution_value() * eff_solar_to_grid if assets_coordinated else solar_to_grid[i] * RES_price[i] * eff_solar_to_grid) if 'solar' in generation_type else 0)
            wind_to_grid_revenues.append(((wind_to_grid[i] * RES_price[i]).solution_value() * eff_wind_to_grid if assets_coordinated else wind_to_grid[i] * RES_price[i] * eff_wind_to_grid) if 'wind' in generation_type else 0)
            RES_to_grid_revenues.append((solar_to_grid[i] * eff_solar_to_grid + wind_to_grid[i] * eff_wind_to_grid).solution_value() * RES_price[i] if assets_coordinated else (solar_to_grid[i] * eff_solar_to_grid + wind_to_grid[i] * eff_wind_to_grid) * RES_price[i])
            if optimise_RES:
                solar_power_list.append(solar_capacity.solution_value()*df_solar_data['electricity'][i] if 'solar' in generation_type else 0)
                wind_power_list.append(wind_capacity.solution_value()*df_wind_data['electricity'][i] if 'wind' in generation_type else 0)
            else:
                solar_power_list.append(solar_capacity*df_solar_data['electricity'][i])
                wind_power_list.append(wind_capacity*df_wind_data['electricity'][i])

            discharged_BM.append(BM_dchg[i].solution_value())
            net_discharged_BM.append(BM_dchg[i].solution_value() * eff_storage_to_grid * storage_dchg_eff)
            storage_dchg_BM_revenues.append((BM_dchg[i] * BM_price[i]).solution_value() * eff_storage_to_grid * storage_dchg_eff)
            storage_chg_BM_costs.append((BM_chg[i] * BM_price[i]).solution_value())
            charged_BM.append(BM_chg[i].solution_value())
            net_charged_BM.append(BM_chg[i].solution_value() * eff_grid_to_storage * storage_chg_eff)
            charged_WS.append(WS_chg[i].solution_value())
            net_charged_WS.append(WS_chg[i].solution_value() * eff_grid_to_storage * storage_chg_eff)
            optimised_DC_high_cleared.append(DC_high_cleared[i].solution_value())
            optimised_DC_low_cleared.append(DC_low_cleared[i].solution_value())
            optimised_DC_high_used.append(epsilon_high[i] * DC_high_cleared[i].solution_value())
            optimised_DC_low_used.append(epsilon_low[i] * DC_low_cleared[i].solution_value())
            storage_DC_high_revenues.append(DC_high_cleared[i].solution_value() * DC_high_price[i])
            storage_DC_low_revenues.append(DC_low_cleared[i].solution_value() * DC_low_price[i])
            storage_RES_costs.append((solar_chg[i] + wind_chg[i]).solution_value() * RES_price[i])
            

        # Assign lists to new columns in the dataframe
        df_master_data['Objective Value'] = objective_values
        df_master_data['Energy Level'] = energy_level
        
        df_master_data['Gross Amount Charged WS'] = charged_WS
        df_master_data['Gross Amount Charged BM'] = charged_BM

        df_master_data['Gross Amount Discharged WS'] = discharged_WS
        df_master_data['Gross Amount Discharged BM'] = discharged_BM

        df_master_data['Solar energy stored'] = solar_power_to_storage
        df_master_data['Net Solar energy sold to grid'] = net_solar_power_to_grid
        df_master_data['Wind energy stored'] = wind_power_to_storage
        df_master_data['Net Wind energy sold to grid'] = net_wind_power_to_grid

        df_master_data['Net Amount Charged WS'] = net_charged_WS
        df_master_data['Net Amount Charged BM'] = net_charged_BM  
        
        df_master_data['Net Amount Discharged WS'] = net_discharged_WS
        df_master_data['Net Amount Discharged BM'] = net_discharged_BM  

        df_master_data['Revenues from storage system WS'] = storage_dchg_WS_revenues
        df_master_data['Revenues from storage system BM'] = storage_dchg_BM_revenues
        df_master_data['Revenues from Solar'] = solar_to_grid_revenues  
        df_master_data['Revenues from Wind'] = wind_to_grid_revenues 

        df_master_data['Solar Power'] = solar_power_list
        df_master_data['Wind Power'] = wind_power_list

        df_master_data['DC high amount cleared'] = optimised_DC_high_cleared
        df_master_data['DC low amount cleared'] = optimised_DC_low_cleared
        df_master_data['Real DC high amount used'] = optimised_DC_high_used 
        df_master_data['Real DC low amount used'] = optimised_DC_low_used 

        # Assess power capacity
        if optimise_RES:
            optimised_solar_capacity = solar_capacity.solution_value() if 'solar' in generation_type else 0
            optimised_wind_capacity = wind_capacity.solution_value() if 'wind' in generation_type else 0
        else:
            optimised_solar_capacity = solar_capacity
            optimised_wind_capacity = wind_capacity

        # Assess total profit
        total_profit = sum(objective_values)

        # Assess different costs and revenues
        WS_revenues = sum(storage_dchg_WS_revenues)
        WS_chg_tot = sum(charged_WS)

        WS_costs = sum(storage_chg_WS_costs)
        WS_dchg_tot = sum(discharged_WS)

        BM_revenues = sum(storage_dchg_BM_revenues)
        BM_chg_tot = sum(charged_BM)

        BM_costs = sum(storage_chg_BM_costs)
        BM_dchg_tot = sum(discharged_BM)

        DC_high_revenues = sum(storage_DC_high_revenues)
        DC_chg_tot = sum(optimised_DC_high_cleared)

        DC_low_revenues = sum(storage_DC_low_revenues)
        DC_dchg_tot = sum(optimised_DC_low_cleared)

        RES_to_storage_cost = sum(storage_RES_costs)

        RES_revenues = sum(solar_to_grid_revenues) + sum(wind_to_grid_revenues)

        RES_to_grid_revenues_tot = sum(RES_to_grid_revenues)

        RES_to_storage = sum(solar_power_to_storage) + sum(wind_power_to_storage)

        # Capacity Market annual revenue
        CM_revenue = CM_price * CM_de_rating * power_capacity
        def CM_revenue_at_year(t):
            return CM_revenue if t <= CM_contract_length else 0
        
        # Cap and Floor - floor/cap thresholds (£/year)
        floor_annual_revenue = (storage_capex_cost * power_capacity * floor_return) + (storage_opex_cost * power_capacity)
        cap_annual_revenue = (storage_capex_cost * power_capacity * cap_return) + (storage_opex_cost * power_capacity)
        def apply_cap_and_floor(revenue, t):
            if t > CF_regime_duration:
                return revenue
            if revenue < floor_annual_revenue:
                return floor_annual_revenue
            elif revenue > cap_annual_revenue: 
                excess = revenue - cap_annual_revenue
                return cap_annual_revenue + (soft_cap_share * excess)
            else:
                return revenue
        
        # Combined revenue function 
        def total_revenue_at_year(t):
            revenue = total_profit
            if capacity_market_available:
                revenue += CM_revenue_at_year(t)
            if cap_and_floor_available:
                revenue = apply_cap_and_floor(revenue, t)
            return revenue
        
        # Diagnostic: how much did cap-and-floor actually adjust revenue by, in year 1?
        CF_adjustment = total_revenue_at_year(1) - (total_profit + CM_revenue_at_year(1))
       
        # Assess the Net Present Value
        if assets_coordinated:
            NPV = sum((total_revenue_at_year(t) - storage_opex_cost * power_capacity - solar_opex_cost * optimised_solar_capacity - wind_opex_cost * optimised_wind_capacity) / (1 + interest_rate)**t for t in range(1, project_duration + 1)) - storage_capex_cost * power_capacity - solar_capex_cost * optimised_solar_capacity - wind_capex_cost * optimised_wind_capacity - storage_replacement_cost * power_capacity * 1/(1 + interest_rate)**(Lt_cal) - grid_connection_cost * grid_connection_capacity
            global_NPV = NPV
        else:
            NPV = sum((total_revenue_at_year(t) - storage_opex_cost * power_capacity) / (1 + interest_rate)**t for t in range(1, project_duration + 1)) - storage_capex_cost * power_capacity - storage_replacement_cost * power_capacity * 1/(1 + interest_rate)**(Lt_cal) - power_capacity/(power_capacity + solar_capacity + wind_capacity) * grid_connection_capacity * grid_connection_cost
            global_NPV = sum((total_revenue_at_year(t) + RES_to_grid_revenues_tot - storage_opex_cost * power_capacity - solar_opex_cost * optimised_solar_capacity - wind_opex_cost * optimised_wind_capacity) / (1 + interest_rate)**t for t in range(1, project_duration + 1)) - storage_capex_cost * power_capacity - solar_capex_cost * optimised_solar_capacity - wind_capex_cost * optimised_wind_capacity - storage_replacement_cost * power_capacity * 1/(1 + interest_rate)**(Lt_cal) - grid_connection_cost * grid_connection_capacity
        # Assess the Internal Rate of Return (IRR)
        if assets_coordinated:
            cash_flows = [-storage_capex_cost * power_capacity - solar_capex_cost * optimised_solar_capacity - wind_capex_cost * optimised_wind_capacity - grid_connection_cost * grid_connection_capacity]
            for t in range(1, int(project_duration) + 1):
                annual_cf = total_revenue_at_year(t)- storage_opex_cost * power_capacity - solar_opex_cost * optimised_solar_capacity - wind_opex_cost * optimised_wind_capacity
                if t == int(Lt_cal):
                    annual_cf -= storage_replacement_cost * power_capacity
                cash_flows.append(annual_cf)
        else:
            cash_flows = [-storage_capex_cost * power_capacity - power_capacity/(power_capacity + solar_capacity + wind_capacity) * grid_connection_capacity * grid_connection_cost]
            for t in range(1, int(project_duration) + 1):
                annual_cf = total_revenue_at_year(t) - storage_opex_cost * power_capacity
                if t == int(Lt_cal):
                    annual_cf -= storage_replacement_cost * power_capacity
                cash_flows.append(annual_cf)

        irr = npf.irr(cash_flows)

        # Assess the Payback Period
        year = 0 
        year_limit = 40
        if assets_coordinated:
            balance = -storage_capex_cost * power_capacity - solar_capex_cost * optimised_solar_capacity - wind_capex_cost * optimised_wind_capacity - grid_connection_cost * grid_connection_capacity
        else:
            balance = -storage_capex_cost * power_capacity - power_capacity/(power_capacity + solar_capacity + wind_capacity) * grid_connection_capacity * grid_connection_cost
        while balance < 0 and year < year_limit:
            if assets_coordinated:
                balance += (total_revenue_at_year(year) - storage_opex_cost * power_capacity - solar_opex_cost * optimised_solar_capacity - wind_opex_cost * optimised_wind_capacity)/(1 + interest_rate)**year
            else:
                balance += (total_revenue_at_year(year) - storage_opex_cost * power_capacity)/(1 + interest_rate)**year
            year += 1
        
        discounted_payback_period = year
        
        # Calculate the initial and the new capture rates for the year considered
        average_price = np.average(WS_price)

        if len(generation_type) != 0:
            if not(optimise_RES) and (optimised_solar_capacity + optimised_wind_capacity > 0):
                initial_weighted_average_price = sum(min((solar_capacity * s * eff_solar_to_grid + wind_capacity * w * eff_wind_to_grid),grid_connection_capacity) * p if p>0 else 0 for s,w,p in zip(df_solar_data['electricity'],df_wind_data['electricity'],RES_price))/sum(min((solar_capacity * s * eff_solar_to_grid + wind_capacity * w * eff_wind_to_grid),grid_connection_capacity) if p>0 else 0 for s,w,p in zip(df_solar_data['electricity'],df_wind_data['electricity'],RES_price))
                initial_capture_rate = initial_weighted_average_price/average_price

            optimised_weighted_average_price = sum(solar_to_grid_revenues + wind_to_grid_revenues)/sum(net_solar_power_to_grid + net_wind_power_to_grid) if ((optimised_solar_capacity + optimised_wind_capacity) >0) else 0 
            optimised_capture_rate = optimised_weighted_average_price/average_price

        # Calculate the initial and the new grid usages for the year considered
        if not(optimise_RES):
            initial_grid_usage = sum(min((solar_capacity * s * eff_solar_to_grid + wind_capacity * w * eff_wind_to_grid),grid_connection_capacity) for s,w in zip(df_solar_data['electricity'],df_wind_data['electricity']))/len(df_solar_data['electricity'])/grid_connection_capacity

        optimised_grid_usage = (sum(net_solar_power_to_grid) + sum(net_wind_power_to_grid) + sum(net_discharged_WS) + sum(net_discharged_BM) + sum(net_charged_WS) + sum(net_charged_BM) + sum(optimised_DC_high_used) + sum(optimised_DC_low_used))/grid_connection_capacity/time_adjust/len(net_solar_power_to_grid)
        
        #Assess the Annualised Net Present Value
        if assets_coordinated:
            objective_value = total_profit + CM_revenue - (storage_capex_cost * annuity_factor + storage_opex_cost) * power_capacity - (storage_replacement_cost/(1 + interest_rate)**(Lt_cal) * annuity_factor * power_capacity) - (solar_capex_cost * annuity_factor + solar_opex_cost) * optimised_solar_capacity - (wind_capex_cost * annuity_factor + wind_opex_cost) * optimised_wind_capacity - grid_connection_capacity * grid_connection_cost * annuity_factor
        else:
            if (optimised_solar_capacity + optimised_wind_capacity > 0):
                objective_value = total_profit + CM_revenue - (storage_capex_cost * annuity_factor + storage_opex_cost) * power_capacity - (storage_replacement_cost/(1 + interest_rate)**(Lt_cal) * annuity_factor * power_capacity) - power_capacity/(power_capacity + optimised_solar_capacity + optimised_wind_capacity) * grid_connection_capacity * grid_connection_cost * annuity_factor
            else:
                objective_value = total_profit + CM_revenue - (storage_capex_cost * annuity_factor + storage_opex_cost) * power_capacity - (storage_replacement_cost/(1 + interest_rate)**(Lt_cal) * annuity_factor * power_capacity) - grid_connection_capacity * grid_connection_cost * annuity_factor
        
        #Save df_master_data in a csv file
        current_dir = current_dir = os.getcwd() # Find current working directory
        file_name_prefix = result_file_name 
        file_name = file_name_prefix + "_" + start_date + "_to_" + end_date + ".csv"
        file_path = os.path.join(current_dir, file_name)
        df_master_data.to_csv(file_path , index=False)
    
    else:
        # In case of an issue during the optimisation, returns the type of problem. 
        print(status)
        if status == pywraplp.Solver.INFEASIBLE:
            return 'The bidding problem does not have an optimal solution: INFEASIBLE.'
    
        elif status == pywraplp.Solver.UNBOUNDED:
            return 'The bidding problem does not have an optimal solution: UNBOUNDED.'

        elif status == pywraplp.Solver.NOT_SOLVED:
            return 'The bidding problem does not have an optimal solution: NOT_SOLVED.' 
        
        elif status == pywraplp.Solver.FEASIBLE:
            return 'The bidding problem does not have an optimal solution: FEASIBLE.' 
        
        elif status == pywraplp.Solver.ABNORMAL:
            return 'The bidding problem does not have an optimal solution: ABNORMAL.' 


        return ('No Solution')

    # Calc and print timing result
    end_time = time.perf_counter()
    execution_time = end_time - start_time

    if len(generation_type) != 0:
        if not(optimise_RES) and (optimised_solar_capacity + optimised_wind_capacity > 0):
            results = {'solar_capacity': optimised_solar_capacity,
                       'wind_capacity': optimised_wind_capacity,
                       'storage_capacity': power_capacity,
                       'annual_NPV': objective_value,
                       'total_profit': total_profit,
                       'NPV': NPV,
                       'IRR': irr,
                       'PBP': discounted_payback_period,
                       'ini_capture_rate': initial_capture_rate,
                       'opti_capture_rate': optimised_capture_rate,
                       'ini_grid_usage': initial_grid_usage,
                       'opti_grid_usage': optimised_grid_usage,
                       'WS_revenues': WS_revenues,
                       'WS_costs': WS_costs,
                       'BM_revenues': BM_revenues,
                       'BM_costs': BM_costs,
                       'DC_high_revenues': DC_high_revenues,
                       'DC_low_revenues': DC_low_revenues,
                       'RES_revenues': RES_revenues,
                       'RES_to_storage_cost': RES_to_storage_cost,
                       'WS_dchg_tot': WS_dchg_tot,
                       'WS_chg_tot': WS_chg_tot,
                       'BM_dchg_tot': BM_dchg_tot,
                       'BM_chg_tot': BM_chg_tot,
                       'DC_dchg_tot': DC_dchg_tot,
                       'DC_chg_tot': DC_chg_tot,
                       'RES_to_storage_tot': RES_to_storage,
                       'global_NPV': global_NPV,
                       'CM_revenue': CM_revenue,
                       'CM_contract_length': CM_contract_length,
                       'floor_annual_revenue': floor_annual_revenue,
                       'cap_annual_revenue': cap_annual_revenue,
                       'CF_adjustment': CF_adjustment,
                       'execution_time': execution_time
                       }
        else:
            results = {'solar_capacity': optimised_solar_capacity,
                       'wind_capacity': optimised_wind_capacity,
                       'storage_capacity': power_capacity,
                       'annual_NPV': objective_value,
                       'total_profit': total_profit,
                       'NPV': NPV,
                       'IRR': irr,
                       'PBP': discounted_payback_period,
                       'opti_capture_rate': optimised_capture_rate,
                       'opti_grid_usage': optimised_grid_usage,
                       'WS_revenues': WS_revenues,
                       'WS_costs': WS_costs,
                       'BM_revenues': BM_revenues,
                       'BM_costs': BM_costs,
                       'DC_high_revenues': DC_high_revenues,
                       'DC_low_revenues': DC_low_revenues,
                       'RES_revenues': RES_revenues,
                       'RES_to_storage_cost': RES_to_storage_cost,
                       'WS_dchg_tot': WS_dchg_tot,
                       'WS_chg_tot': WS_chg_tot,
                       'BM_dchg_tot': BM_dchg_tot,
                       'BM_chg_tot': BM_chg_tot,
                       'DC_dchg_tot': DC_dchg_tot,
                       'DC_chg_tot': DC_chg_tot,
                       'RES_to_storage_tot': RES_to_storage,
                       'global_NPV': global_NPV,
                       'CM_revenue': CM_revenue,
                       'CM_contract_length': CM_contract_length,
                       'floor_annual_revenue': floor_annual_revenue,
                       'cap_annual_revenue': cap_annual_revenue,
                       'CF_adjustment': CF_adjustment,
                       'execution_time': execution_time
                       }
            
    else:
            results = {'solar_capacity': optimised_solar_capacity,
                       'wind_capacity': optimised_wind_capacity,
                       'storage_capacity': power_capacity,
                       'annual_NPV': objective_value,
                       'total_profit': total_profit,
                       'NPV': NPV,
                       'IRR': irr,
                       'PBP': discounted_payback_period,
                       'opti_grid_usage': optimised_grid_usage,
                       'WS_revenues': WS_revenues,
                       'WS_costs': WS_costs,
                       'BM_revenues': BM_revenues,
                       'BM_costs': BM_costs,
                       'DC_high_revenues': DC_high_revenues,
                       'DC_low_revenues': DC_low_revenues,
                       'RES_revenues': RES_revenues,
                       'RES_to_storage_cost': RES_to_storage_cost,
                       'WS_dchg_tot': WS_dchg_tot,
                       'WS_chg_tot': WS_chg_tot,
                       'BM_dchg_tot': BM_dchg_tot,
                       'BM_chg_tot': BM_chg_tot,
                       'DC_dchg_tot': DC_dchg_tot,
                       'DC_chg_tot': DC_chg_tot,
                       'RES_to_storage_tot': RES_to_storage,
                       'global_NPV': global_NPV,
                       'CM_revenue': CM_revenue,
                       'CM_contract_length': CM_contract_length,
                       'floor_annual_revenue': floor_annual_revenue,
                       'cap_annual_revenue': cap_annual_revenue,
                       'CF_adjustment': CF_adjustment,
                       'execution_time': execution_time
                       }
    return results