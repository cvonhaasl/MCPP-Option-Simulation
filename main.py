# main.py
"""
Satellite Integration Simulation Main Module
=========================================

This module orchestrates the simulation of different satellite integration strategies,
including data loading, scenario generation, and analysis.

Simulation Components
------------------
1. Data Processing:
   - Load satellite configurations
   - Process beam data
   - Handle historical spending

2. Scenario Generation:
   - Generate demand scenarios
   - Project future requirements
   - Handle volatility

3. Strategy Evaluation:
   - Option 1: MILSATCOM Primary (with COMSATCOM reserve)
   - Option 2: COMSATCOM Primary (with MILSATCOM reserve)
   - Option 3: 30/70 Augmentation
   - Option 4: Full Integration

4. Analysis:
   - MAU calculation
   - Cost analysis
   - Coverage metrics
   - Optional sensitivity analysis

Usage
-----
Basic run:
    python main.py

With sensitivity analysis:
    python main.py --sensitivity
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from dataclasses import dataclass
import tqdm
import random
from datetime import datetime
import argparse

# Local imports
from data_processing.import_data import load_data
from data_processing.process_data import (
    process_satellite_data,
    process_beam_data
)
from demand_projection.generate_scenarios import generate_demand_scenarios
from options.option_1_milsatcom import option_1_milsatcom
from options.option_2_comsatcom import option_2_comsatcom
from options.option_3_augmentation import option_3_augmentation
from options.option_4_integration import option_4_integration
from evaluation.mau import calculate_mau
from satellite_operations.performance import (calculate_coverage_area, calculate_redundancy)
from satellite_operations.cost_utils import CostCalculator
from visualization.plotting import (
    PlotManager,
    PlotConfig
)

@dataclass
class SimulationConfig:
    """
    Configuration parameters for the simulation.
    
    Analysis Control
    --------------
    - run_sensitivity: Enable/disable sensitivity analysis
    - sensitivity_samples: Number of samples for sensitivity analysis
    
    Simulation Parameters
    -------------------
    - initial_year: Start year for analysis
    - num_simulations: Number of Monte Carlo iterations
    - output_dir: Directory for results
    """
    # Analysis control
    run_sensitivity: bool = True  # Toggle for sensitivity analysis
    sensitivity_samples: int = 1  # Number of samples for sensitivity
    
    # Base simulation parameters
    initial_year: int = 2024
    num_simulations: int = 1
    output_dir: str = "simulation_results"
    
    # Satellite availability parameters
    satellite_availability_prob: float = 0.3
    milsatcom_availability_prob: float = random.uniform(0.2, 0.6)
    comsatcom_availability_prob: float = random.uniform(0.5, 0.8)

    # Cost factors
    integration_cost_factor: float = 1.0
    integration_time_factor: float = 1.0
    comsatcom_band_cost_factor: float = 1.0
    milsatcom_operating_cost_factor: float = 1.0

    # Integration parameters
    base_integration_cost: float = 1e6
    base_integration_time: float = 365
    max_integration_time: float = 365
    integration_batch_size: int = random.choice([5, 10, 15])

    # Option-specific parameters
    milsatcom_split: float = 0.3
    percentage_use_values: List[int] = None
    urgency_levels: List[str] = None
    urgency_time_multipliers: Dict[str, float] = None

    # Coverage and performance parameters
    coverage_radius_multiplier: float = 1.0
    redundancy_threshold: int = 2
    demand_volatility: float = 5
    number_of_percentage_use_samples: int = 5

    # Regional weights
    regional_weights: Dict[str, float] = None

    def __post_init__(self):
        """Set default values for mutable fields."""
        if self.percentage_use_values is None:
            self.percentage_use_values = [60, 70, 80, 90, 100]
            
        if self.urgency_levels is None:
            self.urgency_levels = ['Standard', 'Accelerated', 'Critical']
            
        if self.urgency_time_multipliers is None:
            self.urgency_time_multipliers = {
                'Standard': 1.0,
                'Accelerated': 0.75,
                'Critical': 0.25
            }
            
        if self.regional_weights is None:
            self.regional_weights = {
                'conus': 0.2,
                'apac': 0.15,
                'mena': 0.15,
                'europe': 0.15,
                'africa': 0.1,
                'latam': 0.1,
                'oceania': 0.15
            }
            
        self._validate_weights()

    def _validate_weights(self):
        """Validate regional weights sum to 1.0."""
        total_weight = sum(self.regional_weights.values())
        if not np.isclose(total_weight, 1.0, rtol=1e-5):
            raise ValueError(f"Regional weights must sum to 1.0, got {total_weight}")

@dataclass
class BandPrices:
    """Price configuration for different frequency bands ($/MHz/month)."""
    c_band: float = 6215.72
    ka_band: float = 9471.58
    ku_band: float = 6579.08
    l_band: float = 81.67
    x_band: float = 15660.46

class SimulationManager:
    """Manages the satellite integration simulation process."""

    def __init__(
        self,
        config: SimulationConfig,
        band_prices: BandPrices,
        log_file: str = "simulation.log",
        discount_rate: float = 0.10
    ):
        """Initialize the simulation manager."""
        self.config = config
        self.band_prices = band_prices
        self.cost_calculator = CostCalculator(discount_rate)
        self._setup_logging(log_file)
        self._setup_output_directory()
        self.plot_manager = PlotManager(save_dir=self.config.output_dir)
        # Define variables for sensitivity analysis
        self.variables_to_test = {
            'MILSATCOM/COMSATCOM Split': [0.2, 0.3, 0.4],
            'Demand Volatility': [0.3, 0.5, 0.7],
            'Regional Weight - CONUS': [0.15, 0.2, 0.25],
            'Regional Weight - APAC': [0.1, 0.15, 0.2],
            'Regional Weight - MENA': [0.1, 0.15, 0.2],
            'Regional Weight - Europe': [0.1, 0.15, 0.2],
            'Regional Weight - Africa': [0.05, 0.1, 0.15],
            'Regional Weight - LATAM': [0.05, 0.1, 0.15],
            'Regional Weight - Oceania': [0.05, 0.1, 0.15],
            'Integration Time - Non-Integrated (days)': [180, 365, 540],
            'MILSATCOM Availability Probability': [0.15, 0.2, 0.25],
            'COMSATCOM Availability Probability': [0.4, 0.5, 0.6],
            'Coverage Radius Multiplier': [0.8, 1.0, 1.2],  # Affects all frequency bands
            'Redundancy Threshold': [1, 2, 3],  # Minimum overlapping beams for redundancy
            'Integration Batch Size': [5, 10, 15]  # Number of satellites to integrate simultaneously
        }
        # Add urgency level impacts
        self.urgency_time_multipliers = {
            'Standard': {'name': 'Standard Integration Time', 'values': [1.0]},
            'Accelerated': {'name': 'Accelerated Integration Time', 'values': [0.5, 0.75]},
            'Critical': {'name': 'Critical Integration Time', 'values': [0.1, 0.25]}
        }
        for urgency, settings in self.urgency_time_multipliers.items():
            self.variables_to_test[settings['name']] = settings['values']
        
        # logging.info(f"Initialized SimulationManager with config: {vars(config)}")

    def _setup_logging(self, log_file: str) -> None:
        """Configure logging settings."""
        logging.basicConfig(
            filename=log_file,
            level=logging.DEBUG,
            format='%(asctime)s:%(levelname)s:%(message)s',
            force=True
        )
        # Also output to console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(console_handler)
        
        # logging.info("Starting new simulation run")

    def _setup_output_directory(self) -> None:
        """Create output directory for results."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def run_simulation(self, data_file: str) -> pd.DataFrame:
        """Run the complete simulation process."""
        try:
            # logging.info("Loading and preprocessing data...")
            data = self._load_and_preprocess_data(data_file)
            
            # logging.info("Generating demand scenarios...")
            scenarios = generate_demand_scenarios(
                data['historical_spending'],
                self.config.demand_volatility
            )
            # logging.info(f"Generated {len(scenarios)} scenarios")
            
            # logging.info("Running Monte Carlo simulations...")
            results_df = self._run_monte_carlo_simulations(data, scenarios)
            
            if results_df.empty:
                logging.error("No simulation results generated!")
                return pd.DataFrame()
            
            # logging.info("Calculating MAU scores...")
            results_df = self._calculate_mau_scores(results_df)
            
            # logging.info("Saving results...")
            self._save_results(results_df)
            
            # logging.info("Generating plots...")
            self._generate_plots(results_df)
            
            return results_df

        except Exception as e:
            logging.error(f"Error in simulation: {str(e)}", exc_info=True)
            raise

    def _load_and_preprocess_data(
        self,
        data_file: str
    ) -> Dict[str, pd.DataFrame]:
        """Load and preprocess input data."""
        # logging.info(f"Loading data from {data_file}")
        
        # Load raw data
        beam_data, comsatcom_data, milsatcom_data, historical_spending = load_data(
            data_file
        )
        
        # Filter historical spending and ensure future years are included
        historical_years = historical_spending[
            (historical_spending['Year'] >= 2010) & 
            (historical_spending['Year'] <= 2021)
        ]
        
        # Add future years (2022-2035) for scenario generation
        future_years = pd.DataFrame({
            'Year': range(2022, 2036)
        })
        
        # Copy the last year's values for future years
        last_year_data = historical_spending[historical_spending['Year'] == 2021].iloc[0]
        for col in historical_spending.columns:
            if col != 'Year':
                future_years[col] = last_year_data[col]
        
        # Combine historical and future data
        historical_spending = pd.concat([historical_years, future_years], ignore_index=True)
        
        # Preprocess satellite data
        comsatcom_data = process_satellite_data(comsatcom_data)
        milsatcom_data = process_satellite_data(milsatcom_data)
        beam_data = process_beam_data(beam_data)
        
        # Verify satellite counts
        self._verify_satellite_counts(comsatcom_data, milsatcom_data)
        
        return {
            'beam_data': beam_data,
            'comsatcom_data': comsatcom_data,
            'milsatcom_data': milsatcom_data,
            'historical_spending': historical_spending
        }

    @staticmethod
    def _verify_satellite_counts(
        comsatcom_data: pd.DataFrame,
        milsatcom_data: pd.DataFrame
    ) -> None:
        """Verify the number of satellites in the data."""
        milsatcom_count = milsatcom_data['Satellite Name'].nunique()
        comsatcom_count = comsatcom_data['Satellite Name'].nunique()
        
        assert milsatcom_count == 43, f"Unexpected MILSATCOM count: {milsatcom_count}"
        assert comsatcom_count == 99, f"Unexpected COMSATCOM count: {comsatcom_count}"
        
        # logging.info(f"Verified satellite counts: {milsatcom_count} MILSATCOM, {comsatcom_count} COMSATCOM")

    def _run_monte_carlo_simulations(
        self,
        data: Dict[str, pd.DataFrame],
        scenarios: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Run Monte Carlo simulations for all scenarios and options,
        distributing the workload across HPC ranks if specified.
        """
        simulation_results = []

        # 1) Filter scenarios to only those that have data in the simulation timeframe
        valid_scenarios = []
        for scenario in scenarios:
            if len(scenario['Year']) > 0:
                valid_scenarios.append(scenario)
        if not valid_scenarios:
            logging.error("No valid scenarios after filtering!")
            return pd.DataFrame()

        # 2) Build a list of all tasks
        tasks = []
        for sim in range(self.config.num_simulations):
            for scenario_idx, scenario in enumerate(valid_scenarios):
                scenario_name = scenario['Name']
                scenario_years = scenario['Year']
                scenario_demands = scenario['Total Bandwidth Needed (Mbps)']
                for urgency in self.config.urgency_levels:
                    for percentage_use in self.config.percentage_use_values:
                        for year, demand_mbps in zip(scenario_years, scenario_demands):
                            # Each task is a tuple with all info needed for one loop iteration
                            tasks.append((sim, scenario_idx, urgency, percentage_use, year, demand_mbps))

        total_tasks = len(tasks)
        if total_tasks == 0:
            logging.error("No tasks built for simulation!")
            return pd.DataFrame()

        # 3) HPC rank/size from config
        rank = getattr(self.config, 'hpc_rank', 0)
        size = getattr(self.config, 'hpc_size', 1)

        # 4) Distribute tasks across ranks: rank i does tasks[i::size]
        my_tasks = tasks[rank::size]
        logging.info(f"Rank {rank}: processing {len(my_tasks)} tasks out of {total_tasks} total")

        # 5) Process tasks
        from tqdm import tqdm
        with tqdm(total=len(my_tasks), desc=f"Rank {rank} Sim") as pbar:
            for (sim, scenario_idx, urgency, percentage_use, year, demand_mbps) in my_tasks:
                scenario = valid_scenarios[scenario_idx]
                scenario_name = scenario['Name']

                # Evaluate all options for this single iteration
                option_results = self._evaluate_options(
                    data,
                    demand_mbps,
                    urgency,
                    percentage_use,
                    year,
                    self.config.initial_year,
                    scenario_name,
                    sim
                )
                simulation_results.extend(option_results)
                pbar.update(1)

        # 6) Convert partial results to DataFrame
        results_df = pd.DataFrame(simulation_results)
        logging.info(f"Rank {rank}: generated {results_df.shape[0]} rows of partial results")

        # 7) Write partial CSV for this rank
        #    (You can merge them offline after the job completes)
        output_file = Path(self.config.output_dir) / f"simulation_results_rank_{rank}.csv"
        results_df.to_csv(output_file, index=False)
        logging.info(f"Rank {rank}: wrote partial results to {output_file}")

        return results_df

    def _evaluate_scenario(
        self,
        data: Dict[str, pd.DataFrame],
        scenario: Dict[str, Any],
        sim_number: int,
        pbar: tqdm.tqdm
    ) -> List[Dict[str, Any]]:
        """Evaluate all options for a given scenario."""
        scenario_results = []
        
        scenario_name = scenario['Name']
        years = scenario['Year']
        total_bandwidth_needed = scenario['Total Bandwidth Needed (Mbps)']
        initial_year = self.config.initial_year
        
        logging.debug(f"Evaluating scenario {scenario_name}")
        logging.debug(f"Years in scenario: {min(years)} to {max(years)}")
        logging.debug(f"Number of years: {len(years)}")
        
        for urgency in self.config.urgency_levels:
            for percentage_use in self.config.percentage_use_values:
                logging.debug(f"Processing urgency: {urgency}, percentage use: {percentage_use}")
                for idx, (year, demand_mbps) in enumerate(
                    zip(years, total_bandwidth_needed)
                ):
                    logging.debug(f"Processing year {year} with demand {demand_mbps}")
                    
                    # Evaluate each option
                    option_results = self._evaluate_options(
                        data,
                        demand_mbps,
                        urgency,
                        percentage_use,
                        year,
                        initial_year,
                        scenario_name,
                        sim_number
                    )
                    
                    if option_results:
                        logging.debug(f"Generated {len(option_results)} results for year {year}")
                        scenario_results.extend(option_results)
                    else:
                        logging.warning(f"No results generated for year {year}")
                    
                    pbar.update(1)
        
        logging.info(f"Total results generated for scenario {scenario_name}: {len(scenario_results)}")
        return scenario_results

    def _evaluate_options(
        self,
        data: Dict[str, pd.DataFrame],
        demand_mbps: float,
        urgency: str,
        percentage_use: float,
        year: int,
        initial_year: int,
        scenario_name: str,
        sim_number: int
    ) -> List[Dict[str, Any]]:
        """Evaluate all options for given parameters."""
        results = []
        base_config = self._get_base_config()
        
        # Option 1: MILSATCOM Primary with COMSATCOM Reserve
        option1_result = option_1_milsatcom(
            data['milsatcom_data'].copy(),
            data['comsatcom_data'].copy(),
            demand_mbps,
            urgency,
            percentage_use,
            base_config
        )
        results.append(
            self._process_option_result(
                option1_result,
                'Option 1',
                data['beam_data'],
                demand_mbps,
                urgency,
                percentage_use,
                scenario_name,
                sim_number,
                year
            )
        )
        
        # Option 2: COMSATCOM Primary with MILSATCOM Reserve
        option2_result = option_2_comsatcom(
            data['comsatcom_data'].copy(),
            data['milsatcom_data'].copy(),
            demand_mbps,
            urgency,
            percentage_use,
            base_config
        )
        results.append(
            self._process_option_result(
                option2_result,
                'Option 2',
                data['beam_data'],
                demand_mbps,
                urgency,
                percentage_use,
                scenario_name,
                sim_number,
                year
            )
        )
        
        # Option 3: Augmentation (30/70 split)
        option3_result = option_3_augmentation(
            data['milsatcom_data'].copy(),
            data['comsatcom_data'].copy(),
            demand_mbps,
            urgency,
            percentage_use,
            base_config
        )
        results.append(
            self._process_option_result(
                option3_result,
                'Option 3',
                data['beam_data'],
                demand_mbps,
                urgency,
                percentage_use,
                scenario_name,
                sim_number,
                year
            )
        )
        
        # Option 4: Integration
        option4_result = option_4_integration(
            data['milsatcom_data'].copy(),
            data['comsatcom_data'].copy(),
            demand_mbps,
            urgency,
            percentage_use,
            base_config,
            year,
            initial_year
        )
        results.append(
            self._process_option_result(
                option4_result,
                'Option 4',
                data['beam_data'],
                demand_mbps,
                urgency,
                percentage_use,
                scenario_name,
                sim_number,
                year
            )
        )
        
        return results

    def _get_base_config(self, include_band_prices: bool = True) -> Dict[str, Any]:
        """Get base configuration dictionary."""
        base_config = {
            # Availability probabilities
            'satellite_availability_prob': self.config.satellite_availability_prob,
            'milsatcom_availability_prob': self.config.milsatcom_availability_prob,
            'comsatcom_availability_prob': self.config.comsatcom_availability_prob,
            
            # Cost factors
            'integration_cost_factor': self.config.integration_cost_factor,
            'integration_time_factor': self.config.integration_time_factor,
            'comsatcom_band_cost_factor': self.config.comsatcom_band_cost_factor,
            'milsatcom_operating_cost_factor': self.config.milsatcom_operating_cost_factor,
            
            # Integration parameters
            'base_integration_cost': self.config.base_integration_cost,
            'base_integration_time': self.config.base_integration_time,
            'max_integration_time': self.config.max_integration_time,
            'integration_batch_size': self.config.integration_batch_size,
            
            # Coverage parameters
            'coverage_radius_multiplier': self.config.coverage_radius_multiplier,
            'redundancy_threshold': self.config.redundancy_threshold,
            
            # Option-specific parameters
            'milsatcom_split': self.config.milsatcom_split,
            'urgency_time_multipliers': self.config.urgency_time_multipliers
        }
        
        if include_band_prices:
            base_config['band_prices'] = {
                'c': self.band_prices.c_band,
                'ka': self.band_prices.ka_band,
                'ku': self.band_prices.ku_band,
                'l': self.band_prices.l_band,
                'x': self.band_prices.x_band
            }
        
        return base_config

    def _process_option_result(
        self,
        option_result: Dict[str, Any],
        option_name: str,
        beam_data: pd.DataFrame,
        demand_mbps: float,
        urgency: str,
        percentage_use: float,
        scenario_name: str,
        sim_number: int,
        year: int
    ) -> Dict[str, Any]:
        """Process and format option results."""
        try:
            satellite_data = option_result.get('Satellite Data', pd.DataFrame())
            
            # Calculate performance metrics
            total_capacity = option_result.get('Total Capacity (Mbps)', 0.0)
            demand_met = total_capacity >= demand_mbps
            
            # Calculate time to meet demand based on whether demand is met
            if not demand_met:
                time_to_meet = self.config.max_integration_time
            else:
                integration_time = option_result.get('Time to Meet Demand (days)', 0.0)
                time_to_meet = min(integration_time, self.config.max_integration_time)
            
            # Calculate costs using cost calculator
            band_prices_dict = {
                'c': self.band_prices.c_band,
                'ka': self.band_prices.ka_band,
                'ku': self.band_prices.ku_band,
                'l': self.band_prices.l_band,
                'x': self.band_prices.x_band
            }
            
            total_cost = self.cost_calculator.calculate_total_system_cost(
                satellite_data,
                option_name,
                band_prices_dict,
                year,
                self.config.initial_year,
                self.config.milsatcom_operating_cost_factor,
                self.config.comsatcom_band_cost_factor
            )
            
            # Calculate coverage and redundancy and mau
            coverage_area = calculate_coverage_area(satellite_data, beam_data)
            redundancy = calculate_redundancy(satellite_data, beam_data)
            mau_weights = {
                'time': 0.4,
                'performance': 0.3,
                'coverage': 0.15,
                'redundancy': 0.15
            }
            attributes = {
                'time_to_meet_demand': time_to_meet,
                'performance': total_capacity,
                'demand': demand_mbps,  # Required for the performance SAU.
                'coverage': coverage_area,
                'redundancy': redundancy
            }
            mau_value = calculate_mau(attributes, mau_weights)


            
            return {
                'Simulation': sim_number,
                'Scenario': scenario_name,
                'Year': year,
                'Demand (Mbps)': demand_mbps,
                'Option': option_name,
                'Urgency': urgency,
                'Percentage Use': percentage_use,
                'Total Cost ($)': total_cost,
                'NPV Base Year': self.config.initial_year,
                'Discount Rate': self.cost_calculator.discount_rate,
                'Integration Cost ($)': option_result.get('Integration Cost ($)', 0.0),
                'Time to Meet Demand (days)': time_to_meet,
                'Coverage Area (sq km)': coverage_area,
                'Redundancy': redundancy,
                'Performance (Mbps)': total_capacity,
                'Total Capacity (Mbps)': total_capacity,
                'Satellites Used': option_result.get('Satellites Used', 0),
                'Satellites Available': option_result.get('Satellites Available', 0),
                'Demand Met': demand_met,
                'MAU': mau_value
            }
            
        except Exception as e:
            logging.error(f"Error in _process_option_result: {str(e)}")
            logging.debug(f"Option result: {option_result}")
            raise

    def _calculate_mau_scores(
        self,
        results_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate MAU scores for all results."""
        
        mau_weights = {
            'time': 0.3,
            'performance': 0.4,
            'coverage': 0.15,
            'redundancy': 0.15
        }
        
        results_df['MAU'] = results_df.apply(
            lambda row: calculate_mau({
                'time_to_meet_demand': row['Time to Meet Demand (days)'],
                'performance': row['Performance (Mbps)'],
                'demand': row['Demand (Mbps)'],
                'coverage': row['Coverage Area (sq km)'],
                'redundancy': row['Redundancy']
            }, mau_weights),
            axis=1
        )
        return results_df

    def _save_results(self, results_df: pd.DataFrame) -> None:
        """Save simulation results to files with only data from 2024 onward."""
        output_file = Path(self.config.output_dir) / f"simulation_results_{self.timestamp}.csv"

        if "Year" in results_df.columns:
            results_df = results_df[results_df["Year"] >= 2024]

        results_df.to_csv(output_file, index=False)
        logging.info(f"Saved filtered results (Year >= 2024) to {output_file}")

    def _generate_plots(self, results_df: pd.DataFrame) -> None:
        """Generate and save all visualization plots."""
        try:
            # Cost vs MAU plot
            self.plot_manager.plot_cost_vs_mau(results_df)
            
            # Tradespace analysis
            self.plot_manager.plot_tradespace(results_df)
            
            # MAU S-curve
            self.plot_manager.plot_mau_s_curve(results_df)
            
            # Coverage and redundancy plots
            self.plot_manager.plot_coverage_and_redundancy(results_df)
            
            # Generate sensitivity analysis and tornado diagram
            sensitivity_results = self._perform_sensitivity_analysis(results_df)
            self.plot_manager.plot_tornado_diagram(sensitivity_results)
            
            logging.info("Generated all plots successfully")

        except Exception as e:
            logging.error(f"Error generating plots: {str(e)}")
            raise

    def _perform_sensitivity_analysis(
        self,
        results_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Perform sensitivity analysis on simulation parameters."""
        try:
            sensitivity_results = []
            self.base_mau = results_df['MAU'].mean()
            logging.info(f"Base MAU value: {self.base_mau}")

            for variable, values in self.variables_to_test.items():
                logging.info(f"Analyzing sensitivity for {variable}")
                impacts = []
                
                for value in values:
                    # Create modified configuration
                    modified_config = self._create_modified_config(variable, value)
                    
                    # Run targeted simulation with modified parameter
                    modified_results = self._run_targeted_simulation(
                        modified_config,
                        results_df,
                        variable,
                        value
                    )
                    
                    # Calculate impact based on actual simulation results
                    modified_mau = self._calculate_modified_mau(modified_results)
                    impact = ((modified_mau - self.base_mau) / self.base_mau) * 100
                    impacts.append(impact)

                sensitivity_results.append({
                    'Variable': variable,
                    'Low_Impact': min(impacts),
                    'High_Impact': max(impacts),
                    'Impact Range': max(impacts) - min(impacts)
                })

            sensitivity_df = pd.DataFrame(sensitivity_results)
            
            # Debug logging
            logging.debug(f"Created sensitivity DataFrame with columns: {sensitivity_df.columns}")
            logging.debug(f"Sensitivity results shape: {sensitivity_df.shape}")
            
            sensitivity_df.sort_values('Impact Range', ascending=False, inplace=True)
            
            return sensitivity_df

        except Exception as e:
            logging.error(f"Error in sensitivity analysis: {str(e)}")
            raise

    def _create_modified_config(
        self,
        variable: str,
        value: Any
    ) -> SimulationConfig:
        """Create modified configuration for sensitivity analysis."""
        base_config = {
            'satellite_availability_prob': self.config.satellite_availability_prob,
            'base_integration_cost': self.config.base_integration_cost,
            'base_integration_time': self.config.base_integration_time,
            'integration_cost_factor': self.config.integration_cost_factor,
            'integration_time_factor': self.config.integration_time_factor,
            'comsatcom_band_cost_factor': self.config.comsatcom_band_cost_factor,
            'milsatcom_operating_cost_factor': self.config.milsatcom_operating_cost_factor,
            'num_simulations': 1  # Reduced for sensitivity analysis
        }

        modified_config = SimulationConfig(**base_config)

        if variable == 'Percentage Use':
            modified_config.percentage_use_values = [value]
        elif variable == 'Integration Cost Factor':
            modified_config.integration_cost_factor = value
        elif variable == 'Integration Time Factor':
            modified_config.integration_time_factor = value
        elif variable == 'Demand Volatility':
            modified_config.demand_volatility = value
        elif variable == 'COMSATCOM Band Cost Factor':
            modified_config.comsatcom_band_cost_factor = value
        elif variable == 'MILSATCOM Operating Cost Factor':
            modified_config.milsatcom_operating_cost_factor = value
        elif variable == 'Urgency Level':
            modified_config.urgency_levels = [value]

        return modified_config

    def _run_targeted_simulation(
        self,
        modified_config: SimulationConfig,
        base_results: pd.DataFrame,
        variable: str,
        value: Any
    ) -> pd.DataFrame:
        """Run a targeted simulation with modified parameters."""
        data = self._load_and_preprocess_data('Data_Upload_CASRv2.xlsx')
        
        if variable == 'Demand Volatility':
            scenarios = generate_demand_scenarios(
                data['historical_spending'],
                volatility_std=value
            )
        else:
            scenarios = generate_demand_scenarios(
                data['historical_spending'],
                self.config.demand_volatility
            )

        test_conditions = self._select_test_conditions(base_results)
        
        modified_results = []
        for condition in test_conditions:
            option_results = self._evaluate_options(
                data,
                condition['demand_mbps'],
                condition['urgency'],
                condition['percentage_use'],
                condition['year'],
                self.config.initial_year,
                condition['scenario'],
                0
            )
            modified_results.extend(option_results)

        return pd.DataFrame(modified_results)

    def _select_test_conditions(
        self,
        base_results: pd.DataFrame,
        num_samples: int = 10
    ) -> List[Dict[str, Any]]:
        """Select representative test conditions for sensitivity analysis."""
        stratified_samples = base_results.groupby(
            ['Scenario', 'Year']
        ).apply(
            lambda x: x.sample(
                n=min(2, len(x)),
                random_state=42
            )
        ).reset_index(drop=True)

        conditions = []
        for _, row in stratified_samples.iterrows():
            condition = {
                'demand_mbps': row['Demand (Mbps)'],
                'urgency': row['Urgency'],
                'percentage_use': row['Percentage Use'],
                'year': row['Year'],
                'scenario': row['Scenario']
            }
            conditions.append(condition)

        return conditions[:num_samples]

    def _calculate_modified_mau(
        self,
        modified_results: pd.DataFrame
    ) -> float:
        """Calculate MAU scores for modified results."""
        if modified_results.empty:
            return 0.0
            
        return self._calculate_mau_scores(modified_results)['MAU'].mean()


def main(run_sensitivity: bool = False) -> pd.DataFrame:
    """
    Main entry point for the simulation.
    
    Args:
        run_sensitivity: Toggle sensitivity analysis
    """
    import sys
    import logging

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('simulation.log'),
            logging.StreamHandler()
        ]
    )

    # Attempt to parse rank/size from sys.argv
    rank = 0
    size = 1
    if len(sys.argv) >= 3:
        try:
            rank = int(sys.argv[1])
            size = int(sys.argv[2])
            logging.info(f"Running on HPC with rank={rank}, size={size}")
        except ValueError:
            logging.warning(
                "Could not parse rank/size from sys.argv. "
                "Defaulting to rank=0, size=1"
            )

    # Create configuration with sensitivity control
    config = SimulationConfig(
        run_sensitivity=run_sensitivity,
        sensitivity_samples=10 if run_sensitivity else 0
    )
    
    # Store rank/size in config for HPC
    config.hpc_rank = rank
    config.hpc_size = size

    band_prices = BandPrices()
    
    # Initialize simulation manager
    manager = SimulationManager(config, band_prices)
    
    try:
        # Run simulation
        results_df = manager.run_simulation('Data_Upload_CASRv2.xlsx')
        logging.info(
            f"Simulation completed successfully "
            f"{'with' if run_sensitivity else 'without'} "
            "sensitivity analysis"
        )
        return results_df
        
    except Exception as e:
        logging.error(f"Simulation failed: {str(e)}")
        raise

if __name__ == '__main__':
    import sys
    
    # Set random seeds for reproducibility
    base_seed = 42
    rank = 0
    seed = base_seed + rank
    np.random.seed(seed)
    random.seed(seed)
    
    # Check if running in Jupyter
    is_jupyter = 'ipykernel' in sys.modules
    
    if is_jupyter:
        # If in Jupyter, just run without argument parsing
        results = main(run_sensitivity=False)
    else:
        # If running from command line, use argument parsing
        parser = argparse.ArgumentParser(
            description='Run satellite integration simulation'
        )
        parser.add_argument(
            '--sensitivity',
            action='store_true',
            help='Enable sensitivity analysis'
        )
        
        args = parser.parse_args()
        results = main(run_sensitivity=args.sensitivity)