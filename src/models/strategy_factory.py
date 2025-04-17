"""
Strategy factory for creating and configuring betting strategies.

This module provides a factory pattern implementation for instantiating
and managing various betting strategies with different configurations.
"""

from typing import Dict, List, Any, Optional, Type, Union
import json
import logging
from pathlib import Path
import importlib
import inspect
import yaml

# Import betting strategies
from src.models.betting_strategies import (
    BettingStrategy,
    ValueBettingStrategy,
    DrawNoBetStrategy,
    AsianHandicapStrategy,
    ModelEnsembleStrategy,
    MarketMovementStrategy
)

logger = logging.getLogger(__name__)


class StrategyFactory:
    """
    Factory for creating and managing betting strategies.
    
    This class provides methods for instantiating betting strategies with
    various configurations, loading/saving strategy configurations, and
    registering custom strategies.
    """
    
    def __init__(self):
        """Initialize the strategy factory."""
        # Dictionary mapping strategy names to strategy classes
        self._strategies: Dict[str, Type[BettingStrategy]] = {
            'value': ValueBettingStrategy,
            'draw_no_bet': DrawNoBetStrategy,
            'asian_handicap': AsianHandicapStrategy,
            'ensemble': ModelEnsembleStrategy,
            'market_movement': MarketMovementStrategy
        }
        
        # Dictionary mapping strategy names to default configurations
        self._default_configs: Dict[str, Dict[str, Any]] = {
            'value': {
                'min_edge': 0.05,
                'min_odds': 1.5,
                'max_odds': 7.0,
                'stake_percentage': 1.0,
                'confidence_threshold': 0.7,
                'progressive_staking': True
            },
            'draw_no_bet': {
                'confidence_threshold': 0.6,
                'min_odds': 1.3,
                'max_odds': 4.0,
                'stake_percentage': 1.0
            },
            'asian_handicap': {
                'min_edge': 0.03,
                'max_handicap': 2.0,
                'confidence_threshold': 0.65,
                'min_odds': 1.7,
                'max_odds': 2.3,
                'stake_percentage': 1.0
            },
            'ensemble': {
                'strategies': [],
                'weights': [],
                'min_consensus': 2,
                'stake_percentage': 1.0
            },
            'market_movement': {
                'threshold_percentage': 0.05,
                'min_odds': 1.5,
                'max_odds': 5.0,
                'stake_percentage': 0.8,
                'min_timeframe_hours': 1
            }
        }
    
    def register_strategy(self, 
                         name: str, 
                         strategy_class: Type[BettingStrategy],
                         default_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a new strategy type with the factory.
        
        Args:
            name: Name to register the strategy under
            strategy_class: The strategy class to register
            default_config: Optional default configuration for the strategy
        """
        if not issubclass(strategy_class, BettingStrategy):
            raise ValueError(f"Strategy class must be a subclass of BettingStrategy")
        
        self._strategies[name] = strategy_class
        
        if default_config is not None:
            self._default_configs[name] = default_config
    
    def create_strategy(self, 
                       strategy_type: str, 
                       config: Optional[Dict[str, Any]] = None) -> BettingStrategy:
        """
        Create a strategy instance of the specified type.
        
        Args:
            strategy_type: Type of strategy to create
            config: Optional configuration for the strategy
            
        Returns:
            Instance of the requested strategy
            
        Raises:
            ValueError: If the strategy type is not registered
        """
        if strategy_type not in self._strategies:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        strategy_class = self._strategies[strategy_type]
        
        # Start with default configuration for this strategy type
        final_config = self._default_configs.get(strategy_type, {}).copy()
        
        # Update with provided configuration if any
        if config:
            final_config.update(config)
        
        # Create and return strategy instance
        try:
            # Get the required parameters for the strategy class constructor
            signature = inspect.signature(strategy_class.__init__)
            required_params = {
                name: param for name, param in signature.parameters.items()
                if param.default == inspect.Parameter.empty and name != 'self'
            }
            
            # Check if all required parameters are provided
            missing_params = [name for name in required_params if name not in final_config]
            if missing_params:
                raise ValueError(f"Missing required parameters for {strategy_type} strategy: {missing_params}")
            
            # Filter configuration to include only parameters accepted by the constructor
            valid_params = {
                name: value for name, value in final_config.items()
                if name in signature.parameters
            }
            
            return strategy_class(**valid_params)
        
        except Exception as e:
            logger.error(f"Error creating strategy of type {strategy_type}: {str(e)}")
            raise
    
    def create_from_config_file(self, 
                              config_file: Union[str, Path]) -> Dict[str, BettingStrategy]:
        """
        Create multiple strategies from a configuration file.
        
        Args:
            config_file: Path to the configuration file (JSON or YAML)
            
        Returns:
            Dictionary mapping strategy names to strategy instances
            
        Raises:
            ValueError: If the configuration file is invalid or cannot be read
        """
        config_path = Path(config_file)
        if not config_path.exists():
            raise ValueError(f"Configuration file not found: {config_file}")
        
        try:
            # Load configuration file
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'r') as file:
                    config_data = yaml.safe_load(file)
            else:  # Default to JSON
                with open(config_path, 'r') as file:
                    config_data = json.load(file)
            
            strategies = {}
            
            # Create strategies from configuration
            for strategy_name, strategy_config in config_data.items():
                strategy_type = strategy_config.pop('type', None)
                if not strategy_type:
                    logger.warning(f"Strategy {strategy_name} has no type specified, skipping")
                    continue
                
                try:
                    strategy = self.create_strategy(strategy_type, strategy_config)
                    strategies[strategy_name] = strategy
                except Exception as e:
                    logger.error(f"Error creating strategy {strategy_name}: {str(e)}")
            
            return strategies
        
        except Exception as e:
            logger.error(f"Error loading configuration file: {str(e)}")
            raise ValueError(f"Failed to load configuration from {config_file}: {str(e)}")
    
    def save_config_to_file(self, 
                          strategies: Dict[str, Dict[str, Any]],
                          config_file: Union[str, Path],
                          format: str = 'json') -> None:
        """
        Save strategy configurations to a file.
        
        Args:
            strategies: Dictionary mapping strategy names to configurations
            config_file: Path to save the configuration file
            format: Format to save the file in ('json' or 'yaml')
            
        Raises:
            ValueError: If the format is invalid or the file cannot be written
        """
        config_path = Path(config_file)
        
        try:
            # Create parent directories if they don't exist
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save configuration file
            if format.lower() == 'yaml':
                with open(config_path, 'w') as file:
                    yaml.dump(strategies, file, default_flow_style=False)
            elif format.lower() == 'json':
                with open(config_path, 'w') as file:
                    json.dump(strategies, file, indent=2)
            else:
                raise ValueError(f"Invalid format: {format}. Must be 'json' or 'yaml'")
            
            logger.info(f"Strategy configurations saved to {config_path}")
        
        except Exception as e:
            logger.error(f"Error saving configuration file: {str(e)}")
            raise ValueError(f"Failed to save configuration to {config_file}: {str(e)}")
    
    def get_strategy_parameters(self, strategy_type: str) -> Dict[str, Any]:
        """
        Get the parameters for a strategy type.
        
        Args:
            strategy_type: Type of strategy
            
        Returns:
            Dictionary with parameter names and default values
            
        Raises:
            ValueError: If the strategy type is not registered
        """
        if strategy_type not in self._strategies:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        strategy_class = self._strategies[strategy_type]
        
        # Get the parameters from the constructor signature
        signature = inspect.signature(strategy_class.__init__)
        
        # Filter out 'self' parameter
        parameters = {
            name: param.default if param.default != inspect.Parameter.empty else None
            for name, param in signature.parameters.items()
            if name != 'self'
        }
        
        # Add parameter types and descriptions from docstring if available
        parameters_info = {}
        for name, default in parameters.items():
            param_info = {
                'default': default,
                'required': default == None
            }
            
            # Try to get type hint information
            try:
                type_hints = getattr(strategy_class.__init__, '__annotations__', {})
                if name in type_hints:
                    param_info['type'] = str(type_hints[name])
            except Exception:
                pass
            
            parameters_info[name] = param_info
        
        return parameters_info
    
    def list_available_strategies(self) -> Dict[str, Dict[str, Any]]:
        """
        List all available strategies with their default configurations.
        
        Returns:
            Dictionary mapping strategy names to default configurations
        """
        result = {}
        for name, strategy_class in self._strategies.items():
            result[name] = {
                'class': strategy_class.__name__,
                'description': strategy_class.__doc__.split('\n')[0] if strategy_class.__doc__ else '',
                'default_config': self._default_configs.get(name, {}),
                'parameters': self.get_strategy_parameters(name)
            }
        return result
    
    def create_ensemble_strategy(self, 
                               strategies: List[BettingStrategy],
                               weights: Optional[List[float]] = None,
                               min_consensus: int = 2,
                               stake_percentage: float = 1.0) -> ModelEnsembleStrategy:
        """
        Create an ensemble strategy from multiple strategies.
        
        Args:
            strategies: List of strategy instances to include in the ensemble
            weights: Optional list of weights for each strategy
            min_consensus: Minimum number of strategies that must agree
            stake_percentage: Percentage of standard stake to use
            
        Returns:
            ModelEnsembleStrategy instance
        """
        if len(strategies) < 2:
            raise ValueError("Ensemble strategy requires at least 2 strategies")
        
        # Use equal weights if not provided
        if weights is None:
            weights = [1.0] * len(strategies)
        
        # Normalize weights
        if sum(weights) != 1.0:
            total = sum(weights)
            weights = [w / total for w in weights]
        
        return ModelEnsembleStrategy(
            strategies=strategies,
            weights=weights,
            min_consensus=min_consensus,
            stake_percentage=stake_percentage
        )
    
    def load_dynamic_strategies(self, module_path: str) -> int:
        """
        Load strategy classes from a Python module dynamically.
        
        Args:
            module_path: Import path to the module containing strategy classes
            
        Returns:
            Number of strategies loaded
            
        Raises:
            ImportError: If the module cannot be imported
        """
        try:
            # Import the module
            module = importlib.import_module(module_path)
            
            # Find all BettingStrategy subclasses in the module
            strategy_classes = []
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BettingStrategy) and 
                    obj != BettingStrategy):
                    strategy_classes.append((name, obj))
            
            # Register each strategy class
            count = 0
            for name, cls in strategy_classes:
                # Convert CamelCase to snake_case for the strategy name
                strategy_name = ''.join(['_' + c.lower() if c.isupper() else c 
                                       for c in name]).lstrip('_')
                if strategy_name.endswith('_strategy'):
                    strategy_name = strategy_name[:-9]  # Remove '_strategy' suffix
                
                # Only register if not already registered
                if strategy_name not in self._strategies:
                    self.register_strategy(strategy_name, cls)
                    count += 1
            
            return count
        
        except Exception as e:
            logger.error(f"Error loading dynamic strategies from {module_path}: {str(e)}")
            raise


# Single instance for convenience
strategy_factory = StrategyFactory()


def create_strategy(strategy_type: str, config: Optional[Dict[str, Any]] = None) -> BettingStrategy:
    """
    Convenience function to create a strategy using the global factory.
    
    Args:
        strategy_type: Type of strategy to create
        config: Optional configuration for the strategy
        
    Returns:
        Instance of the requested strategy
    """
    return strategy_factory.create_strategy(strategy_type, config)


def load_strategies_from_config(config_file: Union[str, Path]) -> Dict[str, BettingStrategy]:
    """
    Convenience function to load strategies from a configuration file.
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Dictionary mapping strategy names to strategy instances
    """
    return strategy_factory.create_from_config_file(config_file) 