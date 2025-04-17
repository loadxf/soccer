#!/bin/bash
# Script to fix missing calculate_player_form function in Soccer Prediction System
# Error: ImportError: cannot import name 'calculate_player_form' from 'src.data.features'

set -e  # Exit on error

echo "Soccer Prediction System - Player Form Function Fix"
echo "=================================================="

# Create a backup of the current features.py file
echo "Creating backup of features.py..."
cp src/data/features.py src/data/features.py.bak2

echo "Updating src/data/features.py with calculate_player_form function..."

# Append the missing function to the existing features.py file
cat >> src/data/features.py << 'EOF'

# Default configuration for feature calculations
DEFAULT_CONFIG = {
    "window_sizes": [3, 5, 10],
    "min_matches_required": 3,
    "decay_factor": 0.9,
    "metrics": [
        "goals", "assists", "minutes_played", "yellow_cards", "red_cards",
        "shots", "shots_on_target", "passes", "key_passes", "tackles"
    ]
}

def validate_dataframe(df: pd.DataFrame, feature_type: str) -> Tuple[bool, List[str]]:
    """
    Validate input DataFrame for specific feature calculation.
    
    Args:
        df: Input DataFrame
        feature_type: Type of feature being calculated
        
    Returns:
        Tuple of (valid, errors)
    """
    errors = []
    
    # Check if DataFrame is empty
    if df.empty:
        errors.append("Input DataFrame is empty")
        return False, errors
    
    # Specific validations based on feature type
    if feature_type == "player_form":
        required_cols = ["player_id", "match_id", "match_date", "team_id"]
        for col in required_cols:
            if col not in df.columns:
                errors.append(f"Required column '{col}' not found")
        
    elif feature_type == "team_form":
        required_cols = ["team_id", "match_id", "match_date", "opponent_id"]
        for col in required_cols:
            if col not in df.columns:
                errors.append(f"Required column '{col}' not found")
    
    return len(errors) == 0, errors

def preprocess_data(df: pd.DataFrame, feature_type: str, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Preprocess data for feature calculation.
    
    Args:
        df: Input DataFrame
        feature_type: Type of feature being calculated
        config: Configuration dictionary
        
    Returns:
        Preprocessed DataFrame
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Convert date columns to datetime if they're not already
    if "match_date" in result_df.columns and not pd.api.types.is_datetime64_dtype(result_df["match_date"]):
        result_df["match_date"] = pd.to_datetime(result_df["match_date"])
        
    # Sort by date for time series operations
    if "match_date" in result_df.columns:
        result_df = result_df.sort_values("match_date")
    
    # Fill missing values with sensible defaults
    if feature_type == "player_form":
        # Fill missing numeric columns with 0
        for metric in config["metrics"]:
            if metric in result_df.columns:
                result_df[metric] = result_df[metric].fillna(0)
        
        # Ensure minutes_played is at least 1 to avoid division by zero
        if "minutes_played" in result_df.columns:
            result_df["minutes_played"] = result_df["minutes_played"].fillna(0).clip(lower=1)
    
    return result_df

def calculate_player_form(
    df: pd.DataFrame,
    config: Dict[str, Any] = None,
    show_progress: bool = False,
    window: int = 5
) -> pd.DataFrame:
    """
    Calculate player form features.
    
    Args:
        df: Input DataFrame with player match data
        config: Configuration dictionary
        show_progress: Whether to show a progress bar
        window: Window size for rolling calculations
        
    Returns:
        DataFrame with player form features
    """
    # Validate input data
    is_valid, errors = validate_dataframe(df, "player_form")
    if not is_valid:
        logger.error(f"Invalid input data for player form calculation: {errors}")
        return pd.DataFrame()
    
    # Use default config with any overrides
    if config is None:
        config = DEFAULT_CONFIG.copy()
    else:
        default_config = DEFAULT_CONFIG.copy()
        default_config.update(config)
        config = default_config
    
    # Preprocess data
    df = preprocess_data(df, "player_form", config)
    
    # Prepare result dataframe
    players = df['player_id'].unique()
    result_dfs = []
    
    # Create progress bar if requested
    players_iter = players
    if show_progress:
        try:
            from tqdm import tqdm
            players_iter = tqdm(players, desc="Calculating player form")
        except ImportError:
            logger.warning("tqdm not available, progress bar disabled")
    
    for player in players_iter:
        # Get player matches
        player_df = df[df['player_id'] == player].sort_values('match_date')
        
        if len(player_df) < config['min_matches_required']:
            logger.warning(f"Player {player} has fewer than {config['min_matches_required']} matches, skipping")
            continue
        
        # Calculate basic performance metrics per 90 minutes
        for metric in ['goals', 'assists', 'shots', 'key_passes']:
            if metric in player_df.columns:
                player_df[f'{metric}_per_90'] = player_df[metric] * 90 / player_df['minutes_played'].clip(lower=1)
        
        if 'goals' in player_df.columns and 'assists' in player_df.columns:
            player_df['goal_contributions'] = player_df['goals'] + player_df['assists']
            player_df['goal_contributions_per_90'] = player_df['goal_contributions'] * 90 / player_df['minutes_played'].clip(lower=1)
        
        # Dynamically add rolling calculations for all window sizes
        player_form_df = player_df[['player_id', 'match_id', 'match_date', 'team_id']].copy()
        
        for window_size in config['window_sizes']:
            for metric in config['metrics']:
                if metric in player_df.columns:
                    # Calculate rolling metrics
                    player_form_df[f'{metric}_last{window_size}'] = player_df[metric].rolling(window=window_size, min_periods=1).sum()
                    
                    # Per 90 metrics where appropriate
                    if metric in ['goals', 'assists', 'shots', 'key_passes']:
                        mins_last_n = player_df['minutes_played'].rolling(window=window_size, min_periods=1).sum().clip(lower=90)
                        player_form_df[f'{metric}_per90_last{window_size}'] = player_form_df[f'{metric}_last{window_size}'] * 90 / mins_last_n
        
        result_dfs.append(player_form_df)
    
    if not result_dfs:
        logger.warning("No player form data calculated")
        return pd.DataFrame()
    
    result_df = pd.concat(result_dfs)
    logger.info(f"Calculated player form for {len(result_dfs)} players, resulting in {len(result_df)} rows")
    
    return result_df
EOF

echo "Stopping containers to apply fix..."
docker compose down

echo "Starting containers with fixed functions..."
docker compose up -d

echo -e "\nPlayer form function fixes have been applied!"
echo "The calculate_player_form function has been added to features.py"
echo "The containers should now start correctly"

echo -e "\nTo check the app logs:"
echo "docker compose logs app"
echo -e "\nTo check the frontend logs:"
echo "docker compose logs frontend" 