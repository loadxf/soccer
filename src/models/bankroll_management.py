class KellyCriterionManager(BankrollManager):
    """
    Kelly Criterion bankroll manager.
    
    This strategy calculates bet sizes based on the Kelly Criterion,
    which optimizes bankroll growth rate by considering probability and odds.
    """
    
    def __init__(self, 
                initial_bankroll: float = 1000.0,
                name: str = "Kelly Criterion",
                transaction_log_path: Optional[str] = None,
                risk_level: str = "medium",
                reserve_percentage: float = 0.1,
                fraction: float = 0.5,
                min_probability: float = 0.1,
                max_kelly_percentage: float = 0.1,
                floor_bets: bool = True,
                max_odds: float = 10.0):
        """
        Initialize the Kelly Criterion manager.
        
        Args:
            initial_bankroll: Starting bankroll amount
            name: Name of the bankroll manager
            transaction_log_path: Path to save transaction logs
            risk_level: Risk level ('low', 'medium', 'high')
            reserve_percentage: Percentage of bankroll to keep in reserve
            fraction: Fraction of full Kelly to use (0-1)
            min_probability: Minimum probability to consider for Kelly calculation
            max_kelly_percentage: Maximum percentage of bankroll for any bet
            floor_bets: Whether to use a minimum bet size for very small Kelly values
            max_odds: Maximum odds to consider for Kelly calculation
        """
        super().__init__(
            initial_bankroll=initial_bankroll,
            name=name,
            transaction_log_path=transaction_log_path,
            risk_level=risk_level,
            reserve_percentage=reserve_percentage
        )
        
        self.fraction = min(1.0, max(0.1, fraction))
        self.min_probability = min_probability
        self.max_kelly_percentage = min(max_kelly_percentage, self.max_bet_percentage)
        self.floor_bets = floor_bets
        self.max_odds = max_odds
        
        # Performance tracking
        self.kelly_performance = {
            'highest_kelly': 0.0,
            'average_kelly': 0.0,
            'total_kelly_bets': 0
        }
    
    def calculate_kelly_fraction(self, 
                              probability: float, 
                              odds: float) -> float:
        """
        Calculate the Kelly Criterion fraction.
        
        Args:
            probability: Probability of winning (0-1)
            odds: Decimal odds
            
        Returns:
            float: Fraction of bankroll to bet (0-1)
        """
        # Kelly formula: f* = (bp - q) / b
        # where b = odds - 1, p = probability of win, q = probability of loss
        if probability < self.min_probability:
            return 0.0
            
        if odds > self.max_odds:
            return 0.0
            
        b = odds - 1  # Decimal odds conversion to b
        p = probability
        q = 1 - p
        
        # Calculate full Kelly stake
        if b > 0:
            kelly = (b * p - q) / b
        else:
            kelly = 0.0
            
        # Cap at maximum and floor at 0
        kelly = max(0.0, min(self.max_kelly_percentage, kelly))
        
        # Apply fraction (half-Kelly, quarter-Kelly, etc.)
        return kelly * self.fraction
    
    def calculate_recommended_stake(self, 
                                bet_data: Dict[str, Any]) -> float:
        """
        Calculate the recommended stake using Kelly Criterion.
        
        Args:
            bet_data: Dictionary with bet information
                Required keys: 'probability', 'odds'
                Optional keys: 'confidence'
            
        Returns:
            float: Recommended stake
        """
        if 'probability' not in bet_data or 'odds' not in bet_data:
            logger.warning("Missing required keys for Kelly calculation")
            return 0.0
            
        probability = bet_data['probability']
        odds = bet_data['odds']
        
        # Adjust probability by confidence if available
        if 'confidence' in bet_data:
            confidence = bet_data['confidence']
            # Lower probability for low confidence predictions
            adjusted_probability = probability * confidence
        else:
            adjusted_probability = probability
            
        # Calculate Kelly fraction
        kelly_fraction = self.calculate_kelly_fraction(adjusted_probability, odds)
        
        # Update performance tracking
        self.kelly_performance['highest_kelly'] = max(self.kelly_performance['highest_kelly'], kelly_fraction)
        self.kelly_performance['total_kelly_bets'] += 1
        
        # Running average
        old_avg = self.kelly_performance['average_kelly']
        n = self.kelly_performance['total_kelly_bets']
        self.kelly_performance['average_kelly'] = old_avg + (kelly_fraction - old_avg) / n
        
        # Calculate stake
        available_bankroll = self.current_bankroll * (1 - self.reserve_percentage)
        stake = available_bankroll * kelly_fraction
        
        # Floor small bets if enabled
        if self.floor_bets and stake > 0 and stake < available_bankroll * 0.001:
            stake = available_bankroll * 0.001
            
        return stake
    
    def place_bet(self, 
                amount: float, 
                match_id: Optional[Union[str, int]] = None,
                strategy_name: Optional[str] = None,
                bet_description: Optional[str] = None,
                bet_id: Optional[str] = None,
                extra: Optional[Dict[str, Any]] = None) -> bool:
        """
        Place a bet, deducting the stake from the bankroll.
        
        Args:
            amount: Bet amount
            match_id: Optional match ID
            strategy_name: Optional strategy name
            bet_description: Optional bet description
            bet_id: Optional bet identifier
            extra: Optional additional information
            
        Returns:
            bool: Whether the bet was placed successfully
        """
        # Store probability and odds in extra for performance analysis
        if extra and 'probability' in extra and 'odds' in extra:
            if 'kelly_data' not in extra:
                extra['kelly_data'] = {}
                
            probability = extra['probability']
            odds = extra['odds']
            kelly_fraction = self.calculate_kelly_fraction(probability, odds)
            
            extra['kelly_data']['fraction'] = kelly_fraction
            extra['kelly_data']['recommended'] = self.current_bankroll * kelly_fraction
            
        return super().place_bet(
            amount=amount,
            match_id=match_id,
            strategy_name=strategy_name,
            bet_description=bet_description,
            bet_id=bet_id,
            extra=extra
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of bankroll performance.
        
        Returns:
            Dict[str, Any]: Performance summary statistics
        """
        summary = super().get_performance_summary()
        
        # Add Kelly-specific metrics
        summary['kelly_fraction'] = self.fraction
        summary['highest_kelly'] = self.kelly_performance['highest_kelly']
        summary['average_kelly'] = self.kelly_performance['average_kelly']
        summary['total_kelly_bets'] = self.kelly_performance['total_kelly_bets']
        
        # Calculate efficiency vs. optimal Kelly
        if summary['total_bets'] > 0:
            # Theoretical growth rate with optimal Kelly
            # g = (1 + b*f)^p * (1 - f)^q - 1
            # where f is the fraction of bankroll bet, b is odds-1, p is win probability, q is loss probability
            
            # Since we're using fractional Kelly, efficiency is approximately the fraction
            summary['kelly_efficiency'] = self.fraction
            
            # Compare actual performance to theoretical growth
            # This is a simplified approximation
            growth_rate = (summary['current_bankroll'] / summary['initial_bankroll']) - 1
            bets_per_year = 500  # Assumption
            years = summary['total_bets'] / bets_per_year
            if years > 0:
                annual_growth = (1 + growth_rate) ** (1 / years) - 1
                summary['annual_growth_rate'] = annual_growth
            else:
                summary['annual_growth_rate'] = 0
        else:
            summary['kelly_efficiency'] = 0
            summary['annual_growth_rate'] = 0
            
        return summary 

class StopLossManager(BankrollManager):
    """
    Stop Loss bankroll manager.
    
    This strategy implements advanced stop loss and take profit rules
    to protect the bankroll and lock in profits.
    """
    
    def __init__(self, 
                initial_bankroll: float = 1000.0,
                name: str = "Stop Loss Manager",
                transaction_log_path: Optional[str] = None,
                risk_level: str = "medium",
                reserve_percentage: float = 0.1,
                base_bet_percentage: float = 0.02,
                daily_stop_loss_percentage: float = 0.05,
                weekly_stop_loss_percentage: float = 0.1,
                monthly_stop_loss_percentage: float = 0.2,
                take_profit_percentage: float = 0.3,
                losing_streak_stop: int = 5,
                progressive_stop_loss: bool = True,
                reset_frequency: str = "monthly"):
        """
        Initialize the Stop Loss manager.
        
        Args:
            initial_bankroll: Starting bankroll amount
            name: Name of the bankroll manager
            transaction_log_path: Path to save transaction logs
            risk_level: Risk level ('low', 'medium', 'high')
            reserve_percentage: Percentage of bankroll to keep in reserve
            base_bet_percentage: Base percentage of bankroll to bet
            daily_stop_loss_percentage: Percentage loss to trigger daily stop
            weekly_stop_loss_percentage: Percentage loss to trigger weekly stop
            monthly_stop_loss_percentage: Percentage loss to trigger monthly stop
            take_profit_percentage: Percentage gain to trigger take profit
            losing_streak_stop: Number of consecutive losses to trigger stop
            progressive_stop_loss: Whether to reduce bet size after losses
            reset_frequency: When to reset stop loss counters ('daily', 'weekly', 'monthly')
        """
        super().__init__(
            initial_bankroll=initial_bankroll,
            name=name,
            transaction_log_path=transaction_log_path,
            risk_level=risk_level,
            reserve_percentage=reserve_percentage
        )
        
        self.base_bet_percentage = min(self.max_bet_percentage, max(0.001, base_bet_percentage))
        self.daily_stop_loss_percentage = daily_stop_loss_percentage
        self.weekly_stop_loss_percentage = weekly_stop_loss_percentage
        self.monthly_stop_loss_percentage = monthly_stop_loss_percentage
        self.take_profit_percentage = take_profit_percentage
        self.losing_streak_stop = losing_streak_stop
        self.progressive_stop_loss = progressive_stop_loss
        self.reset_frequency = reset_frequency
        
        # Stop loss tracking
        self.current_streak = 0  # Positive for wins, negative for losses
        self.daily_loss = 0.0
        self.weekly_loss = 0.0
        self.monthly_loss = 0.0
        self.daily_profit = 0.0
        self.weekly_profit = 0.0
        self.monthly_profit = 0.0
        
        # Track dates for reset
        self.last_daily_reset = datetime.now().date()
        self.last_weekly_reset = datetime.now().date()
        self.last_monthly_reset = datetime.now().date()
        
        # Stop status
        self.stop_triggered = False
        self.stop_reason = None
        self.stopped_until = None
    
    def calculate_recommended_stake(self, 
                                bet_data: Dict[str, Any]) -> float:
        """
        Calculate the recommended stake with stop loss adjustments.
        
        Args:
            bet_data: Dictionary with bet information
            
        Returns:
            float: Recommended stake
        """
        # If stop loss is triggered, return 0
        if self.is_stop_loss_triggered():
            return 0.0
        
        # Basic stake calculation
        available_bankroll = self.current_bankroll * (1 - self.reserve_percentage)
        base_stake = available_bankroll * self.base_bet_percentage
        
        # If progressive stop loss is enabled, adjust stake based on current streak
        if self.progressive_stop_loss and self.current_streak < 0:
            # Reduce stake for each consecutive loss
            loss_factor = max(0.5, 1.0 + (self.current_streak * 0.1))
            base_stake *= loss_factor
        
        # Adjust based on risk level and bet edge/value
        multiplier = 1.0
        
        if 'edge' in bet_data and 'odds' in bet_data:
            edge = bet_data['edge']
            odds = bet_data['odds']
            
            # Adjust for value
            if edge > 0:
                # Scale up for higher edge bets
                value_mult = 1.0 + min(1.0, edge * 5)
                multiplier *= value_mult
                
                # But reduce for very high odds (risky bets)
                if odds > 3.0:
                    odds_discount = max(0.5, 1.0 - (odds - 3.0) / 10.0)
                    multiplier *= odds_discount
        
        # Apply bet size adjustments based on recent performance
        if self.monthly_loss > 0:
            # Scale down if approaching monthly stop loss
            monthly_loss_ratio = self.monthly_loss / (self.initial_bankroll * self.monthly_stop_loss_percentage)
            if monthly_loss_ratio > 0.5:
                loss_adjustment = 1.0 - min(0.5, (monthly_loss_ratio - 0.5) * 0.5)
                multiplier *= loss_adjustment
        
        final_stake = base_stake * multiplier
        
        # Ensure stake is within limits
        max_stake = available_bankroll * self.max_bet_percentage
        final_stake = min(max_stake, final_stake)
        
        # Ensure stake is not too small
        min_stake = available_bankroll * 0.001  # 0.1% minimum
        final_stake = max(min_stake, final_stake) if final_stake > 0 else 0.0
        
        return final_stake
    
    def update_streak(self, outcome: str) -> None:
        """
        Update the current winning/losing streak.
        
        Args:
            outcome: Bet outcome ('win', 'loss', 'push')
        """
        if outcome == 'win':
            if self.current_streak >= 0:
                # Continue winning streak
                self.current_streak += 1
            else:
                # Start new winning streak
                self.current_streak = 1
        elif outcome == 'loss':
            if self.current_streak <= 0:
                # Continue losing streak
                self.current_streak -= 1
            else:
                # Start new losing streak
                self.current_streak = -1
        # Push doesn't affect streak
    
    def update_stop_loss_tracking(self, 
                              amount: float, 
                              is_loss: bool) -> None:
        """
        Update stop loss tracking values.
        
        Args:
            amount: Transaction amount
            is_loss: Whether this is a loss
        """
        # Reset tracking if needed
        self._check_reset_dates()
        
        if is_loss:
            self.daily_loss += amount
            self.weekly_loss += amount
            self.monthly_loss += amount
        else:
            self.daily_profit += amount
            self.weekly_profit += amount
            self.monthly_profit += amount
        
        # Check if any stop loss is triggered
        self._check_stop_loss_triggers()
    
    def _check_reset_dates(self) -> None:
        """Check if tracking periods need to be reset based on current date."""
        today = datetime.now().date()
        
        # Daily reset
        if today > self.last_daily_reset:
            self.daily_loss = 0.0
            self.daily_profit = 0.0
            self.last_daily_reset = today
            
            # If stop was triggered for a day, check if it can be lifted
            if self.stop_triggered and self.stopped_until and self.stopped_until <= today:
                self.stop_triggered = False
                self.stop_reason = None
                self.stopped_until = None
        
        # Weekly reset (check if we've moved to a new week)
        if (today - self.last_weekly_reset).days >= 7:
            self.weekly_loss = 0.0
            self.weekly_profit = 0.0
            self.last_weekly_reset = today
            
            # Reset streak if configured for weekly
            if self.reset_frequency == 'weekly':
                self.current_streak = 0
        
        # Monthly reset (check if month has changed)
        if today.month != self.last_monthly_reset.month or today.year != self.last_monthly_reset.year:
            self.monthly_loss = 0.0
            self.monthly_profit = 0.0
            self.last_monthly_reset = today
            
            # Reset streak if configured for monthly
            if self.reset_frequency == 'monthly':
                self.current_streak = 0
    
    def _check_stop_loss_triggers(self) -> None:
        """Check if any stop loss conditions are triggered."""
        today = datetime.now().date()
        
        # Already stopped
        if self.stop_triggered:
            return
            
        # Check daily stop loss
        if self.daily_loss >= self.initial_bankroll * self.daily_stop_loss_percentage:
            self.stop_triggered = True
            self.stop_reason = "Daily stop loss reached"
            self.stopped_until = today + timedelta(days=1)
            logger.warning(f"Daily stop loss triggered: {self.daily_loss:.2f}")
            return
            
        # Check weekly stop loss
        if self.weekly_loss >= self.initial_bankroll * self.weekly_stop_loss_percentage:
            self.stop_triggered = True
            self.stop_reason = "Weekly stop loss reached"
            self.stopped_until = today + timedelta(days=7 - today.weekday())  # Until end of week
            logger.warning(f"Weekly stop loss triggered: {self.weekly_loss:.2f}")
            return
            
        # Check monthly stop loss
        if self.monthly_loss >= self.initial_bankroll * self.monthly_stop_loss_percentage:
            self.stop_triggered = True
            self.stop_reason = "Monthly stop loss reached"
            
            # Calculate days until end of month
            next_month = today.replace(day=28) + timedelta(days=4)
            last_day = next_month - timedelta(days=next_month.day)
            days_remaining = (last_day - today).days + 1
            
            self.stopped_until = today + timedelta(days=days_remaining)
            logger.warning(f"Monthly stop loss triggered: {self.monthly_loss:.2f}")
            return
            
        # Check losing streak
        if abs(self.current_streak) >= self.losing_streak_stop and self.current_streak < 0:
            self.stop_triggered = True
            self.stop_reason = f"Losing streak of {abs(self.current_streak)} reached"
            self.stopped_until = today + timedelta(days=1)  # Stop for a day
            logger.warning(f"Losing streak stop triggered: {abs(self.current_streak)} consecutive losses")
            return
            
        # Check take profit (optional)
        if self.take_profit_percentage > 0:
            # If we've made significant profit for the period, consider stopping
            if self.daily_profit >= self.initial_bankroll * self.take_profit_percentage:
                self.stop_triggered = True
                self.stop_reason = "Daily profit target reached"
                self.stopped_until = today + timedelta(days=1)
                logger.info(f"Take profit triggered: {self.daily_profit:.2f}")
                return
    
    def is_stop_loss_triggered(self) -> bool:
        """
        Check if betting should be stopped due to stop loss.
        
        Returns:
            bool: True if betting should be stopped
        """
        # Reset dates first to clear any expired stops
        self._check_reset_dates()
        return self.stop_triggered
    
    def get_stop_status(self) -> Dict[str, Any]:
        """
        Get the current stop loss status.
        
        Returns:
            Dict[str, Any]: Stop loss status information
        """
        # Reset dates first to clear any expired stops
        self._check_reset_dates()
        
        return {
            'is_stopped': self.stop_triggered,
            'reason': self.stop_reason,
            'stopped_until': self.stopped_until.isoformat() if self.stopped_until else None,
            'current_streak': self.current_streak,
            'daily_loss': self.daily_loss,
            'weekly_loss': self.weekly_loss,
            'monthly_loss': self.monthly_loss,
            'daily_profit': self.daily_profit,
            'weekly_profit': self.weekly_profit,
            'monthly_profit': self.monthly_profit,
            'daily_loss_limit': self.initial_bankroll * self.daily_stop_loss_percentage,
            'weekly_loss_limit': self.initial_bankroll * self.weekly_stop_loss_percentage,
            'monthly_loss_limit': self.initial_bankroll * self.monthly_stop_loss_percentage
        }
    
    def place_bet(self, 
                amount: float, 
                match_id: Optional[Union[str, int]] = None,
                strategy_name: Optional[str] = None,
                bet_description: Optional[str] = None,
                bet_id: Optional[str] = None,
                extra: Optional[Dict[str, Any]] = None) -> bool:
        """
        Place a bet, checking stop loss conditions first.
        
        Args:
            amount: Bet amount
            match_id: Optional match ID
            strategy_name: Optional strategy name
            bet_description: Optional bet description
            bet_id: Optional bet identifier
            extra: Optional additional information
            
        Returns:
            bool: Whether the bet was placed successfully
        """
        # Check if stop loss is triggered
        if self.is_stop_loss_triggered():
            logger.warning(f"Bet rejected due to stop loss: {self.stop_reason}")
            return False
        
        # Add stop loss data to extra
        if extra is None:
            extra = {}
            
        extra['stop_loss_data'] = {
            'current_streak': self.current_streak,
            'daily_loss': self.daily_loss,
            'weekly_loss': self.weekly_loss,
            'monthly_loss': self.monthly_loss
        }
        
        return super().place_bet(
            amount=amount,
            match_id=match_id,
            strategy_name=strategy_name,
            bet_description=bet_description,
            bet_id=bet_id,
            extra=extra
        )
    
    def settle_bet(self, 
                bet_id: str, 
                outcome: str, 
                win_amount: Optional[float] = None) -> bool:
        """
        Settle a bet and update stop loss tracking.
        
        Args:
            bet_id: Bet identifier
            outcome: Result ('win', 'loss', 'push')
            win_amount: Amount won (required for win)
            
        Returns:
            bool: Whether the bet was settled successfully
        """
        # Find the bet
        bet_transaction = None
        for t in self.transactions:
            if t.transaction_type == 'bet' and t.bet_id == bet_id:
                bet_transaction = t
                break
                
        if not bet_transaction:
            logger.warning(f"Bet {bet_id} not found")
            return False
            
        # Handle settlement
        result = super().settle_bet(bet_id, outcome, win_amount)
        
        if result:
            # Update streak
            self.update_streak(outcome)
            
            # Update stop loss tracking
            if outcome == 'loss':
                self.update_stop_loss_tracking(bet_transaction.amount, True)
            elif outcome == 'win' and win_amount:
                # Track net profit (win amount - stake)
                net_profit = win_amount - bet_transaction.amount
                self.update_stop_loss_tracking(net_profit, False)
                
        return result
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of bankroll performance.
        
        Returns:
            Dict[str, Any]: Performance summary statistics
        """
        summary = super().get_performance_summary()
        
        # Add stop loss specific metrics
        summary['stop_loss_status'] = self.get_stop_status()
        summary['base_bet_percentage'] = self.base_bet_percentage
        summary['current_streak'] = self.current_streak
        
        # Calculate percentage of time stopped
        if len(self.transactions) > 0:
            # Get all transactions with stop loss data
            stop_transactions = [t for t in self.transactions if t.extra and 'stop_triggered' in t.extra]
            stopped_count = sum(1 for t in stop_transactions if t.extra['stop_triggered'])
            
            if stop_transactions:
                summary['percent_time_stopped'] = (stopped_count / len(stop_transactions)) * 100
            else:
                summary['percent_time_stopped'] = 0
        else:
            summary['percent_time_stopped'] = 0
            
        return summary 

class BankrollTracker:
    """
    Tracks and analyzes bankroll performance across multiple betting strategies.
    
    This class aggregates data from multiple bankroll managers, providing 
    comprehensive analytics, performance metrics, and visualization capabilities.
    """
    
    def __init__(self, 
                name: str = "Master Bankroll Tracker",
                storage_path: Optional[str] = None,
                auto_save: bool = True,
                save_frequency: int = 100):
        """
        Initialize the BankrollTracker.
        
        Args:
            name: Name of the bankroll tracker
            storage_path: Path to save tracker data
            auto_save: Whether to automatically save data
            save_frequency: How often to auto-save (in transactions)
        """
        self.name = name
        self.storage_path = storage_path
        self.auto_save = auto_save
        self.save_frequency = save_frequency
        
        # Registered bankroll managers
        self.managers: Dict[str, BankrollManager] = {}
        
        # Performance tracking
        self.last_update = datetime.now()
        self.transaction_counter = 0
        
        # Analytics data
        self.daily_snapshots: List[Dict[str, Any]] = []
        self.roi_history: List[Dict[str, float]] = []
        self.win_loss_history: List[Dict[str, int]] = []
        self.bankroll_history: List[Dict[str, float]] = []
        
        # Performance metrics
        self._metrics_cache = {}
        self._metrics_last_update = datetime.now() - timedelta(days=1)  # Force initial update
    
    def register_manager(self, manager: BankrollManager) -> None:
        """
        Register a bankroll manager to be tracked.
        
        Args:
            manager: BankrollManager instance
        """
        if manager.name in self.managers:
            logger.warning(f"Manager with name '{manager.name}' already registered. Overwriting.")
            
        self.managers[manager.name] = manager
        logger.info(f"Registered bankroll manager: {manager.name}")
        
        # Take initial snapshot
        self._take_snapshot()
    
    def unregister_manager(self, manager_name: str) -> bool:
        """
        Unregister a bankroll manager.
        
        Args:
            manager_name: Name of the manager to unregister
            
        Returns:
            bool: Whether the manager was successfully unregistered
        """
        if manager_name in self.managers:
            del self.managers[manager_name]
            logger.info(f"Unregistered bankroll manager: {manager_name}")
            return True
        else:
            logger.warning(f"Manager '{manager_name}' not found")
            return False
    
    def _take_snapshot(self) -> Dict[str, Any]:
        """
        Take a snapshot of current bankroll state across all managers.
        
        Returns:
            Dict[str, Any]: Snapshot data
        """
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'managers': {},
            'total_bankroll': 0.0,
            'total_bets_placed': 0,
            'total_bets_won': 0,
            'total_bets_lost': 0,
            'total_bets_pushed': 0
        }
        
        for name, manager in self.managers.items():
            # Get performance summary
            perf = manager.get_performance_summary()
            
            # Extract key metrics
            manager_data = {
                'bankroll': manager.current_bankroll,
                'initial_bankroll': manager.initial_bankroll,
                'profit_loss': manager.current_bankroll - manager.initial_bankroll,
                'roi': (manager.current_bankroll / manager.initial_bankroll) - 1 if manager.initial_bankroll > 0 else 0,
                'win_rate': perf.get('win_rate', 0),
                'bets_placed': perf.get('bets_placed', 0),
                'bets_won': perf.get('bets_won', 0),
                'bets_lost': perf.get('bets_lost', 0),
                'bets_pushed': perf.get('bets_pushed', 0)
            }
            
            # Add to snapshot
            snapshot['managers'][name] = manager_data
            
            # Update totals
            snapshot['total_bankroll'] += manager_data['bankroll']
            snapshot['total_bets_placed'] += manager_data['bets_placed']
            snapshot['total_bets_won'] += manager_data['bets_won']
            snapshot['total_bets_lost'] += manager_data['bets_lost']
            snapshot['total_bets_pushed'] += manager_data['bets_pushed']
        
        # Update history
        self._update_history(snapshot)
        
        return snapshot
    
    def _update_history(self, snapshot: Dict[str, Any]) -> None:
        """
        Update historical tracking data with new snapshot.
        
        Args:
            snapshot: Current bankroll snapshot
        """
        # Extract date for daily tracking
        snapshot_date = datetime.fromisoformat(snapshot['timestamp']).date()
        
        # Check if we already have a snapshot for today
        today_snapshot_exists = any(
            datetime.fromisoformat(s['timestamp']).date() == snapshot_date 
            for s in self.daily_snapshots
        )
        
        # If no snapshot for today, add to daily history
        if not today_snapshot_exists:
            self.daily_snapshots.append(snapshot)
            
            # Trim history if too large (keep last 365 days)
            if len(self.daily_snapshots) > 365:
                self.daily_snapshots = self.daily_snapshots[-365:]
        
        # Update bankroll history
        bankroll_entry = {'date': snapshot_date.isoformat(), 'total': snapshot['total_bankroll']}
        for name, data in snapshot['managers'].items():
            bankroll_entry[name] = data['bankroll']
        
        self.bankroll_history.append(bankroll_entry)
        
        # Update ROI history
        roi_entry = {'date': snapshot_date.isoformat()}
        for name, data in snapshot['managers'].items():
            roi_entry[name] = data['roi']
        
        self.roi_history.append(roi_entry)
        
        # Update win/loss history
        wl_entry = {
            'date': snapshot_date.isoformat(),
            'total_bets': snapshot['total_bets_placed'],
            'total_wins': snapshot['total_bets_won'],
            'total_losses': snapshot['total_bets_lost']
        }
        
        self.win_loss_history.append(wl_entry)
        
        # Maybe save data
        self.transaction_counter += 1
        if self.auto_save and self.transaction_counter % self.save_frequency == 0:
            self.save_data()
    
    def update(self) -> Dict[str, Any]:
        """
        Update tracker with latest data from all managers.
        
        Returns:
            Dict[str, Any]: Current snapshot
        """
        snapshot = self._take_snapshot()
        self.last_update = datetime.now()
        
        # Clear metrics cache
        self._metrics_cache = {}
        
        return snapshot
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get the current state of all bankroll managers.
        
        Returns:
            Dict[str, Any]: Current state data
        """
        return self._take_snapshot()
    
    def get_manager_performance(self, 
                               manager_name: str) -> Dict[str, Any]:
        """
        Get detailed performance data for a specific manager.
        
        Args:
            manager_name: Name of the manager
            
        Returns:
            Dict[str, Any]: Performance data
        """
        if manager_name not in self.managers:
            logger.warning(f"Manager '{manager_name}' not found")
            return {}
            
        return self.managers[manager_name].get_performance_summary()
    
    def get_overall_performance(self) -> Dict[str, Any]:
        """
        Get overall performance across all managers.
        
        Returns:
            Dict[str, Any]: Overall performance metrics
        """
        # Check if cached metrics are recent enough
        if (datetime.now() - self._metrics_last_update).total_seconds() < 3600 and self._metrics_cache:
            return self._metrics_cache
            
        # Get current snapshot
        snapshot = self._take_snapshot()
        
        # Calculate overall metrics
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'total_bankroll': snapshot['total_bankroll'],
            'initial_bankroll': sum(m.initial_bankroll for m in self.managers.values()),
            'profit_loss': snapshot['total_bankroll'] - sum(m.initial_bankroll for m in self.managers.values()),
            'total_bets_placed': snapshot['total_bets_placed'],
            'total_bets_won': snapshot['total_bets_won'],
            'total_bets_lost': snapshot['total_bets_lost'],
            'total_bets_pushed': snapshot['total_bets_pushed'],
            'managers_count': len(self.managers),
            'best_performing_manager': None,
            'worst_performing_manager': None,
            'manager_data': {}
        }
        
        # Calculate overall ROI
        if metrics['initial_bankroll'] > 0:
            metrics['overall_roi'] = (metrics['total_bankroll'] / metrics['initial_bankroll']) - 1
        else:
            metrics['overall_roi'] = 0
            
        # Calculate win rate
        if metrics['total_bets_placed'] > 0:
            metrics['win_rate'] = metrics['total_bets_won'] / metrics['total_bets_placed']
        else:
            metrics['win_rate'] = 0
            
        # Find best and worst managers
        best_roi = -float('inf')
        worst_roi = float('inf')
        
        for name, data in snapshot['managers'].items():
            metrics['manager_data'][name] = {
                'bankroll': data['bankroll'],
                'roi': data['roi'],
                'win_rate': data['win_rate']
            }
            
            if data['roi'] > best_roi and data['bets_placed'] > 10:
                best_roi = data['roi']
                metrics['best_performing_manager'] = name
                
            if data['roi'] < worst_roi and data['bets_placed'] > 10:
                worst_roi = data['roi']
                metrics['worst_performing_manager'] = name
        
        # Add historical performance data
        if len(self.bankroll_history) >= 2:
            # Calculate recent growth
            latest = self.bankroll_history[-1]
            week_ago_idx = max(0, len(self.bankroll_history) - 8)
            month_ago_idx = max(0, len(self.bankroll_history) - 31)
            
            week_ago = self.bankroll_history[week_ago_idx]
            month_ago = self.bankroll_history[month_ago_idx]
            
            metrics['weekly_growth'] = latest['total'] - week_ago['total']
            metrics['monthly_growth'] = latest['total'] - month_ago['total']
            
            if week_ago['total'] > 0:
                metrics['weekly_growth_percentage'] = (metrics['weekly_growth'] / week_ago['total']) * 100
            else:
                metrics['weekly_growth_percentage'] = 0
                
            if month_ago['total'] > 0:
                metrics['monthly_growth_percentage'] = (metrics['monthly_growth'] / month_ago['total']) * 100
            else:
                metrics['monthly_growth_percentage'] = 0
        
        # Cache metrics
        self._metrics_cache = metrics
        self._metrics_last_update = datetime.now()
        
        return metrics
    
    def get_historical_data(self, 
                          period: str = 'all',
                          manager_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get historical performance data.
        
        Args:
            period: Time period ('week', 'month', 'year', 'all')
            manager_name: Optional specific manager to report on
            
        Returns:
            Dict[str, Any]: Historical data
        """
        # Determine cutoff date based on period
        now = datetime.now().date()
        if period == 'week':
            cutoff = (now - timedelta(days=7)).isoformat()
        elif period == 'month':
            cutoff = (now - timedelta(days=30)).isoformat()
        elif period == 'year':
            cutoff = (now - timedelta(days=365)).isoformat()
        else:  # 'all'
            cutoff = '0001-01-01'  # Beginning of time
            
        # Filter history by date
        filtered_bankroll = [
            entry for entry in self.bankroll_history 
            if entry['date'] >= cutoff
        ]
        
        filtered_roi = [
            entry for entry in self.roi_history 
            if entry['date'] >= cutoff
        ]
        
        filtered_win_loss = [
            entry for entry in self.win_loss_history 
            if entry['date'] >= cutoff
        ]
        
        # If specific manager requested, filter data
        if manager_name:
            if manager_name not in self.managers:
                logger.warning(f"Manager '{manager_name}' not found")
                return {}
                
            # Filter bankroll data for just this manager
            filtered_bankroll = [
                {k: v for k, v in entry.items() if k == 'date' or k == manager_name}
                for entry in filtered_bankroll
            ]
            
            # Filter ROI data for just this manager
            filtered_roi = [
                {k: v for k, v in entry.items() if k == 'date' or k == manager_name}
                for entry in filtered_roi
            ]
        
        # Build result
        result = {
            'period': period,
            'manager': manager_name,
            'bankroll_history': filtered_bankroll,
            'roi_history': filtered_roi,
            'win_loss_history': filtered_win_loss
        }
        
        # Calculate summary stats
        if filtered_bankroll:
            first = filtered_bankroll[0]
            last = filtered_bankroll[-1]
            
            if manager_name:
                start_value = first.get(manager_name, 0)
                end_value = last.get(manager_name, 0)
            else:
                start_value = first.get('total', 0)
                end_value = last.get('total', 0)
                
            result['starting_value'] = start_value
            result['ending_value'] = end_value
            result['net_change'] = end_value - start_value
            
            if start_value > 0:
                result['percentage_change'] = (result['net_change'] / start_value) * 100
            else:
                result['percentage_change'] = 0
        
        return result
    
    def save_data(self) -> bool:
        """
        Save tracker data to storage path.
        
        Returns:
            bool: Whether save was successful
        """
        if not self.storage_path:
            logger.warning("No storage path specified for bankroll tracker")
            return False
            
        try:
            # Prepare data for saving
            data = {
                'name': self.name,
                'last_update': self.last_update.isoformat(),
                'daily_snapshots': self.daily_snapshots,
                'roi_history': self.roi_history,
                'win_loss_history': self.win_loss_history,
                'bankroll_history': self.bankroll_history
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            
            # Save to file
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Bankroll tracker data saved to {self.storage_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving bankroll tracker data: {str(e)}")
            return False
    
    def load_data(self) -> bool:
        """
        Load tracker data from storage path.
        
        Returns:
            bool: Whether load was successful
        """
        if not self.storage_path or not os.path.exists(self.storage_path):
            logger.warning(f"Storage path {self.storage_path} not found")
            return False
            
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                
            # Update properties
            self.name = data.get('name', self.name)
            self.last_update = datetime.fromisoformat(data.get('last_update', datetime.now().isoformat()))
            self.daily_snapshots = data.get('daily_snapshots', [])
            self.roi_history = data.get('roi_history', [])
            self.win_loss_history = data.get('win_loss_history', [])
            self.bankroll_history = data.get('bankroll_history', [])
            
            logger.info(f"Bankroll tracker data loaded from {self.storage_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading bankroll tracker data: {str(e)}")
            return False
    
    def generate_report(self, 
                       period: str = 'month',
                       include_managers: bool = True) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.
        
        Args:
            period: Time period ('week', 'month', 'year', 'all')
            include_managers: Whether to include individual manager stats
            
        Returns:
            Dict[str, Any]: Performance report
        """
        # Get current performance
        performance = self.get_overall_performance()
        
        # Get historical data
        historical = self.get_historical_data(period=period)
        
        # Build report
        report = {
            'timestamp': datetime.now().isoformat(),
            'period': period,
            'bankroll': {
                'current': performance['total_bankroll'],
                'initial': performance['initial_bankroll'],
                'profit_loss': performance['profit_loss'],
                'roi': performance['overall_roi'] * 100,  # Convert to percentage
                'change_over_period': historical.get('percentage_change', 0)
            },
            'betting': {
                'total_bets': performance['total_bets_placed'],
                'wins': performance['total_bets_won'],
                'losses': performance['total_bets_lost'],
                'pushes': performance['total_bets_pushed'],
                'win_rate': performance['win_rate'] * 100  # Convert to percentage
            },
            'managers_count': performance['managers_count'],
            'best_manager': performance['best_performing_manager'],
            'worst_manager': performance['worst_performing_manager']
        }
        
        # Add manager details if requested
        if include_managers:
            report['managers'] = {}
            for name, manager in self.managers.items():
                manager_perf = manager.get_performance_summary()
                manager_hist = self.get_historical_data(period=period, manager_name=name)
                
                report['managers'][name] = {
                    'bankroll': manager.current_bankroll,
                    'initial': manager.initial_bankroll,
                    'profit_loss': manager.current_bankroll - manager.initial_bankroll,
                    'roi': manager_perf.get('roi', 0) * 100,  # Convert to percentage
                    'win_rate': manager_perf.get('win_rate', 0) * 100,  # Convert to percentage
                    'bets_placed': manager_perf.get('bets_placed', 0),
                    'period_change': manager_hist.get('percentage_change', 0)
                }
                
                # Add additional metrics if available
                if 'yield' in manager_perf:
                    report['managers'][name]['yield'] = manager_perf['yield'] * 100
                    
                if 'average_odds' in manager_perf:
                    report['managers'][name]['average_odds'] = manager_perf['average_odds']
        
        return report
    
    def plot_bankroll_history(self, 
                            period: str = 'all',
                            manager_name: Optional[str] = None,
                            save_path: Optional[str] = None) -> None:
        """
        Plot bankroll history over time.
        
        Args:
            period: Time period ('week', 'month', 'year', 'all')
            manager_name: Optional specific manager to plot
            save_path: Optional path to save the plot
            
        Returns:
            None
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from matplotlib.ticker import FuncFormatter
            
            # Get historical data
            historical = self.get_historical_data(period=period, manager_name=manager_name)
            bankroll_data = historical['bankroll_history']
            
            if not bankroll_data:
                logger.warning("No bankroll data available to plot")
                return
                
            # Convert dates to datetime
            dates = [datetime.fromisoformat(entry['date']) for entry in bankroll_data]
            
            # Create figure
            plt.figure(figsize=(12, 6))
            
            # Plot total or specific manager
            if manager_name:
                values = [entry.get(manager_name, 0) for entry in bankroll_data]
                plt.plot(dates, values, linewidth=2, label=manager_name)
            else:
                values = [entry.get('total', 0) for entry in bankroll_data]
                plt.plot(dates, values, linewidth=2, label='Total Bankroll')
                
                # Plot individual managers if there are fewer than 5
                if len(self.managers) < 5:
                    for name in self.managers.keys():
                        manager_values = [entry.get(name, 0) for entry in bankroll_data]
                        plt.plot(dates, manager_values, linestyle='--', linewidth=1, label=name)
            
            # Format date axis
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
            
            # Format y-axis to display currency
            def currency_formatter(x, pos):
                return f'${x:.0f}'
                
            plt.gca().yaxis.set_major_formatter(FuncFormatter(currency_formatter))
            
            # Add labels and title
            plt.xlabel('Date')
            plt.ylabel('Bankroll Amount')
            
            if manager_name:
                plt.title(f'Bankroll History for {manager_name} ({period})')
            else:
                plt.title(f'Overall Bankroll History ({period})')
                
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Rotate date labels
            plt.gcf().autofmt_xdate()
            
            # Save or show
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Bankroll history plot saved to {save_path}")
            else:
                plt.tight_layout()
                plt.show()
                
        except ImportError:
            logger.warning("Matplotlib not available. Cannot generate plot.")
            
        except Exception as e:
            logger.error(f"Error generating bankroll history plot: {str(e)}")
            
    def plot_roi_comparison(self, 
                          period: str = 'all',
                          save_path: Optional[str] = None) -> None:
        """
        Plot ROI comparison between managers.
        
        Args:
            period: Time period ('week', 'month', 'year', 'all')
            save_path: Optional path to save the plot
            
        Returns:
            None
        """
        if len(self.managers) < 2:
            logger.warning("Not enough managers to generate comparison plot")
            return
            
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Get performance data
            performance = self.get_overall_performance()
            
            # Extract manager ROIs
            manager_names = []
            roi_values = []
            
            for name, data in performance['manager_data'].items():
                if 'roi' in data:
                    manager_names.append(name)
                    roi_values.append(data['roi'] * 100)  # Convert to percentage
                    
            if not manager_names:
                logger.warning("No ROI data available to plot")
                return
                
            # Create bar chart
            plt.figure(figsize=(10, 6))
            
            # Plot bars
            bars = plt.bar(manager_names, roi_values)
            
            # Color bars based on performance
            for i, bar in enumerate(bars):
                if roi_values[i] >= 0:
                    bar.set_color('green')
                else:
                    bar.set_color('red')
                    
            # Add value labels on bars
            for i, v in enumerate(roi_values):
                plt.text(i, v + np.sign(v) * 0.5, f'{v:.1f}%', 
                       ha='center', va='bottom' if v >= 0 else 'top')
                       
            # Add labels and title
            plt.xlabel('Manager')
            plt.ylabel('ROI (%)')
            plt.title(f'ROI Comparison Between Managers ({period})')
            
            # Add overall ROI line
            overall_roi = performance['overall_roi'] * 100
            plt.axhline(y=overall_roi, color='blue', linestyle='--', 
                      label=f'Overall ROI: {overall_roi:.1f}%')
                      
            plt.grid(True, alpha=0.3, axis='y')
            plt.legend()
            
            # Save or show
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"ROI comparison plot saved to {save_path}")
            else:
                plt.tight_layout()
                plt.show()
                
        except ImportError:
            logger.warning("Matplotlib not available. Cannot generate plot.")
            
        except Exception as e:
            logger.error(f"Error generating ROI comparison plot: {str(e)}")
            
    def export_data(self, 
                  export_path: str,
                  format: str = 'json') -> bool:
        """
        Export tracker data to file.
        
        Args:
            export_path: Path to export data
            format: Export format ('json' or 'csv')
            
        Returns:
            bool: Whether export was successful
        """
        try:
            # Prepare data
            data = {
                'bankroll_tracker': {
                    'name': self.name,
                    'last_update': self.last_update.isoformat(),
                    'managers_count': len(self.managers)
                },
                'performance': self.get_overall_performance(),
                'bankroll_history': self.bankroll_history,
                'roi_history': self.roi_history,
                'win_loss_history': self.win_loss_history,
                'managers': {}
            }
            
            # Add manager data
            for name, manager in self.managers.items():
                data['managers'][name] = manager.get_performance_summary()
                
            # Export based on format
            if format.lower() == 'json':
                with open(export_path, 'w') as f:
                    json.dump(data, f, indent=2)
                    
            elif format.lower() == 'csv':
                # For CSV, we'll export multiple files in a directory
                os.makedirs(export_path, exist_ok=True)
                
                # Export bankroll history
                bankroll_df = pd.DataFrame(self.bankroll_history)
                bankroll_df.to_csv(os.path.join(export_path, 'bankroll_history.csv'), index=False)
                
                # Export ROI history
                roi_df = pd.DataFrame(self.roi_history)
                roi_df.to_csv(os.path.join(export_path, 'roi_history.csv'), index=False)
                
                # Export win/loss history
                wl_df = pd.DataFrame(self.win_loss_history)
                wl_df.to_csv(os.path.join(export_path, 'win_loss_history.csv'), index=False)
                
                # Export manager summaries
                manager_data = []
                for name, manager in self.managers.items():
                    summary = manager.get_performance_summary()
                    summary['name'] = name
                    manager_data.append(summary)
                
                if manager_data:
                    manager_df = pd.DataFrame(manager_data)
                    manager_df.to_csv(os.path.join(export_path, 'manager_summary.csv'), index=False)
                    
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
                
            logger.info(f"Bankroll tracker data exported to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting bankroll tracker data: {str(e)}")
            return False 