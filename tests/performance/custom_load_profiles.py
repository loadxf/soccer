"""
Custom load profiles for the Soccer Prediction System API performance tests.
These profiles define different load patterns to simulate various user scenarios.
"""

from locust import LoadTestShape
import math
import random
import time


class StepLoadShape(LoadTestShape):
    """
    Step load shape that steps up and down in user count.
    
    The shape increases load in steps and then decreases in steps.
    This is useful for testing how the system handles gradual increases in load.
    """
    
    step_time = 30  # Time (seconds) between steps
    step_load = 10  # User count increase per step
    spawn_rate = 10  # Users to spawn per second at peak
    time_limit = 600  # Time limit for the entire test
    
    def tick(self):
        run_time = self.get_run_time()
        
        if run_time > self.time_limit:
            return None
        
        current_step = math.floor(run_time / self.step_time) + 1
        
        # Increase for half the duration, then decrease
        midpoint_step = self.time_limit / (2 * self.step_time)
        
        if current_step > midpoint_step:
            # Decreasing phase
            step_number = math.floor(self.time_limit / self.step_time) - current_step
        else:
            # Increasing phase
            step_number = current_step
        
        user_count = step_number * self.step_load
        return (max(1, round(user_count)), self.spawn_rate)


class PeakHoursShape(LoadTestShape):
    """
    Simulates peak hours traffic with sudden spikes in user load.
    
    This shape creates periodic spikes to simulate high traffic periods,
    followed by periods of lower traffic.
    """
    
    time_limit = 1200  # Time limit for the entire test in seconds
    peak_users = 100   # Maximum number of users during peaks
    baseline_users = 10  # Baseline number of users during non-peak periods
    peak_duration = 60  # Duration of each peak in seconds
    trough_duration = 120  # Duration of each trough (non-peak) in seconds
    spawn_rate = 20  # Maximum spawn rate during peak
    
    def tick(self):
        run_time = self.get_run_time()
        
        if run_time > self.time_limit:
            return None
        
        # Calculate if we're in a peak or trough
        cycle_time = self.peak_duration + self.trough_duration
        cycle_position = run_time % cycle_time
        
        if cycle_position < self.peak_duration:
            # We're in a peak period
            user_count = self.peak_users
            current_spawn_rate = self.spawn_rate
        else:
            # We're in a trough period
            user_count = self.baseline_users
            current_spawn_rate = round(self.spawn_rate / 4)  # Lower spawn rate during troughs
        
        return (user_count, current_spawn_rate)


class SinusoidalLoadShape(LoadTestShape):
    """
    Creates a sinusoidal load pattern to simulate natural daily traffic patterns.
    
    This provides a smooth increase and decrease in load over time,
    which can mimic real-world traffic patterns.
    """
    
    time_limit = 900  # 15 minutes
    min_users = 5
    max_users = 80
    spawn_rate = 10
    
    def tick(self):
        run_time = self.get_run_time()
        
        if run_time > self.time_limit:
            return None
            
        # Sinusoidal user count between min and max
        amplitude = (self.max_users - self.min_users) / 2
        offset = (self.max_users + self.min_users) / 2
        user_count = round(amplitude * math.sin((run_time / self.time_limit) * 2 * math.pi) + offset)
        
        # Adjust spawn rate based on whether we're in increasing or decreasing phase
        if run_time < self.time_limit / 2:
            current_spawn_rate = self.spawn_rate
        else:
            current_spawn_rate = max(1, round(self.spawn_rate / 2))
            
        return (max(1, user_count), current_spawn_rate)


class WeekdayLoadShape(LoadTestShape):
    """
    Simulates a typical weekday traffic pattern with morning and evening peaks.
    
    This creates a realistic double-peak pattern that mimics workday traffic,
    with higher load during morning and evening rush hours.
    """
    
    time_limit = 1440  # 24 minutes to represent 24 hours
    min_users = 10
    morning_peak = 70
    afternoon_dip = 30
    evening_peak = 100
    night_dip = 5
    spawn_rate = 15
    
    def tick(self):
        run_time = self.get_run_time()
        
        if run_time > self.time_limit:
            return None
        
        hour = (run_time / 60) % 24  # Convert to 24-hour time
        
        # Early morning (0-6 hours): night_dip to min_users
        if hour < 6:
            user_count = self.night_dip + (self.min_users - self.night_dip) * (hour / 6)
        
        # Morning ramp up (6-9 hours): min_users to morning_peak
        elif hour < 9:
            user_count = self.min_users + (self.morning_peak - self.min_users) * ((hour - 6) / 3)
        
        # Morning peak to afternoon dip (9-14 hours): morning_peak to afternoon_dip
        elif hour < 14:
            user_count = self.morning_peak - (self.morning_peak - self.afternoon_dip) * ((hour - 9) / 5)
        
        # Afternoon dip to evening peak (14-19 hours): afternoon_dip to evening_peak
        elif hour < 19:
            user_count = self.afternoon_dip + (self.evening_peak - self.afternoon_dip) * ((hour - 14) / 5)
        
        # Evening ramp down (19-24 hours): evening_peak to night_dip
        else:
            user_count = self.evening_peak - (self.evening_peak - self.night_dip) * ((hour - 19) / 5)
        
        # Set spawn rate based on user count
        current_spawn_rate = max(1, round(self.spawn_rate * (user_count / self.evening_peak)))
        
        return (max(1, round(user_count)), current_spawn_rate)


class SpikeLoadShape(LoadTestShape):
    """
    Simulates sudden traffic spikes to stress test the system's ability to handle
    rapid changes in load, as might occur during a major event or news.
    """
    
    time_limit = 300  # 5 minutes
    baseline_users = 10
    spike_users = 150
    spike_duration = 30  # 30 seconds
    spawn_rate = 50  # High spawn rate to create rapid spike
    
    def tick(self):
        run_time = self.get_run_time()
        
        if run_time > self.time_limit:
            return None
        
        # Create 2 spikes during the test
        if (60 <= run_time <= 60 + self.spike_duration) or (180 <= run_time <= 180 + self.spike_duration):
            # During spike
            return (self.spike_users, self.spawn_rate)
        else:
            # Baseline load
            return (self.baseline_users, self.spawn_rate // 5) 