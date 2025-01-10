import numpy as np

def poisson_process(rate, time_duration):
    """
    Simulates a Poisson process for a given rate and time duration.
    Parameters:
    - rate: float
        The Poisson rate (lambda) for the process.
    - time_duration: float
        The total time duration for the simulation.
    Returns:
    - num_events: int
        Total number of events that occurred.
    - event_times: list
        List of event times.
    - inter_arrival_times: list
        List of inter-arrival times between events.
    """
    inter_arrival_times = np.random.exponential(1 / rate, int(rate * time_duration * 10))
    event_times = np.cumsum(inter_arrival_times)
    event_times = event_times[event_times <= time_duration]
    num_events = len(event_times)
    return num_events, event_times.tolist(), inter_arrival_times[:num_events].tolist()

def poisson_simulation(rate, time_duration):
    """
    Simulates the Poisson Process for one or multiple rates.
    Parameters:
    - rate: float or list of floats
        The Poisson rate(s) at which shocks occur.
    - time_duration: float
        The time in years.
    Returns:
    - If rate is a single value:
        num_events: int
        event_times: list
        inter_arrival_times: list
    - If rate is a list:
        num_events_list: list of ints
        event_times_list: list of lists
        inter_arrival_times_list: list of lists
    """
    if isinstance(rate, (float, int)):
        num_events, event_times, inter_arrival_times = poisson_process(rate, time_duration)
        return num_events, event_times, inter_arrival_times
    
    elif isinstance(rate, list):
        num_events_list = []
        event_times_list = []
        inter_arrival_times_list = []
        
        for individual_rate in rate:
            num_events, event_times, inter_arrival_times = poisson_process(individual_rate, time_duration)
            num_events_list.append(num_events)
            event_times_list.append(event_times)
            inter_arrival_times_list.append(inter_arrival_times)
        
        return num_events_list, event_times_list, inter_arrival_times_list
    else:
        raise ValueError("Rate must be a float, int, or list of floats/ints.")

# Test for a single rate
rate = 0.1  # Events per unit time
time_duration = 1  # Simulate for 1 year
num_events, event_times, inter_arrival_times = poisson_simulation(rate, time_duration)
print("Single Rate Test")
print("Number of Events:", num_events)
print("Event Times:", event_times)
print("Inter-Arrival Times:", inter_arrival_times)

# Test for multiple rates
rates = [0.1, 0.2, 0.3]
time_duration = 10
num_events_list, event_times_list, inter_arrival_times_list = poisson_simulation(rates, time_duration)
print("\nMultiple Rates Test")
for i, rate in enumerate(rates):
    print(f"Rate: {rate}")
    print(f"Number of Events: {num_events_list[i]}")
    print(f"Event Times: {event_times_list[i]}")
    print(f"Inter-Arrival Times: {inter_arrival_times_list[i]}")
