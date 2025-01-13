import time
from functools import wraps

# Dictionary to store total time and number of calls for each function
function_stats = {}

# Define iteration counts for different functions
iteration_counts = {
    'step': 50000,
    'update': 1000
}

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__  # Get the name of the function being decorated

        # Initialize stats for this function if not present
        if func_name not in function_stats:
            function_stats[func_name] = {"total_time": 0, "num_calls": 0}

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        # Update the stats for the current function
        elapsed_time = end_time - start_time
        function_stats[func_name]["total_time"] += elapsed_time
        function_stats[func_name]["num_calls"] += 1

        # Get iteration count for the function, default to 1000 if not specified
        iterations = iteration_counts.get(func_name, 1000)

        if function_stats[func_name]["num_calls"] % iterations == 0:
            avg_time = function_stats[func_name]["total_time"] / function_stats[func_name]["num_calls"]
            print(f"Average Execution Time for {func_name} after {function_stats[func_name]['num_calls']} calls: {avg_time:.6f} seconds")

        return result

    return wrapper
