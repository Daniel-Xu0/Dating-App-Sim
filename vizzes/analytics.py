import os
import json
import numpy as np


def numpy_to_python(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in list(obj)]
    return obj


def save_analytics(analytics, config, output_dir):
    """
    Save analytics data to a JSON file, handling numpy data types.
    
    Args:
        analytics: Dictionary containing analytics data
        config: Configuration file path
        output_dir: Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)
    converted_analytics = numpy_to_python(analytics)
    output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(config))[0]}_results.json")
    
    with open(output_file, 'w') as f:
        json.dump(converted_analytics, f, indent=4)

    print(f"Analytics saved in {output_file}")
    return output_file
