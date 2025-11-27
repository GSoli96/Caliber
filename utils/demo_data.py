# utils/demo_data.py
"""
Demo data for video presentation.
Contains preset queries, metrics, and monitoring data for simulation.
"""

import pandas as pd
from datetime import datetime, timezone, timedelta
import random

# ============================================================================
# DEMO MODE CONFIGURATION
# ============================================================================
DEMO_MODE_ENABLED = True  # Set to False to use real LLM/DB execution

# ============================================================================
# PRESET QUERY DATA
# ============================================================================

DEMO_ORIGINAL_QUERY = """SELECT 
    occupation,
    education,
    COUNT(*) as total_count,
    AVG(age) as avg_age,
    AVG(`hours-per-week`) as avg_hours_per_week,
    SUM(CASE WHEN `capital-gain` > 0 THEN 1 ELSE 0 END) as people_with_capital_gain
FROM Adult
WHERE age > 25
GROUP BY occupation, education
ORDER BY total_count DESC
LIMIT 100;"""

DEMO_QUERY_RESULT_DATA = pd.DataFrame({
    'occupation': ['Prof-specialty', 'Exec-managerial', 'Craft-repair', 'Sales', 
                   'Adm-clerical', 'Tech-support', 'Machine-op-inspct', 'Transport-moving',
                   'Handlers-cleaners', 'Other-service'],
    'education': ['Bachelors', 'Bachelors', 'HS-grad', 'Some-college', 
                  'Some-college', 'Assoc-voc', 'HS-grad', 'HS-grad',
                  'HS-grad', 'HS-grad'],
    'total_count': [3420, 2890, 2650, 2340, 2180, 1950, 1820, 1670, 1540, 1420],
    'avg_age': [42.5, 45.2, 38.7, 36.4, 35.8, 34.2, 37.5, 39.1, 33.8, 32.5],
    'avg_hours_per_week': [42.3, 45.8, 41.2, 40.5, 39.7, 40.1, 42.8, 43.5, 41.3, 38.9],
    'people_with_capital_gain': [450, 520, 180, 210, 190, 150, 120, 95, 75, 60]
})

# ============================================================================
# GREENEFY OPTIMIZED QUERIES
# ============================================================================

DEMO_GREENEFY_QUERIES = [
    {
        'sql': """SELECT 
    occupation,
    education,
    COUNT(*) as total_count,
    AVG(age) as avg_age,
    AVG(`hours-per-week`) as avg_hours_per_week,
    SUM(CASE WHEN `capital-gain` > 0 THEN 1 ELSE 0 END) as people_with_capital_gain
FROM Adult
WHERE age > 25
GROUP BY occupation, education
HAVING total_count > 100
ORDER BY total_count DESC
LIMIT 100;""",
        'description': 'Optimized with HAVING clause to filter before sorting',
        'duration': 0.082,
        'rows': 10,
        'co2_reduction': 15.3,  # percentage
        'cpu_reduction': 12.5,
        'power_reduction': 14.2
    },
    {
        'sql': """WITH occupation_stats AS (
    SELECT 
        occupation,
        education,
        COUNT(*) as total_count,
        AVG(age) as avg_age,
        AVG(`hours-per-week`) as avg_hours_per_week
    FROM Adult
    WHERE age > 25
    GROUP BY occupation, education
)
SELECT 
    occupation,
    education,
    total_count,
    avg_age,
    avg_hours_per_week,
    (SELECT COUNT(*) FROM Adult a2 
     WHERE a2.occupation = occupation_stats.occupation 
     AND a2.education = occupation_stats.education 
     AND a2.`capital-gain` > 0) as people_with_capital_gain
FROM occupation_stats
ORDER BY total_count DESC
LIMIT 100;""",
        'description': 'CTE-based optimization with subquery for capital gain calculation',
        'duration': 0.075,
        'rows': 10,
        'co2_reduction': 22.7,
        'cpu_reduction': 18.9,
        'power_reduction': 20.5
    },
    {
        'sql': """SELECT 
    occupation,
    education,
    total_count,
    avg_age,
    avg_hours_per_week,
    people_with_capital_gain
FROM (
    SELECT 
        occupation,
        education,
        COUNT(*) as total_count,
        AVG(age) as avg_age,
        AVG(`hours-per-week`) as avg_hours_per_week,
        SUM(CASE WHEN `capital-gain` > 0 THEN 1 ELSE 0 END) as people_with_capital_gain
    FROM Adult
    WHERE age > 25
    GROUP BY occupation, education
    ORDER BY COUNT(*) DESC
    LIMIT 100
) as top_occupations
ORDER BY total_count DESC;""",
        'description': 'Subquery with early LIMIT for reduced processing',
        'duration': 0.068,
        'rows': 10,
        'co2_reduction': 28.4,
        'cpu_reduction': 25.1,
        'power_reduction': 26.8
    }
]

# ============================================================================
# TIMING CONFIGURATION
# ============================================================================

# Simulation delays (in seconds)
DEMO_LLM_GENERATION_TIME = 3.5  # Time to "generate" query
DEMO_DB_EXECUTION_TIME = 1.2    # Time to "execute" query
DEMO_GREENEFY_GENERATION_TIME = 4.0  # Time to generate optimized queries
DEMO_GREENEFY_EXECUTION_TIME = 0.8   # Time per optimized query execution

# Original query metrics
DEMO_ORIGINAL_DURATION = 0.145  # seconds
DEMO_ORIGINAL_ROWS = 10

# ============================================================================
# MONITORING DATA GENERATION
# ============================================================================

def generate_demo_monitoring_data(start_time, duration_seconds, phase='generation', intensity='medium'):
    """
    Generate simulated monitoring data for demo purposes.
    
    Args:
        start_time: datetime object for start
        duration_seconds: how long the phase lasts
        phase: 'generation', 'db_execution', or 'greenefy'
        intensity: 'low', 'medium', 'high' - affects CPU/GPU usage
    
    Returns:
        List of monitoring data points
    """
    monitoring_data = []
    
    # Intensity settings
    intensity_settings = {
        'low': {'cpu_base': 15, 'cpu_var': 10, 'gpu_base': 5, 'gpu_var': 5, 'power_mult': 0.6},
        'medium': {'cpu_base': 45, 'cpu_var': 15, 'gpu_base': 30, 'gpu_var': 15, 'power_mult': 1.0},
        'high': {'cpu_base': 75, 'cpu_var': 15, 'gpu_base': 60, 'gpu_var': 20, 'power_mult': 1.4}
    }
    
    settings = intensity_settings.get(intensity, intensity_settings['medium'])
    
    # Generate data points every 0.1 seconds
    num_points = int(duration_seconds / 0.1)
    
    for i in range(num_points):
        timestamp = start_time + timedelta(seconds=i * 0.1)
        
        # Simulate varying CPU/GPU usage
        cpu_percent = max(0, min(100, settings['cpu_base'] + random.uniform(-settings['cpu_var'], settings['cpu_var'])))
        gpu_percent = max(0, min(100, settings['gpu_base'] + random.uniform(-settings['gpu_var'], settings['gpu_var'])))
        
        # Calculate power based on usage (simplified model)
        cpu_power_w = (cpu_percent / 100) * 65.0 * settings['power_mult']  # Assuming 65W TDP
        gpu_power_w = (gpu_percent / 100) * 150.0 * settings['power_mult']  # Assuming 150W TDP
        
        # Calculate CO2 (g/s) - simplified: 250 g CO2/kWh
        emission_factor = 250.0  # g CO2/kWh
        cpu_co2_gs = (cpu_power_w / 1000) * (emission_factor / 3600)  # Convert W to kW, then to g/s
        gpu_co2_gs = (gpu_power_w / 1000) * (emission_factor / 3600)
        
        data_point = {
            'timestamp': timestamp.isoformat(),
            'cpu': {
                'percent': cpu_percent,
                'power_w': cpu_power_w,
                'co2_gs_cpu': cpu_co2_gs
            },
            'gpu': {
                'percent': gpu_percent,
                'power_w': gpu_power_w,
                'co2_gs_gpu': gpu_co2_gs
            }
        }
        
        monitoring_data.append(data_point)
    
    return monitoring_data


def get_demo_full_process_results(user_question):
    """
    Generate complete demo results for the full process (generation + execution).
    
    Returns:
        Tuple of (result_holder, monitoring_data)
    """
    now = datetime.now(timezone.utc)
    
    # Phase 1: LLM Generation
    start_process = now
    end_generation = start_process + timedelta(seconds=DEMO_LLM_GENERATION_TIME)
    
    # Phase 2: DB Execution
    start_db = end_generation
    end_db = start_db + timedelta(seconds=DEMO_DB_EXECUTION_TIME)
    
    # Generate monitoring data
    monitoring_data = []
    monitoring_data.extend(generate_demo_monitoring_data(start_process, DEMO_LLM_GENERATION_TIME, 'generation', 'high'))
    monitoring_data.extend(generate_demo_monitoring_data(start_db, DEMO_DB_EXECUTION_TIME, 'db_execution', 'medium'))
    
    # Result holder
    result_holder = {
        'timestamps': {
            'start_process': start_process,
            'end_generation': end_generation,
            'start_db': start_db,
            'end_db': end_db
        },
        'info': {
            'raw_llm_output': DEMO_ORIGINAL_QUERY,
            'generated_sql': DEMO_ORIGINAL_QUERY,
            'query_result': {
                'data': DEMO_QUERY_RESULT_DATA.copy(),
                'rows': DEMO_ORIGINAL_ROWS,
                'error': None
            }
        },
        'metrics': {
            'duration_s': DEMO_ORIGINAL_DURATION
        }
    }
    
    return result_holder, monitoring_data


def get_demo_greenefy_results(original_query, original_co2=0.0):
    """
    Generate demo results for Greenefy optimization phase.
    
    Returns:
        Tuple of (greenefy_data, monitoring_data, timestamps)
    """
    now = datetime.now(timezone.utc)
    
    start_greenefy = now
    
    # Generate monitoring for greenefy generation phase
    monitoring_data = []
    monitoring_data.extend(generate_demo_monitoring_data(
        start_greenefy, 
        DEMO_GREENEFY_GENERATION_TIME, 
        'greenefy', 
        'high'
    ))
    
    # Simulate execution of each optimized query
    current_time = start_greenefy + timedelta(seconds=DEMO_GREENEFY_GENERATION_TIME)
    
    greenefy_results = []
    for query_data in DEMO_GREENEFY_QUERIES:
        # Add monitoring for this query execution
        monitoring_data.extend(generate_demo_monitoring_data(
            current_time,
            DEMO_GREENEFY_EXECUTION_TIME,
            'db_execution',
            'low'
        ))
        
        # Create result entry
        result = {
            'sql': query_data['sql'],
            'status': 'success',
            'rows': query_data['rows'],
            'duration': query_data['duration'],
            'result': {
                'data': DEMO_QUERY_RESULT_DATA.copy(),
                'rows': query_data['rows'],
                'error': None
            },
            'co2_reduction': query_data['co2_reduction'],
            'cpu_reduction': query_data['cpu_reduction'],
            'power_reduction': query_data['power_reduction'],
            'description': query_data['description']
        }
        greenefy_results.append(result)
        
        current_time += timedelta(seconds=DEMO_GREENEFY_EXECUTION_TIME)
    
    end_greenefy = current_time
    
    timestamps = {
        'start_greenefy': start_greenefy,
        'end_greenefy': end_greenefy
    }
    
    greenefy_data = {
        'greenefy_candidates': [q['sql'] for q in DEMO_GREENEFY_QUERIES],
        'greenefy_results': greenefy_results
    }
    
    return greenefy_data, monitoring_data, timestamps


# ============================================================================
# BENCHMARKING TAB DEMO DATA
# ============================================================================

DEMO_BENCHMARKING_NL_QUERIES = [
    "Show all people with age greater than 30",
    "Count how many people work more than 40 hours per week",
    "Find the average age by occupation",
    "List all people with Bachelors education",
    "Show occupation distribution for people with capital gain"
]

DEMO_BENCHMARKING_SQL_RESULTS = [
    "SELECT * FROM Adult WHERE age > 30",
    "SELECT COUNT(*) FROM Adult WHERE `hours-per-week` > 40",
    "SELECT occupation, AVG(age) as avg_age FROM Adult GROUP BY occupation",
    "SELECT * FROM Adult WHERE education = 'Bachelors'",
    "SELECT occupation, COUNT(*) as count FROM Adult WHERE `capital-gain` > 0 GROUP BY occupation"
]

def get_demo_benchmarking_results():
    """
    Generate demo results for benchmarking NL->SQL tab.
    """
    results = []
    for i, (nl_query, sql_query) in enumerate(zip(DEMO_BENCHMARKING_NL_QUERIES, DEMO_BENCHMARKING_SQL_RESULTS)):
        results.append({
            'nl_query': nl_query,
            'generated_sql': sql_query,
            'latency_s': 0.5 + (i * 0.1),  # Simulated latency
            'co2_g': 0.0015 + (i * 0.0002)  # Simulated CO2
        })
    
    return pd.DataFrame(results)
