import math

def co2_to_smartphones(co2_g: float) -> float:
    """
    Converts CO2 grams to number of smartphones charged.
    Assumption: Charging a smartphone releases approx 8.22 g CO2 (EPA data).
    """
    if co2_g <= 0:
        return 0.0
    # Source: EPA Greenhouse Gas Equivalencies Calculator
    # 1 smartphone charged = 0.00000822 metric tons CO2 = 8.22 g CO2
    return co2_g / 8.22

def co2_to_car_km(co2_g: float) -> float:
    """
    Converts CO2 grams to km driven by a standard passenger vehicle.
    Assumption: Average car emits approx 120 g CO2/km (EU target/avg).
    """
    if co2_g <= 0:
        return 0.0
    # Source: European Environment Agency (approx 120g/km for new cars)
    return co2_g / 120.0

def calculate_green_score(co2_g: float, rows_returned: int, execution_time_s: float) -> int:
    """
    Calculates a heuristic 'Green Score' (0-100).
    Higher is better (more efficient).
    
    Logic:
    - Efficiency = Rows / (CO2 * Time)
    - We normalize this to a 0-100 scale using a logarithmic mapping.
    """
    if co2_g <= 0 or execution_time_s <= 0:
        return 100 # Ideal state (no consumption)
    
    # Avoid division by zero
    co2_g = max(co2_g, 0.0001)
    execution_time_s = max(execution_time_s, 0.0001)
    
    # Base metric: Rows per (Gram of CO2 * Second)
    # High rows, low CO2, low time => High Score
    # Example: 1000 rows, 0.1g CO2, 0.1s => 1000 / 0.01 = 100,000
    # Example: 10 rows, 10g CO2, 10s => 10 / 100 = 0.1
    
    efficiency_raw = rows_returned / (co2_g * execution_time_s)
    
    # Logarithmic scaling to map wide range to 0-100
    # We assume an "excellent" efficiency is around 10,000 and "poor" is < 1
    # log10(100,000) = 5
    # log10(0.1) = -1
    
    try:
        log_val = math.log10(efficiency_raw)
    except ValueError:
        return 0
        
    # Map range [-1, 5] to [0, 100]
    # -1 -> 0
    # 5 -> 100
    # score = (log_val - min) / (max - min) * 100
    
    min_log = -1.0
    max_log = 5.0
    
    score = ((log_val - min_log) / (max_log - min_log)) * 100
    
    return max(0, min(100, int(score)))



def co2_to_led_bulbs_on(co2_g: float,
                        hours_on: float = 0.10,
                        bulb_watt: float = 9.0,
                        emission_factor_g_per_kwh: float = 226.4) -> float:
    """
    Converts CO2 grams to an equivalent number of LED bulbs (on) for a given time.

    Interpretation:
    - Result = how many LED bulbs of 'bulb_watt' you could keep ON
      for 'hours_on' hours, emettendo la stessa CO2.

    Defaults:
    - bulb_watt = 9 W (LED ≈ 60 W incandescenza)
    - emission_factor_g_per_kwh = 226.4 g CO2/kWh (media UE 2021)
    """
    if co2_g <= 0 or hours_on <= 0:
        return 0.0

    # kWh consumati da *una* lampadina accesa per 'hours_on' ore
    kwh_per_bulb = (bulb_watt / 1000.0) * hours_on

    # CO2 emessa da una lampadina per 'hours_on' ore
    co2_per_bulb = kwh_per_bulb * emission_factor_g_per_kwh  # ≈ 2.04 g/ora per 9 W

    return co2_g / co2_per_bulb
