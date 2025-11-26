# utils/system_monitor_utilities.py

import psutil
import time
import GPUtil as GPU
import platform
import os
import streamlit as st

EMISSION_FACTOR = 250.0
_CPU_TDP_W = 65.0
_GPU_POWER_AVAILABLE = False
_IS_FIRST_RUN = True

def _detect_cpu_tdp():
    """
    Detect and estimate CPU Thermal Design Power (TDP) based on core count and frequency.
    
    Uses heuristics based on CPU specifications to estimate power consumption.
    Higher frequency and more cores typically indicate higher TDP.
    
    Returns:
        float: Estimated CPU TDP in Watts
    
    Note:
        This is an estimation. Actual TDP may vary by CPU model.
    """
    cores = psutil.cpu_count(logical=False)
    try:
        freq = psutil.cpu_freq().max
    except AttributeError:
        freq = 3500
    if freq > 4000:
        if cores >= 8: return 125.0
        if cores >= 4: return 95.0
    elif freq > 3000:
        if cores >= 6: return 65.0
        if cores >= 4: return 45.0
    return _CPU_TDP_W

def _setup_power_monitoring():
    """
    Initialize power monitoring system and detect available hardware.
    
    Detects CPU TDP, checks for NVIDIA GPU availability, and initializes
    session state with default emission factor and CPU TDP values.
    This function runs only once on first call.
    
    Side Effects:
        - Sets global variables _CPU_TDP_W, _GPU_POWER_AVAILABLE, _IS_FIRST_RUN
        - Initializes st.session_state['emission_factor'] and st.session_state['cpu_tdp']
    """
    global _CPU_TDP_W, _IS_FIRST_RUN, _GPU_POWER_AVAILABLE
    if not _IS_FIRST_RUN:
        return
    _CPU_TDP_W = _detect_cpu_tdp()
    try:
        import pynvml
        pynvml.nvmlInit()
        if pynvml.nvmlDeviceGetCount() > 0:
            _GPU_POWER_AVAILABLE = True
    except Exception:
        pass
    _IS_FIRST_RUN = False
    if 'emission_factor' not in st.session_state:
        st.session_state['emission_factor'] = EMISSION_FACTOR
    if 'cpu_tdp' not in st.session_state:
        st.session_state['cpu_tdp'] = _CPU_TDP_W

def estimate_cpu_power(cpu_percent, max_tdp):
    """
    Estimate current CPU power consumption based on utilization percentage.
    
    Args:
        cpu_percent: CPU utilization percentage (0-100)
        max_tdp: Maximum Thermal Design Power in Watts
    
    Returns:
        float: Estimated power consumption in Watts
    """
    return (cpu_percent / 100.0) * max_tdp

def watt_to_co2_gs(watt_power, emission_factor):
    """
    Convert power consumption (Watts) to CO2 emissions (grams per second).
    
    Args:
        watt_power: Power consumption in Watts
        emission_factor: CO2 emission factor in g CO2/kWh
    
    Returns:
        float: CO2 emissions in grams per second
    """
    return watt_power / 3600000 * emission_factor

def co2_gs_to_co2_kgh(co2_gs):
    """
    Convert CO2 emissions from grams/second to kilograms/hour.
    
    Args:
        co2_gs: CO2 emissions in grams per second
    
    Returns:
        float: CO2 emissions in kilograms per hour
    """
    return co2_gs * 3.6

def get_static_system_info():
    """
    Retrieve static system information (CPU, RAM, GPU specifications).
    
    Collects hardware specifications that don't change during runtime,
    such as CPU cores, frequency, RAM total, and GPU model.
    
    Returns:
        dict: Dictionary containing static system information with keys:
            - 'cpu': CPU specifications (cores, frequency, TDP)
            - 'ram': RAM specifications (total, percent)
            - 'gpu': GPU specifications (name, memory) if available
    """
    _setup_power_monitoring()
    info = {}
    if 'cpu_tdp' not in st.session_state:
        st.session_state['cpu_tdp'] = 0.0
    info['cpu'] = {
        'Cores': psutil.cpu_count(logical=False),
        'Frequency': f"{psutil.cpu_freq().max:.2f} Mhz",
        'TDP_W': f"{(st.session_state.cpu_tdp):.1f} W (Est.)",
    }
    ram = psutil.virtual_memory()
    info['ram'] = {'Totale': f"{ram.total / (1024 ** 3):.2f} GB", 'percent': ram.percent}
    try:
        gpus = GPU.getGPUs()
        if gpus:
            gpu = gpus[0]
            info['gpu'] = {
                'Nome': gpu.name,
                'Memoria Totale': f"{gpu.memoryTotal / 1024:.2f} GB",
                'percent': gpu.load * 100,
            }
    except Exception:
        pass
    return info

# MODIFICA CHIAVE: Aggiunti argomenti opzionali
def get_dynamic_system_info(emission_factor=None, cpu_tdp=None):
    """
    Retrieve current system metrics including resource usage and CO2 emissions.
    
    Collects real-time metrics for CPU, RAM, and GPU (if available),
    including utilization percentages, power consumption, and CO2 emissions.
    
    Args:
        emission_factor: Optional CO2 emission factor (g CO2/kWh).
            If None, uses session_state value or default.
        cpu_tdp: Optional CPU TDP in Watts.
            If None, uses session_state value or detected value.
    
    Returns:
        dict: Dictionary containing dynamic system metrics with keys:
            - 'cpu': CPU metrics (utilization, power, CO2)
            - 'ram': RAM metrics (available, used, percent)
            - 'gpu': GPU metrics (memory, power, CO2) if available
    """
    _setup_power_monitoring()
    dynamic_info = {}

    # Usa i valori passati se disponibili, altrimenti fallback su session_state
    emission_factor = emission_factor if emission_factor is not None else st.session_state.get('emission_factor', EMISSION_FACTOR)
    cpu_tdp = cpu_tdp if cpu_tdp is not None else st.session_state.get('cpu_tdp', _CPU_TDP_W)

    cpu_percent = psutil.cpu_percent(interval=0.1)
    cpu_power_w = estimate_cpu_power(cpu_percent, cpu_tdp)
    co2_gs_cpu = watt_to_co2_gs(cpu_power_w, emission_factor)
    co2_kgh_cpu = co2_gs_to_co2_kgh(co2_gs_cpu)
    dynamic_info['cpu'] = {
        'Utilizzo': f"{cpu_percent:.1f}%", 'percent': cpu_percent, 'power_w': cpu_power_w,
        'co2_gs_cpu': co2_gs_cpu, 'co2_kgh_cpu': co2_kgh_cpu,
    }
    ram = psutil.virtual_memory()
    dynamic_info['ram'] = {
        'Disponibile': f"{ram.available / (1024 ** 3):.2f} GB",
        'Utilizzo': f"{ram.used / (1024 ** 3):.2f} GB",
        'percent': ram.percent
    }
    try:
        gpus = GPU.getGPUs()
        if gpus:
            gpu = gpus[0]
            gpu_percent = gpu.load * 100
            gpu_power_w = 0.0
            if _GPU_POWER_AVAILABLE:
                import pynvml
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            else:
                gpu_power_w = (gpu_percent / 100.0) * 150.0
            co2_gs_gpu = watt_to_co2_gs(gpu_power_w, emission_factor)
            co2_kgh_gpu = co2_gs_to_co2_kgh(co2_gs_gpu)
            dynamic_info['gpu'] = {
                'Memoria Disponibile': f"{(gpu.memoryTotal - gpu.memoryUsed) / 1024:.2f} GB",
                'Utilizzo': f"{gpu.memoryUsed / 1024:.2f} GB",
                'percent': gpu_percent, 'power_w': gpu_power_w,
                'co2_gs_gpu': co2_gs_gpu, 'co2_kgh_gpu': co2_kgh_gpu,
            }
    except Exception:
        if 'gpu' in dynamic_info: del dynamic_info['gpu']
        pass
    return dynamic_info

import threading
from datetime import datetime, timezone

class SystemMonitor(threading.Thread):
    def __init__(self, data_list, stop_event, emission_factor, cpu_tdp):
        super().__init__()
        self.data_list, self.stop_event = data_list, stop_event
        self.emission_factor, self.cpu_tdp = emission_factor, cpu_tdp
        self.daemon = True

    def run(self):
        while not self.stop_event.is_set():
            try:
                metrics = get_dynamic_system_info(emission_factor=self.emission_factor, cpu_tdp=self.cpu_tdp)
                metrics['timestamp'] = datetime.now(timezone.utc)
                self.data_list.append(metrics)
                time.sleep(0.5)
            except Exception as e:
                print(f"[ERROR in THREAD MONITOR]: {e}")