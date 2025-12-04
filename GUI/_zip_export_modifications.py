# ============================================================================
# MODIFICHE PER dataset_analytics_tab.py - Export Resource Metrics to ZIP
# ============================================================================

# STEP 1: Aggiungere import pandas (circa riga 30-35)
# Dopo "import time" aggiungi:
import pandas as pd

# STEP 2: Aggiungere funzione helper PRIMA di export_analytics_zip (circa riga 440)

def _aggregate_metrics(metrics_data):
    """
    Aggregate resource monitoring metrics into a summary DataFrame.
    
    Args:
        metrics_data: List of metric dicts from SystemMonitor
    
    Returns:
        pandas.DataFrame with aggregated metrics or None
    """
    if not metrics_data:
        return None
    
    try:
        cpu_percents = [m.get('cpu', {}).get('percent', 0) for m in metrics_data]
        gpu_percents = [m.get('gpu', {}).get('percent', 0) for m in metrics_data]
        co2_gs_cpu = [m.get('cpu', {}).get('co2_gs_cpu', 0) for m in metrics_data]
        co2_gs_gpu = [m.get('gpu', {}).get('co2_gs_gpu', 0) for m in metrics_data]
        co2_total = [cpu + gpu for cpu, gpu in zip(co2_gs_cpu, co2_gs_gpu)]
        
        summary = {
            'Metric': ['CPU %', 'GPU %', 'CO2 (g/s)'],
            'Min': [
                round(min(cpu_percents), 2) if cpu_percents else 0,
                round(min(gpu_percents), 2) if any(gpu_percents) else 0,
                round(min(co2_total), 6) if co2_total else 0
            ],
            'Max': [
                round(max(cpu_percents), 2) if cpu_percents else 0,
                round(max(gpu_percents), 2) if any(gpu_percents) else 0,
                round(max(co2_total), 6) if co2_total else 0
            ],
            'Mean': [
                round(sum(cpu_percents) / len(cpu_percents), 2) if cpu_percents else 0,
                round(sum(gpu_percents) / len(gpu_percents), 2) if any(gpu_percents) else 0,
                round(sum(co2_total) / len(co2_total), 6) if co2_total else 0
            ],
            'Total_CO2_g': [
                0,  # N/A for CPU%
                0,  # N/A for GPU%
                round(sum(co2_total) * 0.5, 4)  # Total CO2 (0.5s sampling interval)
            ]
        }
        
        return pd.DataFrame(summary)
    except Exception:
        return None


# STEP 3: Modificare la sezione "# 3. Profiling Relazionale" (circa riga 553-569)
# SOSTITUIRE il blocco esistente con:

                # 3. Profiling Relazionale (ui_profiling_relazionale)
                prof_ss_key = f"relprof:prof_{key_alter}:{table_name}"
                if prof_ss_key in st.session_state:
                    results = st.session_state[prof_ss_key].get("results", {})
                    # Semantic
                    if "semantic" in results:
                        z.writestr(f"{db_name}/{table_name}/profiling/semantic.csv", results["semantic"].to_csv(index=False))
                        
                        # Export semantic resource metrics if available
                        futures = st.session_state[prof_ss_key].get("futures", {})
                        if "sem" in futures and futures["sem"].done():
                            try:
                                sem_res = futures["sem"].result()
                                if isinstance(sem_res, dict) and "metrics" in sem_res:
                                    metrics_data = sem_res["metrics"]
                                    if metrics_data:
                                        # Export raw metrics as JSON
                                        z.writestr(f"{db_name}/{table_name}/profiling/semantic_resource_metrics.json", 
                                                   json.dumps(metrics_data, default=str, indent=4))
                                        
                                        # Export aggregated metrics as CSV
                                        metrics_df = _aggregate_metrics(metrics_data)
                                        if metrics_df is not None:
                                            z.writestr(f"{db_name}/{table_name}/profiling/semantic_resource_summary.csv", 
                                                       metrics_df.to_csv(index=False))
                            except:
                                pass
                    
                    # Heatmap data
                    futures = st.session_state[prof_ss_key].get("futures", {})
                    if "heat" in futures and futures["heat"].done():
                        try:
                            heat_res = futures["heat"].result()
                            if isinstance(heat_res, dict) and "miss_by_col" in heat_res:
                                z.writestr(f"{db_name}/{table_name}/profiling/missing_heatmap_stats.csv", heat_res["miss_by_col"].to_csv())
                        except:
                            pass


# STEP 4: Modificare la sezione "# 4. Integrità" (circa riga 571-576)
# SOSTITUIRE il blocco esistente con:

                # 4. Integrità (ui_integrita_dataset)
                intg_ss_key = f"relprof:intg_{key_alter}:{table_name}"
                if intg_ss_key in st.session_state:
                    results = st.session_state[intg_ss_key].get("results", {})
                    if "anomalies" in results:
                        z.writestr(f"{db_name}/{table_name}/profiling/anomalies.csv", results["anomalies"].to_csv(index=False))
                        
                        # Export anomalies resource metrics if available
                        futures = st.session_state[intg_ss_key].get("futures", {})
                        if "anom" in futures and futures["anom"].done():
                            try:
                                anom_res = futures["anom"].result()
                                if isinstance(anom_res, dict) and "metrics" in anom_res:
                                    metrics_data = anom_res["metrics"]
                                    if metrics_data:
                                        # Export raw metrics as JSON
                                        z.writestr(f"{db_name}/{table_name}/profiling/anomalies_resource_metrics.json", 
                                                   json.dumps(metrics_data, default=str, indent=4))
                                        
                                        # Export aggregated metrics as CSV
                                        metrics_df = _aggregate_metrics(metrics_data)
                                        if metrics_df is not None:
                                            z.writestr(f"{db_name}/{table_name}/profiling/anomalies_resource_summary.csv", 
                                                       metrics_df.to_csv(index=False))
                            except:
                                pass


# ============================================================================
# RISULTATO FINALE NEL ZIP
# ============================================================================
# Per ogni tabella, se i task di profiling sono stati eseguiti, il ZIP conterrà:
#
# db_name/
#   table_name/
#     profiling/
#       semantic.csv                          (già esistente)
#       semantic_resource_metrics.json        (NUOVO - metriche raw)
#       semantic_resource_summary.csv         (NUOVO - metriche aggregate)
#       anomalies.csv                         (già esistente)
#       anomalies_resource_metrics.json       (NUOVO - metriche raw)
#       anomalies_resource_summary.csv        (NUOVO - metriche aggregate)
#
# I file *_resource_summary.csv conterranno:
#   Metric    | Min   | Max   | Mean  | Total_CO2_g
#   CPU %     | 12.5  | 45.2  | 28.3  | 0
#   GPU %     | 0.0   | 15.3  | 5.2   | 0
#   CO2 (g/s) | 0.003 | 0.012 | 0.007 | 0.0035
