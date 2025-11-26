# GUI/green_optimizer_display.py
"""
Enhanced Green Query Optimizer Display Module
Provides side-by-side comparison view for standard vs optimized queries
"""

import streamlit as st
import pandas as pd
from utils import green_metrics
from utils.translations import get_text


def display_green_optimizer_results(results, info, query_res, timestamps, monitoring_data):
    """
    Display enhanced side-by-side comparison of Standard Query vs Green Query
    with CO2 savings badge and visual winner indicator.
    
    Args:
        results: Process results dictionary containing greenefy_results
        info: Query info dictionary with generated_sql
        query_res: Original query execution results
        timestamps: Execution timestamps dictionary
        monitoring_data: System monitoring data list
    """
    if 'greenefy_results' not in results:
        return False  # Not ready to display
    
    st.subheader("🌱 Green Query Optimizer - Side-by-Side Comparison")
    
    if results.get('greenefy_error'):
        st.error(results['greenefy_error'])
        return True
    
    # Get original query metrics
    orig_sql = info.get('generated_sql', '')
    orig_duration = results.get('metrics', {}).get('duration_s', 0)
    orig_rows = query_res.get('rows', 0)
    
    # Calculate original CO2 (approximate from monitoring data)
    orig_co2 = 0.0
    if monitoring_data and 'start_db' in timestamps and 'end_db' in timestamps:
        try:
            mon_df = pd.json_normalize(monitoring_data)
            mon_df['timestamp'] = pd.to_datetime(mon_df['timestamp'])
            db_mon = mon_df[(mon_df['timestamp'] >= timestamps['start_db']) & 
                           (mon_df['timestamp'] <= timestamps['end_db'])]
            if not db_mon.empty:
                mon_df['time_diff_s'] = mon_df['timestamp'].diff().dt.total_seconds().fillna(0)
                mon_df['total_co2_gs'] = mon_df.get('cpu.co2_gs_cpu', 0).fillna(0)
                if 'gpu.co2_gs_gpu' in mon_df.columns:
                    mon_df['total_co2_gs'] += mon_df['gpu.co2_gs_gpu'].fillna(0)
                orig_co2 = (mon_df['total_co2_gs'] * mon_df['time_diff_s']).sum()
        except Exception:
            pass
    
    # Find best green alternative
    best_green = None
    best_savings = 0
    for res in results['greenefy_results']:
        if res['status'] == 'success' and res.get('duration', float('inf')) < orig_duration:
            # Estimate CO2 based on duration ratio (simplified)
            est_co2 = orig_co2 * (res['duration'] / orig_duration) if orig_duration > 0 else 0
            savings = orig_co2 - est_co2
            if savings > best_savings:
                best_savings = savings
                best_green = res.copy()
                best_green['est_co2'] = est_co2
    
    if best_green:
        # SIDE-BY-SIDE COMPARISON
        st.markdown("### 🔬 Query Comparison")
        
        col_std, col_green = st.columns(2)
        
        with col_std:
            st.markdown("#### 📊 Standard Query")
            st.code(orig_sql, language='sql')
            st.metric("⏱️ Execution Time", f"{orig_duration:.4f}s")
            st.metric("📦 Rows Returned", f"{orig_rows:,}")
            st.metric("🌍 CO₂ Emissions", f"{orig_co2:.6f}g")
        
        with col_green:
            st.markdown("#### 🌱 Green Optimized Query")
            st.code(best_green['sql'], language='sql')
            st.metric("⏱️ Execution Time", f"{best_green['duration']:.4f}s", 
                     delta=f"{best_green['duration'] - orig_duration:.4f}s",
                     delta_color="inverse")
            st.metric("📦 Rows Returned", f"{best_green.get('rows', 0):,}")
            st.metric("🌍 CO₂ Emissions", f"{best_green['est_co2']:.6f}g",
                     delta=f"{best_green['est_co2'] - orig_co2:.6f}g",
                     delta_color="inverse")
        
        # CO2 SAVINGS BADGE
        savings_pct = (best_savings / orig_co2 * 100) if orig_co2 > 0 else 0
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #00ff9f 0%, #00cc7f 100%); 
                    padding: 20px; border-radius: 10px; text-align: center; 
                    margin: 20px 0; box-shadow: 0 4px 6px rgba(0,255,159,0.3);'>
            <h2 style='color: #0e1117; margin: 0;'>🎉 Potential CO₂ Savings</h2>
            <h1 style='color: #0e1117; margin: 10px 0; font-size: 3em;'>{best_savings:.6f}g</h1>
            <h3 style='color: #0e1117; margin: 0;'>({savings_pct:.1f}% reduction)</h3>
            <p style='color: #0e1117; margin-top: 10px;'>
                ≈ {green_metrics.co2_to_smartphones(best_savings):.4f} smartphones charged<br>
                ≈ {green_metrics.co2_to_car_km(best_savings):.4f} meters driven by car
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # WINNER INDICATOR
        st.success("✅ **Green Query is the Winner!** - More efficient and sustainable choice")
        
        # Show all alternatives in expander
        if len(results['greenefy_results']) > 1:
            with st.expander(f"📋 All {len(results['greenefy_results'])} Alternative Queries"):
                for i, res in enumerate(results['greenefy_results']):
                    with st.container(border=True):
                        st.markdown(f"**Option {i+1}**")
                        st.code(res['sql'], language='sql')
                        if res['status'] == 'success':
                            st.success(f"Rows: {res.get('rows')} | Duration: {res.get('duration'):.4f}s")
                        else:
                            st.error(f"Error: {res.get('error')}")
    else:
        # No successful green alternative found
        st.warning("⚠️ No green alternative was faster than the original query.")
        st.markdown("### All Attempted Optimizations")
        for i, res in enumerate(results['greenefy_results']):
            with st.container(border=True):
                st.markdown(f"**Option {i+1}**")
                st.code(res['sql'], language='sql')
                if res['status'] == 'success':
                    st.info(f"Rows: {res.get('rows')} | Duration: {res.get('duration'):.4f}s")
                else:
                    st.error(f"Error: {res.get('error')}")
    
    return True
