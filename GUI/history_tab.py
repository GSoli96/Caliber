import streamlit as st
import json
import streamlit as st
import json
import pandas as pd
from utils.history_manager import get_all_history_entries
from utils.translations import get_text
from charts.plotly_charts import generate_usage_chart, generate_power_chart, generate_cumulative_co2_chart


def history_tab():
    st.header(get_text("tabs", "history"))
    entries = get_all_history_entries()
    if not entries:
        st.info(get_text("history", "no_entries"))
        return

    st.markdown(get_text("history", "entries_found", n=len(entries)))

    for entry in entries:
        # Formatta il timestamp per la visualizzazione
        try:
            run_time_dt = pd.to_datetime(entry['run_timestamp'])
            run_time_str = run_time_dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            run_time_str = entry['run_timestamp']

        status_icon = "✅" if entry['status'] == 'Success' else "❌"

        with st.expander(f"{status_icon} {get_text('history', 'run_title', date=run_time_str, model=entry['llm_model'] or 'N/A')}"):

            st.markdown(f"**{get_text('history', 'user_question')}** `{entry['user_question']}`")

            # --- Metriche Principali ---
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(get_text("history", "status"), entry['status'])
            col2.metric(get_text("history", "total_duration"), f"{entry['total_duration_s']:.2f}" if entry['total_duration_s'] else "N/A")
            col3.metric(get_text("history", "llm_backend"), entry['llm_backend'] or 'N/A')
            col4.metric(get_text("history", "db_engine"), entry['db_engine_type'] or 'N/A')

            if entry['error_message']:
                st.error(f"**{get_text('history', 'main_error')}** {entry['error_message']}")

            # --- Query Originale ---
            st.subheader(get_text("history", "original_query"))
            if entry['original_query']:
                st.code(entry['original_query'], language='sql')
                st.metric(get_text("history", "db_exec_duration"),
                          f"{entry['original_query_duration_s']:.3f}" if entry['original_query_duration_s'] else "N/A")

                try:
                    query_res = json.loads(entry['original_query_result'])
                    if query_res.get('error'):
                        st.error(f"{get_text('history', 'db_exec_error')} {query_res['error']}")
                    else:
                        st.success(get_text("history", "query_executed_rows", rows=query_res.get('rows', 'N/A')))
                except json.JSONDecodeError:
                    st.warning(get_text("history", "json_error"))
            else:
                st.info(get_text("history", "no_orig_query"))

            # --- Query Alternative ---
            st.subheader(get_text("history", "alt_queries"))
            try:
                alternatives = json.loads(entry['alternatives'])
                if not alternatives:
                    st.info(get_text("history", "no_alt_queries"))
                else:
                    st.write(get_text("history", "alt_attempts", n=len(alternatives)))
                    for i, alt in enumerate(alternatives):
                        st.markdown(f"--- \n**{get_text('history', 'alternative_n', n=i+1)}**")
                        sql_to_show = alt.get('generated_sql', get_text("history", "no_sql_output"))
                        st.code(sql_to_show, language="sql" if 'SELECT' in str(sql_to_show).upper() else "text")

                        if alt.get('status') == 'success':
                            alt_duration = alt.get('metrics', {}).get('duration_s', 'N/A')
                            st.success(get_text("history", "success_duration", s=alt_duration))
                        else:
                            st.error(get_text("history", "failed_reason", reason=alt.get('reason', 'N/A')))
            except json.JSONDecodeError:
                st.warning(get_text("history", "alt_json_error"))

            # --- Report Consumi ---
            st.subheader(get_text("history", "consumption_report"))
            with st.container(border=True):
                try:
                    monitoring_data_list = json.loads(entry['monitoring_data'])
                    if not monitoring_data_list:
                        st.info(get_text("history", "no_monitor_data"))
                    else:
                        monitoring_df = pd.json_normalize(monitoring_data_list)
                        monitoring_df['timestamp'] = pd.to_datetime(monitoring_df['timestamp'])
                        monitoring_df.rename(
                            columns={'cpu.percent': 'cpu_util_percent', 'cpu.power_w': 'cpu_power_w',
                                     'gpu.percent': 'gpu_util_percent', 'gpu.power_w': 'gpu_power_w'},
                            inplace=True, errors='ignore')

                        st.plotly_chart(generate_usage_chart(monitoring_df), use_container_width=True)
                        st.plotly_chart(generate_power_chart(monitoring_df), use_container_width=True)

                        monitoring_df['time_diff_s'] = monitoring_df['timestamp'].diff().dt.total_seconds().fillna(0)
                        monitoring_df['total_co2_gs'] = monitoring_df.get('cpu.co2_gs_cpu', 0).fillna(0)
                        if 'gpu.co2_gs_gpu' in monitoring_df.columns:
                            monitoring_df['total_co2_gs'] += monitoring_df['gpu.co2_gs_gpu'].fillna(0)

                        monitoring_df['cumulative_gco2'] = (
                                    monitoring_df['total_co2_gs'] * monitoring_df['time_diff_s']).cumsum()
                        st.plotly_chart(generate_cumulative_co2_chart([{'df': monitoring_df, 'name': 'Run'}],
                                                                      get_text("gen_eval", "live_co2_chart")),
                                        use_container_width=True)

                except json.JSONDecodeError:
                    st.warning(get_text("history", "monitor_json_error"))
                except Exception as e:
                    st.error(get_text("history", "chart_error", e=e))