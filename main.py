import pm4py
import pandas as pd

FILENAME = "BPI_Challenge_2013_closed_problems.xes.gz"

print(f"Loading dataset: {FILENAME} ...")

try:
    log = pm4py.read_xes(FILENAME)
    df = pm4py.convert_to_dataframe(log)

    print(f"Total Events (rows): {len(df)}")
    print(f"Total Cases (unique events): {len(df['case:concept:name'].unique())}")
    
    print("\n--- First 5 rows (test DF) ---")
    cols_to_show = ['case:concept:name', 'concept:name', 'time:timestamp', 'org:resource']
    available_cols = [c for c in cols_to_show if c in df.columns]
    print(df[available_cols].head())

    print("\nGenerating DFG (Directly-Follows Graph)...")
    dfg, start_act, end_act = pm4py.discover_dfg(log)
    pm4py.save_vis_dfg(dfg, start_act, end_act, "dfg.png")

except FileNotFoundError:
    print(f"ERROR: '{FILENAME}' file not found!")
except Exception as e:
    print(f"GENERIC ERROR: {e}")