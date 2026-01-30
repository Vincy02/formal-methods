import pm4py
import pandas as pd
import matplotlib.pyplot as plt
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.algo.evaluation.precision import algorithm as precision_eval
from pm4py.algo.evaluation.generalization import algorithm as generalization_eval
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_eval

FILENAME = "BPI_Challenge_2013_closed_problems.xes.gz"
# model_*.png   -> Petri Net PNG
# metrics_*.png -> Metrics chart PNG
# final_results.csv -> final results CSV

def analyze_algorithm(log, discovery_func, algo_name):
    print(f"\n--- {algo_name.upper()} ---")
    
    # 1
    print(f"   [1/4] Discovery model used...")
    if algo_name == "Alpha Miner":
        net, im, fm = pm4py.discover_petri_net_alpha(log)
    elif algo_name == "Heuristic Miner":
        net, im, fm = pm4py.discover_petri_net_heuristics(log)
    elif algo_name == "Inductive Miner":
        net, im, fm = pm4py.discover_petri_net_inductive(log)
    
    # 2    
    print(f"   [2/4] Generating petri net png file...")
    petri_filename = f"model_{algo_name.lower().replace(' ', '_')}.png"
    try:
        pm4py.save_vis_petri_net(net, im, fm, petri_filename)
        print(f"         -> Saved as: {petri_filename}")
    except Exception as e:
        print(f"         -> Error: {e}")

    # 3
    print(f"   [3/4] Calculation of the 4 metrics...")
    try:
        # Fitness
        fit_res = replay_fitness.apply(log, net, im, fm, variant=replay_fitness.Variants.TOKEN_BASED)
        fitness = fit_res['log_fitness']
        # Precision
        precision = precision_eval.apply(log, net, im, fm)
        # Generalization
        generalization = generalization_eval.apply(log, net, im, fm)
        # Simplicity
        simplicity = simplicity_eval.apply(net)
    except Exception as e:
        print(f"         -> Error: {e}")
        fitness, precision, generalization, simplicity = 0, 0, 0, 0

    metrics = {
        "Fitness": fitness,
        "Precision": precision,
        "Generalization": generalization,
        "Simplicity": simplicity
    }
    
    # 4
    print(f"   [4/4] Generating metrics png file...")
    chart_filename = f"metrics_{algo_name.lower().replace(' ', '_')}.png"
    save_metrics_chart(metrics, algo_name, chart_filename)
    
    # Added in metrics for csv file 
    metrics['Algorithm'] = algo_name
    return metrics

def save_metrics_chart(metrics_dict, title, filename):
    labels = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    colors = ['#0000FF', '#008000', '#FFA500', '#FF0000']
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, values, color=colors)
    plt.title(f"Metrics: {title}")
    plt.ylabel("Score (0 - 1)")
    plt.ylim(0, 1.1)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{height:.3f}', ha='center', va='bottom')
        
    plt.savefig(filename)
    plt.close()
    print(f"         -> Saved as: {filename}")

# --- MAIN EXECUTION --- #

print(f"Loading log file: {FILENAME} ...")
log = pm4py.read_xes(FILENAME)

results = []

results.append(analyze_algorithm(log, None, "Alpha Miner"))
results.append(analyze_algorithm(log, None, "Heuristic Miner"))
results.append(analyze_algorithm(log, None, "Inductive Miner"))

# Saving CSV
df = pd.DataFrame(results)
cols = ['Algorithm'] + [c for c in df.columns if c != 'Algorithm']
df = df[cols]
df.to_csv("final_results.csv", index=False)