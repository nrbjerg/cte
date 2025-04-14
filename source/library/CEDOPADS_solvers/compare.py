# A script for comparing logs.
import os 
import numpy as np 
import matplotlib.pyplot as plt 
from typing import List
import pprint
    
def compare_logs(file_ids: List[str]):
    """Generates a comparison plot for the different logs supplied."""
    data = dict()

    for file_id in file_ids:
        with open(os.path.join(os.getcwd(), "logs", file_id + ".log"), "r") as file:
            lines = file.readlines()[1:]
            
            for id_part, _, _, mean_part, std_part in map(lambda line: line.split(","), lines):
                problem_id = id_part.split(" ")[-1]
                mean = float(mean_part.split(" ")[-1])
                std = float(std_part.split(" ")[-1])
                data[problem_id] = dict(data.get(problem_id, {}), **{file_id: (mean, std)})

    pprint.pprint(data)
        
    plt.style.use("ggplot")
    width = 0.55
    x = np.arange(len(data.keys()))

    tab10 = plt.get_cmap("tab10")
    for idx, problem_id in enumerate(sorted(data.keys())):
        xs = np.linspace(idx - 0.15, idx + 0.15, len(file_ids))
        for jdx, file_id in enumerate(sorted(file_ids)):
            plt.errorbar(xs[jdx], data[problem_id][file_id][0], data[problem_id][file_id][1], marker="o", color=tab10(jdx))

    plt.xticks(ticks=x, labels=list(sorted(data.keys())))
    plt.title('Comparison of logs from quick benchmark (10 runs each)')
    plt.xlabel("Problem ID")
    plt.ylabel("MBS & STD")
    import matplotlib.lines as mlines
    indicators = [mlines.Line2D([], [], color=tab10(jdx), label=file_id, marker = "s") for jdx, file_id in enumerate(sorted(file_ids))]
    plt.legend(handles=indicators, loc=2)

    # Show the plot
    plt.show()
        
if __name__ == "__main__":
    compare_logs(["updated_grasp", "grasp", "grasp_with_higher_p", "SA_with_SI"])