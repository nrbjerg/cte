import numpy as np

filepath = "C:/Users/NX83SQ/GitHub/Benchmark_instances/Set_100/p4.2.a.txt"

def parse_top_file(file_path):
    top_data = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines[3:]:
            x, y, score = line.split()
            top_data.append([float(x), float(y), int(score)])

    return np.array(top_data)

def import_numbers(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        n = int(lines[0].split()[1])
        m = int(lines[1].split()[1])
        tmax = float(lines[2].split()[1])
        numbers = [n,m,tmax]
    
    return numbers

if __name__ =='__main__':
    temp_TOP = parse_top_file(filepath)
    TOP = temp_TOP[1:-1]
    Number_of_nodes = import_numbers(filepath)[0]
    Number_of_units = import_numbers(filepath)[1]
    T_max = import_numbers(filepath)[2]

    print(Number_of_nodes)
    print(Number_of_units)
    print(T_max)
    print(TOP)
    