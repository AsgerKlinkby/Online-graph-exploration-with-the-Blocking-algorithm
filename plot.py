import math
import random
import numpy as np
import matplotlib.pyplot as plt
import copy
from GenerateGraphs import GenarateGraphs
from TSP import TSP

# This code is from: https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
def running_mean(y_in, x_in, N_out=1000, sigma=1):
    N_in = len(y_in)
    x_out = np.linspace(np.min(x_in), np.max(x_in), N_out)
    x_in_mesh, x_out_mesh = np.meshgrid(x_in, x_out)
    gauss_kernel = np.exp(-np.square(x_in_mesh - x_out_mesh) / (2 * sigma**2))
    normalization = np.tile(np.reshape(np.sum(gauss_kernel, axis=1), (N_out, 1)), (1, N_in))
    gauss_kernel_normalized = gauss_kernel / normalization
    y_out = gauss_kernel_normalized @ y_in
    return y_out, x_out

def test_experiments_delta():
    f = open("exp_delta.txt", "w");f.write("");f.close()
    f = open("exp_delta.txt", "a")
    delta = [0.0001, 0.1, 1, 2, 5, 10]
    generator = GenarateGraphs()
    graphs = generator.generate(100, 15, 30)
    for graph in graphs:
        tsp = TSP()
        tsp_cost, perm = tsp.python_tspm_solver(graph)
        for d in delta:
            graph_copy = copy.deepcopy(graph)
            blocking_cost = graph_copy.blocking(graph_copy.nodes[0], d, None)
            comp_ratio = blocking_cost / tsp_cost
            f.write("delta="+str(d)+" tsp="+str(tsp_cost)+" blocking="+str(blocking_cost)+" comp_ratio="+str(comp_ratio)+" size="+str(len(graph.nodes))+" edges="+str(len(graph.edges))+" weigths=uniform connectivity=30\n")
    f.close()

def test_experiments_random_delta():
    f = open("exp_random_delta.txt", "w");f.write("");f.close()
    f = open("exp_random_delta.txt", "a")
    size = 15
    generator = GenarateGraphs()
    graphs = generator.generate(100, size, 30)
    for graph in graphs:
        tsp = TSP()
        tsp_cost, perm = tsp.python_tspm_solver(graph)
        for d in range(1):
            delta = random.uniform(0.00000001, 30)
            graph_copy = copy.deepcopy(graph)
            blocking_cost = graph_copy.blocking(graph_copy.nodes[0], delta, None)
            comp_ratio = blocking_cost / tsp_cost
            f.write("delta="+str(delta)+" tsp="+str(tsp_cost)+" blocking="+str(blocking_cost)+" comp_ratio="+str(comp_ratio)+" size="+str(len(graph.nodes))+" edges="+str(len(graph.edges))+" weigths=uniform connectivity=30\n")
    f.close()

def test_experiments_connectivity():
    f = open("exp_connectivity_random.txt", "w");f.write("");f.close()
    f = open("exp_connectivity_random.txt", "a")
    generator = GenarateGraphs()
    delta = 2
    for i in range(100):
        connectivity = random.randint(0,100)
        graphs = generator.generate(1, 15, connectivity)
        graph = graphs[0]
        tsp = TSP()
        tsp_cost, perm = tsp.python_tspm_solver(graph)
        blocking_cost = graph.blocking(graph.nodes[0], delta, None)
        comp_ratio = blocking_cost / tsp_cost
        f.write("delta="+str(delta)+" tsp="+str(tsp_cost)+" blocking="+str(blocking_cost)+" comp_ratio="+str(comp_ratio)+" size="+str(len(graph.nodes))+" edges="+str(len(graph.edges))+" weigths=uniform connectivity="+str(connectivity)+"\n")
    f.close()

def test_experiments_size():
    f = open("exp_size.txt", "w");f.write("");f.close()
    f = open("exp_size.txt", "a")
    size = [4, 8, 12, 16, 18]
    generator = GenarateGraphs()
    delta = 2
    for s in size:
        graphs = generator.generate(100, s, 30)
        for graph in graphs:
            tsp = TSP()
            tsp_cost, perm = tsp.python_tspm_solver(graph)
            blocking_cost = graph.blocking(graph.nodes[0], delta, None)
            comp_ratio = blocking_cost / tsp_cost
            f.write("delta="+str(delta)+" tsp="+str(tsp_cost)+" blocking="+str(blocking_cost)+" comp_ratio="+str(comp_ratio)+" size="+str(len(graph.nodes))+" edges="+str(len(graph.edges))+" weigths=uniform connectivity=30\n")
    f.close()

def test_experiments_size_MST_lowerbound():
    f = open("exp_size_MST_lowerbound.txt", "w");f.write("");f.close()
    f = open("exp_size_MST_lowerbound.txt", "a")
    generator = GenarateGraphs()
    delta = 2
    for i in range(100):
        size = random.randint(2, 120)
        print("size: ", size)
        graphs = generator.generate(1, size, 30)
        for graph in graphs:
            mst = graph.mst_prim(graph.nodes[0])
            opt_lower_bound = 1
            for e in mst:
                opt_lower_bound += e.weight
            blocking_cost = graph.blocking(graph.nodes[0], delta, None)
            comp_ratio = blocking_cost / opt_lower_bound
            f.write("delta="+str(delta)+" opt_lower_bound="+str(opt_lower_bound)+" blocking="+str(blocking_cost)+" comp_ratio="+str(comp_ratio)+" size="+str(len(graph.nodes))+" edges="+str(len(graph.edges))+" weigths=uniform connectivity=30\n")
    f.close()

def test_experiments_size_random():
    f = open("exp_size_random.txt", "w");f.write("");f.close()
    f = open("exp_size_random.txt", "a")
    generator = GenarateGraphs()
    delta = 2
    for i in range(100):
        size = random.randint(3, 19)
        graphs = generator.generate(1, size, 30)
        graph = graphs[0]
        tsp = TSP()
        tsp_cost, perm = tsp.python_tspm_solver(graph)
        blocking_cost = graph.blocking(graph.nodes[0], delta, None)
        comp_ratio = blocking_cost / tsp_cost
        f.write("delta="+str(delta)+" tsp="+str(tsp_cost)+" blocking="+str(blocking_cost)+" comp_ratio="+str(comp_ratio)+" size="+str(len(graph.nodes))+" edges="+str(len(graph.edges))+" weigths=uniform connectivity=30\n")
    f.close()

def test_experiments_weights_uniform():
    f = open("exp_weights_uniform_r.txt", "w");f.write("");f.close()
    f = open("exp_weights_uniform_r.txt", "a")
    delta = 2
    size = 15
    generator = GenarateGraphs()
    graphs = generator.generate(100, size, 30)
    for graph in graphs:
        tsp = TSP()
        tsp_cost, perm = tsp.python_tspm_solver(graph)
        delta = random.uniform(0.0001, 10)
        blocking_cost = graph.blocking(graph.nodes[0], delta, None)
        comp_ratio = blocking_cost / tsp_cost
        f.write("delta="+str(delta)+" tsp="+str(tsp_cost)+" blocking="+str(blocking_cost)+" comp_ratio="+str(comp_ratio)+" size="+str(len(graph.nodes))+" edges="+str(len(graph.edges))+" weigths=uniform connectivity=30\n")
    f.close()

def test_experiments_weights_linear():
    f = open("exp_weights_linear_r.txt", "w");f.write("");f.close()
    f = open("exp_weights_linear_r.txt", "a")
    weights = []
    delta = 2
    size = 15
    num_of_edges_upperbound = math.ceil((size * (size + 1)) / 2)
    for w in range(num_of_edges_upperbound):
        weights.append(w)
    generator = GenarateGraphs()
    graphs = generator.generate(100, size, 30, weights)
    for graph in graphs:
        tsp = TSP()
        tsp_cost, perm = tsp.python_tspm_solver(graph)
        delta = random.uniform(0.0001, 10)
        blocking_cost = graph.blocking(graph.nodes[0], delta, None)
        comp_ratio = blocking_cost / tsp_cost
        f.write("delta="+str(delta)+" tsp="+str(tsp_cost)+" blocking="+str(blocking_cost)+" comp_ratio="+str(comp_ratio)+" size="+str(len(graph.nodes))+" edges="+str(len(graph.edges))+" weigths=linear connectivity=30\n")
    f.close()

def test_experiments_weights_poly():
    f = open("exp_weights_poly_r.txt", "w");f.write("");f.close();
    f = open("exp_weights_poly_r.txt", "a")
    weights = []
    delta = 2
    size = 15
    num_of_edges_upperbound = math.ceil((size * (size + 1)) / 2)
    for w in range(num_of_edges_upperbound):
        weights.append((w+1)**2)
    generator = GenarateGraphs()
    graphs = generator.generate(100, size, 30, weights)
    for graph in graphs:
        tsp = TSP()
        tsp_cost, perm = tsp.python_tspm_solver(graph)
        delta = random.uniform(0.0001, 10)
        blocking_cost = graph.blocking(graph.nodes[0], delta, None)
        comp_ratio = blocking_cost / tsp_cost
        f.write("delta="+str(delta)+" tsp="+str(tsp_cost)+" blocking="+str(blocking_cost)+" comp_ratio="+str(comp_ratio)+" size="+str(len(graph.nodes))+" edges="+str(len(graph.edges))+" weigths=poly connectivity=30\n")
    f.close()

def test_experiments_weights_set_exp():
    f = open("exp_weights_exp_set_r.txt", "w");f.write("");f.close();
    f = open("exp_weights_exp_set_r.txt", "a")
    weights = set()
    delta = 2
    size = 15
    num_of_edges_upperbound = math.ceil((size * (size + 1)) / 2)
    for w in range(num_of_edges_upperbound):
        weights.add(math.pow(1.5, float(w)))
    generator = GenarateGraphs()
    graphs = generator.generate(100, size, 30, list(weights))
    for graph in graphs:
        tsp = TSP()
        tsp_cost, perm = tsp.python_tspm_solver(graph)
        delta = random.uniform(0.0001, 10)
        blocking_cost = graph.blocking(graph.nodes[0], delta, None)
        comp_ratio = blocking_cost / tsp_cost
        f.write("delta="+str(delta)+" tsp="+str(tsp_cost)+" blocking="+str(blocking_cost)+" comp_ratio="+str(comp_ratio)+" size="+str(len(graph.nodes))+" edges="+str(len(graph.edges))+" weigths=exp_set connectivity=30\n")
    f.close()

def test_experiments_weights_list_exp():
    f = open("exp_weights_exp_list_r.txt", "w");f.write("");f.close();
    f = open("exp_weights_exp_list_r.txt", "a")
    weights = []
    delta = 2
    size = 15
    num_of_edges_upperbound = math.ceil((size * (size + 1)) / 2)
    for w in range(num_of_edges_upperbound):
        weights.append(math.pow(1.5, float(w)))
    generator = GenarateGraphs()
    graphs = generator.generate(100, size, 30, weights)
    for graph in graphs:
        tsp = TSP()
        tsp_cost, perm = tsp.python_tspm_solver(graph)
        delta = random.uniform(0.0001, 10)
        blocking_cost = graph.blocking(graph.nodes[0], delta, None)
        comp_ratio = blocking_cost / tsp_cost
        f.write("delta="+str(delta)+" tsp="+str(tsp_cost)+" blocking="+str(blocking_cost)+" comp_ratio="+str(comp_ratio)+" size="+str(len(graph.nodes))+" edges="+str(len(graph.edges))+" weigths=exp_list connectivity=30\n")
    f.close()

def test_experiments_weights_normal():
    f = open("exp_weights_normal_r.txt", "w");f.write("");f.close();
    f = open("exp_weights_normal_r.txt", "a")
    weights = []
    size = 15
    num_of_edges_upperbound = math.ceil((size * (size + 1)) / 2)
    for w in range(num_of_edges_upperbound):
        we = max(0.00001, random.normalvariate(5, 3))
        weights.append(we)
    generator = GenarateGraphs()
    graphs = generator.generate(100, size, 30, weights)
    for graph in graphs:
        tsp = TSP()
        tsp_cost, perm = tsp.python_tspm_solver(graph)
        delta = random.uniform(0.0001, 10)
        blocking_cost = graph.blocking(graph.nodes[0], delta, None)
        comp_ratio = blocking_cost / tsp_cost
        f.write("delta="+str(delta)+" tsp="+str(tsp_cost)+" blocking="+str(blocking_cost)+" comp_ratio="+str(comp_ratio)+" size="+str(len(graph.nodes))+" edges="+str(len(graph.edges))+" weigths=normal connectivity=30\n")
    f.close()

def show_plot(filename, x_axis_lbl, y_axis_lbl, x_axis_title="", y_axis_title=""):
    f = open(filename, "r")
    x_list = []
    y_list = []
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"]})
    plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command
    for line in f.readlines():
        attributes = line.split(" ")
        for at in attributes:
            if x_axis_lbl in at:
                value = at.split("=")
                x_list.append(float(value[1]))
            if y_axis_lbl in at:
                value = at.split("=")
                y_list.append(float(value[1]))

    mean_y_list, mean_x_list = running_mean(y_list, x_list, sigma=5)
    plt.plot(mean_x_list, mean_y_list, c="#ff7f00")
    plt.xlabel(x_axis_title)
    plt.ylabel(y_axis_title)
    plt.scatter(x_list, y_list, c="#ff7f00", alpha=0.4)
    plt.savefig(filename.split('.')[0] + ".pdf", format='pdf')
    plt.show()
    plt.clf()



def show_weights_boxplot(files, names, colors, x_axis_lbl, y_axis_lbl, x_axis_title="", y_axis_title=""):
    x_list = []
    y_list_list = []
    colors_plot = []
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"]})
    plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
    for i, file in enumerate(files):
        y_list = []
        f = open(file, "r")
        for line in f.readlines():
            colors_plot.append(colors[i])
            attributes = line.split(" ")
            for at in attributes:
                if x_axis_lbl in at:
                    value = at.split("=")
                    k = value[1] if value[0] == "weigths" else float(value[1])
                    x_list.append(k)
                if y_axis_lbl in at:
                    value = at.split("=")
                    k = value[1] if value[0] == "weigths" else float(value[1])
                    y_list.append(k)
        y_list_list.append(y_list)
    plt.boxplot(y_list_list, labels=names)
    plt.xlabel(x_axis_title)
    plt.ylabel(y_axis_title)
    # plt.savefig(filename.split('.')[0] + ".pdf", format='pdf')
    plt.show()

def show_weights_plot(files, names, colors, x_axis_lbl, y_axis_lbl, x_axis_title="", y_axis_title=""):
    x_list = []
    y_list = []
    colors_plot = []
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"]})
    plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
    fig, ax = plt.subplots()
    for i, file in enumerate(files):
        f = open(file, "r")
        for line in f.readlines():
            colors_plot.append(colors[i])
            attributes = line.split(" ")
            for at in attributes:
                if x_axis_lbl in at:
                    value = at.split("=")
                    k = value[1] if value[0] == "weigths" else float(value[1])
                    x_list.append(k)
                if y_axis_lbl in at:
                    value = at.split("=")
                    k = value[1] if value[0] == "weigths" else float(value[1])
                    y_list.append(k)
        if x_axis_lbl != "weigths":
            #co = np.polyfit(np.array(x_list), np.array(y_list), 1)
            #poly_1d = np.poly1d(co)
            #y_list = poly_1d(np.array(x_list))
            #ax.plot(x_list, y_list, colors[i])
            mean_y_list, mean_x_list = running_mean(y_list, x_list)
            ax.plot(mean_x_list, mean_y_list, colors[i])
        ax.scatter(x_list, y_list, c=colors[i], label=names[i], alpha=5)
        x_list = []
        y_list = []

    ax.legend(title="Weights")
    plt.xlabel(x_axis_title)
    plt.ylabel(y_axis_title)
    # plt.savefig(filename.split('.')[0] + ".pdf", format='pdf')
    plt.show()

def run_forever():
    for i in range(2):
        #test_experiments_delta()
        #test_experiments_random_delta()
        #test_experiments_connectivity()
        #test_experiments_size()
        # test_experiments_weights_linear()
        # test_experiments_weights_poly()
        # test_experiments_weights_set_exp()
        # test_experiments_weights_list_exp()
        # test_experiments_weights_uniform()
        test_experiments_weights_normal()

def exp_list_higher_exponent():
    f = open("exp_list_weight_randon.txt", "w");f.write("");f.close();
    f = open("exp_list_weight_randon.txt", "a")
    size = 15
    num_of_edges_upperbound = math.ceil((size * (size + 1)) / 2)
    for i in range(100):
        base = random.uniform(1, 10)
        weights = []
        for w in range(num_of_edges_upperbound):
            weights.append(math.pow(base, float(w)))
        generator = GenarateGraphs()
        graphs = generator.generate(1, size, 30, weights)
        graph = graphs[0]
        tsp = TSP()
        tsp_cost, perm = tsp.python_tspm_solver(graph)
        blocking_cost = graph.blocking(graph.nodes[0], 2, None)
        comp_ratio = blocking_cost / tsp_cost
        f.write("delta="+str(2)+" tsp="+str(tsp_cost)+" blocking="+str(blocking_cost)+" comp_ratio="+str(comp_ratio)+" size="+str(len(graph.nodes))+" edges="+str(len(graph.edges))+" weigths=" + str(base) + " connectivity=30\n")
    f.close()

def gen_one():
    weights = []
    size = 10
    num_of_edges_upperbound = math.ceil((size * (size + 1)) / 2)
    for w in range(num_of_edges_upperbound):
        weights.append(math.pow(8, float(w)))
    print(weights)
    generator = GenarateGraphs()
    graphs = generator.generate(1, size, 30, weights)

    tsp = TSP()
    tsp_cost, perm = tsp.python_tspm_solver(graphs[0])
    blocking_cost = graphs[0].blocking(graphs[0].nodes[0], 2, "fffff.txt")
    comp_ratio = blocking_cost / tsp_cost
    print("comp", comp_ratio)
    # graphs[0].draw_spacial_graph("exp_list_weight_drawing.png")
    graphs[0].tikz("graph_exp.txt", 2, graphs[0].nodes[0], True)

def change_weights(filename, new_weight):
    f = open(filename, "r")
    new_lines = []
    for line in f.readlines():
        new_line = ""
        attributes = line.split(" ")
        for at in attributes:
            value = at.split("=")
            if value[0] == "weigths":
                new_line += "weigths=" + new_weight + " "
            else:
                new_line += at + " "
        new_lines.append(new_line.strip() + "\n")
    f = open(filename, "w")
    f.writelines(new_lines)

def make_all_plots():

    files = ['exp_weights_uniform_r.txt', 'exp_weights_exp_normal_r.txt', 'exp_weights_linear_r.txt', 'exp_weights_poly_r.txt', 'exp_weights_exp_list_r.txt', 'exp_weights_exp_set_r.txt']
    names = ['uniform', 'normal', 'linear', 'polynomial', 'small exponential', 'large exponential']
    colors = ['#377eb8', '#984ea3', '#4daf4a', '#a65628', '#ff7f00', '#e41a1c']
    y_axis_title = "Competitive ratio"
    y_axis_lbl = "comp_ratio"

    filename = "exp_size_MST_lowerbound.txt"
    x_axis_lbl = "size"
    x_axis_title = "Number of vertices"
    # show_plot(filename, x_axis_lbl, y_axis_lbl, x_axis_title, y_axis_title)

    filename = "exp_connectivity_random.txt"
    x_axis_lbl = "edges"
    x_axis_title = "Number of edges"
    show_plot(filename, x_axis_lbl, y_axis_lbl, x_axis_title, y_axis_title)

    filename = "exp_size_random.txt"
    x_axis_lbl = "size"
    x_axis_title = "Number of vertices"
    show_plot(filename, x_axis_lbl, y_axis_lbl, x_axis_title, y_axis_title)

    # filename = "exp_list_weight_randon.txt"
    # x_axis_lbl = "weigths"
    # x_axis_title = "The base of the exponential function: $a$"
    # show_plot(filename, x_axis_lbl, y_axis_lbl, x_axis_title, y_axis_title)

    x_axis_lbl = "weigths"
    x_axis_title = "Weight distribution"
    # show_weights_boxplot(files, names, colors, x_axis_lbl, y_axis_lbl, x_axis_title, y_axis_title)

    x_axis_lbl = "delta"
    x_axis_title = r"$\delta$"
    show_weights_plot(files, names, colors, x_axis_lbl, y_axis_lbl, x_axis_title, y_axis_title)


if __name__ == '__main__':
    filename = "exp_size_MST_lowerbound.txt"
    x_axis_lbl = "size"
    y_axis_lbl = "comp_ratio"
    files = ['exp_weights_uniform_r.txt', 'exp_weights_normal_r.txt', 'exp_weights_linear_r.txt', 'exp_weights_poly_r.txt', 'exp_weights_exp_list_r.txt', 'exp_weights_exp_set_r.txt']
    names = ['uniform', 'normal', 'linear', 'polynomial', 'small exponential', 'large exponential']
    colors = ['#377eb8', '#984ea3', '#4daf4a', '#a65628', '#ff7f00', '#e41a1c',]

    make_all_plots()
    # test_experiments_size_MST_lowerbound()
    # show_plot(filename, x_axis_lbl, y_axis_lbl)
    # run_forever()
    # run_forever()
    # colors = ['blue', 'red', 'green', 'black', 'cyan']
    # test_experiments_weights_normal_exp()
    # show_weights_boxplot(files, names, colors, x_axis_lbl, y_axis_lbl)
    # show_weights_plot(files, names, colors, x_axis_lbl, y_axis_lbl)
    # gen_one()
    # change_weights("exp_weights_exp_list_r.txt", "exp_list")
    # test_experiments_size_random()
    # test_experiments_connectivity()
