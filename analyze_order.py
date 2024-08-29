import torch
import matplotlib.pyplot as plt
import pandas as pd
import os

def analyze_order(x, dir):

    batch_size, max_router_iter, n_layer = x.shape

    # Create layer/layer correlogram 
    d = {"layer"+str(i): [] for i in range(n_layer)}
    for batch in range(batch_size):
        for iter in range(max_router_iter):
            for layer in range(n_layer):
                d["layer"+str(layer)].append(x[batch, iter, layer])

    if not os.path.isdir(dir):
        os.makedirs(dir)


    df = pd.DataFrame(d)
    print(len(df), batch_size, max_router_iter, n_layer)
    correlations = df.corr()
    plt.matshow(correlations, cmap="cividis")
    plt.colorbar()
    plt.xlabel("Layer")
    plt.ylabel("Layer")
    plt.savefig(dir + "/layer_layer.png")

    # Now plot the strength of the layer at each iteration using matshow
    plt.matshow(x.mean(axis=0).T, cmap="cividis")
    plt.colorbar()
    plt.xlabel("Iteration")
    plt.ylabel("Layer")
    plt.savefig(dir + "/layer_iter.png")




