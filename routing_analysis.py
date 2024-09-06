import torch
import matplotlib.pyplot as plt
import pandas as pd
import os

def routing_analysis(x, dir):

    batch_size, max_router_iter, block_size, n_layer = x.shape

    # Create layer/layer correlogram 
    d = {"layer"+str(i): [] for i in range(n_layer)}
    for batch in range(batch_size):
        for iter in range(max_router_iter):
            for token in range(block_size):
                for layer in range(n_layer):
                    d["layer"+str(layer)].append(x[batch, iter, token, layer])

    if not os.path.isdir(dir):
        os.makedirs(dir)


    df = pd.DataFrame(d)

    correlations = df.corr(method="kendall")

    plt.matshow(correlations, cmap="cividis")
    plt.colorbar()
    plt.xlabel("Layer")
    plt.ylabel("Layer")
    plt.savefig(dir + "/layer_layer.png")
    plt.close()

    # Now plot the strength of the layer for each sequence using matshow
    plt.matshow(x.mean(axis=(1,2)).T, cmap="cividis", aspect='auto')
    plt.colorbar()
    plt.xlabel("Sequence")
    plt.ylabel("Layer")
    plt.savefig(dir + "/layer_seq.png")
    plt.close()

    # Now plot the strength of the layer for each iteration using matshow
    plt.matshow(x.mean(axis=(0,2)).T, cmap="cividis", aspect='auto')
    plt.colorbar()
    plt.xlabel("Iteration")
    plt.ylabel("Layer")
    plt.savefig(dir + "/layer_iter.png")
    plt.close()

    # Now plot the strength of the layer for each token using matshow
    plt.matshow(x.mean(axis=(0,1)).T, cmap="cividis", aspect='auto')
    plt.colorbar()
    plt.xlabel("Token")
    plt.ylabel("Layer")
    plt.savefig(dir + "/layer_token.png")
    plt.close()

    # Now plot the average strength of each layer using bar plot
    plt.bar(range(n_layer), x.mean(axis=(0,1,2)))
    plt.xlabel("Layer")
    plt.ylabel("Mean strength")
    plt.savefig(dir + "/layer_mean.png")
    plt.close()






