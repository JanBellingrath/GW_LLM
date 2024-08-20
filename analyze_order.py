import torch
import matplotlib.pyplot as plt
import pandas as pd

batch_size = 32
max_router_iter = 10
n_layer = 3
random_data = torch.rand(batch_size, max_router_iter, n_layer)

x = random_data.numpy()




# Create layer/layer correlogram 
d = {"layer"+str(i): [] for i in range(n_layer)}
for batch in range(batch_size):
    for iter in range(max_router_iter):
        for layer in range(n_layer):
            d["layer"+str(layer)].append(x[batch, iter, layer])

df = pd.DataFrame(d)
correlations = df.corr()
plt.matshow(correlations)
plt.colorbar()
plt.savefig("tmp.png")

# Now plot the strength of the layer at each iteration using matshow
plt.matshow(x.mean(axis=0).T)
plt.colorbar()
plt.savefig("tmp2.png")




