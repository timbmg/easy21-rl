import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import dill as pickle

sns.set()

def plot(Q, actions):

    from mpl_toolkits.mplot3d import Axes3D

    pRange = list(range(1,22))
    dRange = list(range(1,11))
    vStar = list()
    for p in pRange:
        for d in dRange:
            vStar.append( [p, d, np.max([Q[p, d, a] for a in actions])] )

    df = pd.DataFrame(vStar, columns=['player', 'dealer', 'value'])

    # And transform the old column name in something numeric
    # df['player']=pd.Categorical(df['player'])
    # df['player']=df['player'].cat.codes

    # Make the plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(df['dealer'], df['player'], df['value'], cmap=plt.cm.viridis, linewidth=0.2)
    plt.show()

    # to Add a color bar which maps values to colors.
    surf=ax.plot_trisurf(df['dealer'], df['player'], df['value'], cmap=plt.cm.viridis, linewidth=0.2)
    fig.colorbar( surf, shrink=0.5, aspect=5)
    plt.show()

    # Rotate it
    ax.view_init(30, 45)
    plt.show()

    # Other palette
    ax.plot_trisurf(df['dealer'], df['player'], df['value'], cmap=plt.cm.jet, linewidth=0.01)
    plt.show()

def plotMseEpisodesLambdas(arr):

    # https://stackoverflow.com/questions/45857465/create-a-2d-array-from-another-array-and-its-indices-with-numpy
    m,n = arr.shape
    I,J = np.ogrid[:m,:n]
    out = np.empty((m,n,3), dtype=arr.dtype)
    out[...,0] = I
    out[...,1] = J
    out[...,2] = arr
    out.shape = (-1,3)

    df = pd.DataFrame(out, columns=['lambda', 'Episode', 'MSE'])
    df['lambda'] = df['lambda'] / 10
    #df = df.loc[df.index % 100 == 0]
    g = sns.FacetGrid(df, hue="lambda", size=8, legend_out=True)
    #g.map(plt.scatter, "episode", "mse")
    g = g.map(plt.plot, "Episode", "MSE").add_legend()

    plt.subplots_adjust(top=0.9)
    g.fig.suptitle('Mean Squared Error per Episode')

    plt.show()

def plotMseLambdas(data, lambdas):
    df = pd.DataFrame(data, columns=['MSE'])
    df['lambda'] = lambdas

    sns.pointplot(x=df['lambda'], y=df['MSE'])
    plt.title("Mean Squared Error per Lambda")
    plt.show()
