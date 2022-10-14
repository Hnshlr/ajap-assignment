import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d as plt3d
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from tabulate import tabulate

class MultiLinearRegression:

    def __init__(self, filename, predColIndexes, targetColIndex, iterations, learningRate):
        self.filename = filename

        # Load data, remove first row (columns names), remove nan values and normalize it:
        self.columnsNames = np.genfromtxt(self.filename, delimiter=',', dtype=str, max_rows=1)
        self.data = np.genfromtxt(self.filename, delimiter=',', skip_header=1)
        self.data = self.data[~np.isnan(self.data).any(axis=1)]
        self.data = preprocessing.scale(self.data)

        self.Y = self.data[:, targetColIndex]
        self.data = self.data[:, predColIndexes]
        self.data = np.insert(self.data, 0, 1, axis=1)

        # Number of features and observations:
        self.r = len(self.data[0])  # number of features
        self.n = len(self.data)  # number of observations

        # Initialize thetas, hypothesis, iterations and learning rate:
        self.thetas = np.ones(self.r)  # initialize thetas
        self.hypothesis = np.zeros(self.n)  # initialize hypothesis
        self.iterations = iterations
        self.learningRate = learningRate

        # Cost:
        self.cost = []

    # Print amount features and observations:
    def printInfo(self):
        print('Features: ', self.r)
        print('Observations: ', self.n)

    # Plot data with different colors for each feature:
    def plotData(self):
        colors = ['purple', 'red', 'blue', 'orange', 'grey']
        for i in range(self.r - 1):
            plt.scatter(self.data[:, i], self.Y, color=colors[i], marker='o', s=1, label=self.columnsNames[i])
            plt.title('Energy output (EP) vs ' + self.columnsNames[i])
            plt.legend()
        plt.show()

    # Plot features:
    def plotFeatures(self):
        for i in range(self.r - 1):
            plt.scatter(self.data[:, i], self.Y, color='grey', marker='o', s=1, label=self.columnsNames[i])
            plt.title('Energy output (EP) vs ' + self.columnsNames[i])
            plt.legend()
            plt.show()

    # Plot each feature's prediction line using thetas:
    def plotFeaturesAndTheirPredictionLine(self):
        for i in range(self.r - 1):
            plt.scatter(self.data[:, i], self.Y, color='grey', marker='o', s=1, label=self.columnsNames[i])
            plt.plot(self.data[:, i], self.thetas[i]*self.data[:, i], color='red', label='Prediction line')
            plt.title('Energy output (EP) vs ' + self.columnsNames[i])
            plt.legend()
            plt.show()

    # Cost function:
    def costFunction(self):
        return (1 / (2 * self.n)) * np.sum(np.square(self.hypothesis - self.Y))

    # Gradient descent:
    def gradientDescent(self):
        cost = []
        for i in range(self.iterations):
            self.hypothesis = np.dot(self.data, self.thetas)
            for j in range(self.r):
                self.thetas[j] -= self.learningRate * (1 / self.n) * np.sum((self.hypothesis - self.Y) * self.data[:, j])
            cost.append(self.costFunction())
        self.cost = cost
        return cost

    # Gradient descent while thetas successive differences are greater than 0.0001:
    def gradientDescentUntilConvergence(self):
        cost = []
        thetas = np.zeros(self.r)
        while np.sum(np.abs(thetas - self.thetas)) > 0.0001:
            thetas = np.copy(self.thetas)
            self.hypothesis = np.dot(self.data, self.thetas)
            for j in range(self.r):
                self.thetas[j] -= self.learningRate * (1 / self.n) * np.sum((self.hypothesis - self.Y) * self.data[:, j])
            cost.append(self.costFunction())
        self.cost = cost
        return cost

    # Plot cost:
    def plotCost(self):
        plt.plot(self.cost, label='Cost evolution after ' + str(len(self.cost)) + ' iterations, and eta=' + str(self.learningRate))
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.legend()
        plt.show()

    # Compute R2:
    def computeR2(self):
        return 1 - np.sum(np.square(self.hypothesis - self.Y)) / (self.n*np.var(self.Y))

    # Print R2:
    def printR2(self):
        print('R2 for the ' + self.filename + ' data, in percentage: ', str(np.round(self.computeR2()*100, 2)), ' %')

# Plot multiple plots in one figure:
def plotMultiple(filename, predColIndexes, targetColIndex, iterations, learningRates):
    fig, axs = plt.subplots(4, 4)
    fig.suptitle('Cost for various iterations and learning rates values.')
    for i in range(len(learningRates)):
        for j in range(len(iterations)):
            mlr = MultiLinearRegression(filename, predColIndexes, targetColIndex, iterations[j], learningRates[i])
            mlr.gradientDescent()
            axs[i, j].plot(mlr.cost, label='Cost, ' + str(len(mlr.cost)) + ' iterations,  n=' + str(learningRates[i]))
            axs[i, j].set_xlabel('Iterations')
            axs[i, j].set_ylabel('Cost')
            axs[i, j].legend()
            axs[i, j].text(0.25, 0.25, 'R2: ' + str(round(mlr.computeR2(), 2)), horizontalalignment='center',
                           verticalalignment='center', transform=axs[i, j].transAxes)
    plt.show()

# 3D plot of R2 for various iterations and learning rates values with contour plot:
def plot3D(filename, predColIndexes, targetColIndex, iterations, learningRates):
    fig = plt.figure()
    ax = plt3d.Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Learning rate')
    ax.set_zlabel('R2')
    ax.set_title('R2 for various iterations and learning rates values.')
    X, Y = np.meshgrid(iterations, learningRates)
    Z = np.zeros((len(learningRates), len(iterations)))
    for i in range(len(learningRates)):
        for j in range(len(iterations)):
            mlr = MultiLinearRegression(filename, predColIndexes, targetColIndex, iterations[j], learningRates[i])
            mlr.gradientDescent()
            Z[i, j] = mlr.computeR2()
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='cool', edgecolor='black')
    ax.view_init(30, 230)
    ax.text2D(0.05, 0.95, 'R2 for various iterations and learning rates values.', transform=ax.transAxes)
    maxR2 = np.max(Z)
    maxR2Index = np.where(Z == maxR2)
    ax.text2D(0.05, 0.9, 'Max R2: ' + str(round(maxR2, 2)) + ', iterations: ' + str(iterations[maxR2Index[1][0]]) + ', learning rate: ' + str(learningRates[maxR2Index[0][0]]), transform=ax.transAxes)
    plt.show()


# Table of R2 for various iterations and learning rates values using tabulate:
def tableOfR2(filename, predColIndexes, targetColIndex, iterations, learningRates):
    table = []
    for i in range(len(learningRates)):
        row = [learningRates[i]]
        for j in range(len(iterations)):
            mlr = MultiLinearRegression(filename, predColIndexes, targetColIndex, iterations[j], learningRates[i])
            mlr.gradientDescent()
            row.append(mlr.computeR2())
        table.append(row)
    print(tabulate(table, headers=['LR \ ITR'] + iterations, tablefmt='fancy_grid'))


# Table of iterations needed for convergence for various learning rates values using tabulate:
def tableOfIterations(filename, predColIndexes, targetColIndex, learningRates):
    table = []
    for i in range(len(learningRates)):
        mlr = MultiLinearRegression(filename, predColIndexes, targetColIndex, 100000, learningRates[i]) # Iterations parameter is not important here
        mlr.gradientDescentUntilConvergence()
        table.append([learningRates[i], len(mlr.cost)])
    print(tabulate(table, headers=['Learning rate', 'Iterations'], tablefmt='fancy_grid'))



# TODO: Compute gradient descent using SK learn library and return thetas:
def computeGradientDescentUsingSKLearn(self):
    reg = LinearRegression().fit(self.data, self.Y)
    coef = reg.coef_
    return coef


# Main:
def main():
    # Files and their predictor and target columns indexes:
    files = [['dataEnergy.csv', [0, 1, 2, 3], 4], ['dataLoans.csv', [1, 4], 0]]

    for file in files:
        # Create MultiLinearRegression object, initializing a gradient descent with 1000 iterations and learning rate 0.01:
        # mlr = MultiLinearRegression(file[0], file[1], file[2], 1000, 0.01)
        # mlr.gradientDescent()

        # Print iterations amount needed to converge:
        # mlruc = MultiLinearRegression(file[0], file[1], file[2], 1000, 0.01)    # Iterations parameter is not important here.
        # cost = mlruc.gradientDescentUntilConvergence()
        # print('Iterations needed for convergence, using eta = 0.01: ', len(cost))

        # Print its R2:
        # mlr.printR2()

        # Plot its cost:
        # mlr.plotCost()

        # Plot cost for various iterations and learning rates values:
        # plotMultiple(file[0], file[1], file[2], [10, 50, 100, 1000], [0.0001, 0.001, 0.01, 0.1])

        # Plot R2 for various iterations and learning rates values:
        # plot3D(file[0], file[1], file[2], [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 2500, 3000],
        #        [0.0003, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3])

        # Table of R2 for various iterations and learning rates values:
        # tableOfR2(file[0], file[1], file[2], [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 2500, 3000], [0.0003, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3])

        # Table of iterations needed for convergence for various learning rates values:
        tableOfIterations(file[0], file[1], file[2], [0.0003, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3])


    print('Done.')


main()
