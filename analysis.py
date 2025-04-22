# IRIS DATASET ANALYSIS
#
# Author: Anna Lozenko

# import all the necessary libraries
import numpy as np # https://numpy.org/
import matplotlib.pyplot as plt # https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.html
import sklearn as skl # https://scikit-learn.org/stable/
from sklearn import datasets 
import pandas as pd # https://pandas.pydata.org/

# load the Iris dataset from sklearn datasets library
iris = skl.datasets.load_iris(as_frame= True) # load the "data" from the dataset as a Pandas DataFrame (https://scikit-learn.org/1.4/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris)

# Indexing columns in Pandas DataFrame (https://pandas.pydata.org/docs/user_guide/indexing.html)
sl = iris.data.iloc[:, 0] # sepal length
sw = iris.data.iloc[:, 1] # sepal width
pl = iris.data.iloc[:, 2] # petal length
pw = iris.data.iloc[:, 3] # petal width


# 1. SUMMARY OF EACH VARIABLE TO A SINGLE TXT FILE

# create a function that performs the feature analysis and prints out the results
def feature_analysis (x):
    mean = np.mean(x) # mean
    median = np.median(x) # median
    std = np.std(x) # standard deviation
    min = np.min(x) # minum value
    max = np.max(x) # maximum value
    return(f"Mean: {mean:.2f}.\nMedian: {median:.2f}.\nStandard deviation: {std:.2f}.\nMinimum value: {min:.2f}.\nMaximum value: {max:.2f}.")

# store analysis results
sl_analysis = feature_analysis(sl) # sepal length analysis
sw_analysis = feature_analysis(sw) # sepal width analysis
pl_analysis = feature_analysis(pl) # petal length analysis
pw_analysis = feature_analysis(pw) # petal width analysis

# creation of a DataFrame with features summary  (https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)
features_summary= pd.DataFrame({
    "Mean" : iris["data"].mean(),
    "Median" : iris["data"].median(),
    "Standard Deviation" : iris["data"].std(),
    "Minimum" : iris["data"].min(),
    "Maximum" : iris["data"].max()
})


# create a text file to store the output of iris variables analysis
with open ("Iris Dataset Variables Analysis.txt", "w") as f:
    f.write(f"IRIS SEPAL LENGTH ANALYSIS:\n{sl_analysis}\n\n\n")
    f.write(f"IRIS SEPAL WIDTH ANALYSIS:\n{sw_analysis}\n\n\n")
    f.write(f"IRIS PETAL LENGTH ANALYSIS:\n{pl_analysis}\n\n\n")
    f.write(f"IRIS PETAL WIDTH ANALYSIS:\n{pw_analysis}\n\n\n")
    f.write(f"FEATURES SUMMARY:\n{features_summary}")



# 2. HISTOGRAM PLOT OF EACH VARIABLE TO PNG FILES

# define a function that create histogram plots 
def histogram_creator (x, title = "", xlab = "", ylab = "",  color = "blue", edgecolor = "black", grid = False, save = False, **kwargs): # define the arguments of the function
    plt.xlabel(f"{xlab}") # define the x axes label
    plt.ylabel(f"{ylab}") # define the y axes label
    title_font= {"family": "sans-serif", "color" : "black", "size" : 16, "weight" : "bold"} # create a different font for the title of the plot (https://how.dev/answers/how-to-set-font-properties-for-title-and-labels-in-matplotlib, https://matplotlib.org/stable/users/explain/text/text_props.html)
    plt.title (f"{title}", fontdict= title_font) # define the title of the plot
    plt.hist(x, edgecolor = edgecolor, color = color, **kwargs) # create the plot with Matplotlib (https://matplotlib.org/, https://www.w3schools.com/python/matplotlib_histograms.asp)
    if grid == True: # define how the grid will be visualized by default
        plt.grid(alpha = 0.2) # grid transparency set to 80%
    if save == True: # define how the image of the plot will be saved by default
        plt.savefig(f"{title}.png")
    plt.show()

histogram_creator(sl, title = "Sepal Length Histogram", xlab = "Sepal Length", ylab= "Frequency", grid = True, save = True) # save histogram of sepal length
histogram_creator(sw, title = "Sepal Width Histogram", xlab = "Sepal Width", ylab = "Frequency", color = "green", grid = True, save = True) # save histogram of sepal width
histogram_creator(pl, title = "Petal Length Histogram", xlab = "Petal Length", ylab = "Frequency", color = "red", grid = True, save = True) # save histogram of petal lenght
histogram_creator(pw, title = "Petal Width Histogram", xlab = "Petal Width", ylab = "Frequency", color = "orange", grid = True, save = True) # save histogram of petal width



# 3. SCATTERPLOT OF EACH PAIR OF VARIABLES
# https://en.wikipedia.org/wiki/Scatter_plot 
# scatterplots of each pair of variable highlighting the three species in different colors
species = iris["target"] 

# define a function that creates scatterplots
def scatterplot_creator (x, y, xlab = "", ylab = "", c = "red", title = "", grid = False, save = False, fitline = False, **kwargs): # define the arguments of the function
    plt.scatter(x, y , c = c, **kwargs) # create the plot (https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html)
    plt.xlabel(f"{xlab}") # define the x axes label
    plt.ylabel(f"{ylab}") # define the y axes label
    title_font= {"family": "sans-serif", "color" : "black", "size" : 16, "weight" : "bold"} # use a custom font for the title
    plt.title(f"{title}", fontdict= title_font) # define the title of the plot
    if grid == True: # define how grid will appear by default
        plt.grid(alpha = 0.2) # grid transparency set to 80%
    if save == True: # define how images of plots will be saved by default
        plt.savefig(f"{title}.png")
    if fitline == True: # calculate the parameters of the regression line to examine the linear relationship between variables (https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html, https://www.geeksforgeeks.org/line-of-best-fit/ )
        m, c = np.polyfit(x, y, deg = 1) # slope (m), intercept (c)
        plt.plot(x, m*x + c, color = "red") # plot of the regression line
        plt.legend (["Data", "Line of Best Fit"])
    plt.show()

scatterplot_creator (sl, sw, xlab= "Sepal Length (cm)", ylab= "Sepal Width (cm)", title= "Sepal Length vs Sepal Width", c = species) # scatterplot of sepal length vs sepal width
scatterplot_creator(sl, pl, xlab= "Sepal Length (cm)", ylab= "Petal Length (cm)", title= "Sepal Length vs Petal Length", c= species) # scatterplot of sepal length vs petal length
scatterplot_creator(sl, pw, xlab= "Sepal Length (cm)", ylab= "Petal Width (cm)", title= "Sepal Length vs Sepal Width", c= species) # scatterplot of sepal length vs petal width
scatterplot_creator(pl, sw, xlab= "Petal Length (cm)", ylab= "Sepal Width (cm)", title= "Petal Length vs Sepal Width", c= species) # scatterplot of petal length vs sepal width
scatterplot_creator(pw, sw, xlab= "Petal Width", ylab= "Sepal Width (cm)", title= "Petal Width vs Sepal Width", c= species) # scatterplot of petal width vs sepal width
scatterplot_creator(pl, pw, xlab= "Petal Length (cm)", ylab= "Petal Width (cm)", title= "Petal Length vs Petal Width", c= species) # scatterplot of petal length vs petal width



# 4. ANY OTHER APPROPRIATE ANALYSIS

# BOXPLOTS (https://en.wikipedia.org/wiki/Box_plot  , https://www.khanacademy.org/math/statistics-probability/summarizing-quantitative-data/box-whisker-plots/a/box-plot-review)
# define the species based on target number
setosa = iris["data"][iris["target"] == 0]
versicolor = iris["data"][iris["target"] == 1]
virginica = iris["data"][iris["target"] == 2]

# define a function that creates boxplots
def boxplots_creator (x, title = "", xlab = "", ylab= "", ticklabs = "", grid = False, save = False, **kwargs):
    fig, ax = plt.subplots() # option to create multiple plots on the same figure (https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html)
    ax.boxplot(x, patch_artist= True) # create boxplot filled with color
    title_font= {"family": "sans-serif", "color" : "black", "size" : 16, "weight" : "bold"} # create a custom font for the title
    ax.set_ylabel (f"{ylab}", fontsize = 12) # set the y axes label
    ax.set_xlabel(f"{xlab}", fontsize = 12) # set the x axes label
    ax.set_xticklabels(ticklabs) # create labels for each tick mark on the x axes
    ax.set_title (f"{title}", fontdict = title_font) # set a title 
    if grid == True: # define how grid will be visualized by deafult
        ax.grid(axis = "y", alpha = 0.5, linestyle = "--") # grid transparency set on 50%, grid made with dashed line, located on the y axes
    if save == True:  # define how the image will be savedby default
        plt.savefig(f"{title}.png")
    plt.show()

# define petal lengths of the different species
setosa_pl = setosa["petal length (cm)"]
versicolor_pl = versicolor["petal length (cm)"]
virginica_pl = virginica ["petal length (cm)"]
boxplots_creator(setosa_pl, title= "Iris Setosa Petal Length", xlab= "Iris setosa", ylab="Petal Length (cm)", grid = True) # setosa petal length
boxplots_creator(versicolor_pl, title= "Iris Versicolor Petal Length", xlab= "Iris versicolor", ylab="Petal Length (cm)", grid = True ) # versicolor petal length
boxplots_creator(virginica_pl, title= "Iris Virginica Petal Length", xlab= "Iris virginica", ylab="Petal Length (cm)", grid = True ) # virginica petal length
boxplots_creator([setosa_pl, versicolor_pl, virginica_pl], ticklabs= ["setosa", "versicolor", "virginica"], xlab= "Species", ylab= "Petal Length (cm)", title = "Petal Lengths of All Species", grid = True) # all species in one plot

# define the petal width of the different species
setosa_pw = setosa["petal width (cm)"] 
versicolor_pw = versicolor["petal width (cm)"]
virginica_pw = virginica["petal width (cm)"]
boxplots_creator(setosa_pw, title= "Iris Setosa Petal Width", xlab= "Iris setosa", ylab="Petal Width (cm)", grid = True ) # setosa petal width
boxplots_creator(versicolor_pw, title= "Iris Versicolor Petal Width", xlab= "Iris versicolor", ylab="Petal Width (cm)", grid = True ) # versicolor petal width
boxplots_creator(virginica_pw, title= "Iris Virginica Petal Width", xlab= "Iris virginica", ylab="Petal Width (cm)", grid = True ) # virginica petal width
boxplots_creator([setosa_pw, versicolor_pw, virginica_pw], ticklabs= ["setosa", "versicolor", "virginica"], xlab= "Species", ylab= "Petal Width (cm)", title = "Petal Width of All Species", grid = True) # all species in one plot

# define the sepal length of the different species
setosa_sl = setosa["sepal length (cm)"]
versicolor_sl = versicolor["sepal length (cm)"]
virginica_sl = virginica["sepal length (cm)"]
boxplots_creator(setosa_sl, title= "Iris Setosa Sepal Length", xlab= "Iris setosa", ylab="Sepal Length (cm)", grid = True ) # setosa sepal length
boxplots_creator(versicolor_sl, title= "Iris Versicolor Sepal Length", xlab= "Iris versicolr", ylab="Sepal Length (cm)", grid = True ) # versicolor sepal length
boxplots_creator(virginica_sl, title= "Iris Virginica Sepal Length", xlab= "Iris virginica", ylab="Sepal Length (cm)", grid = True ) # virginica sepal length
boxplots_creator([setosa_sl, versicolor_sl, virginica_sl], ticklabs= ["setosa", "versicolor", "virginica"], xlab= "Species", ylab= "Sepal Length (cm)", title = "Sepal Lengths of All Species", grid = True) # all species in one plot

# define the sepal width of the different species
setosa_sw = setosa["sepal width (cm)"]
versicolor_sw = versicolor["sepal width (cm)"]
virginica_sw = virginica["sepal width (cm)"]
boxplots_creator(setosa_sw, title= "Iris Setosa Sepal Width", xlab= "Iris setosa", ylab="Sepal Width (cm)", grid = True ) # setosa sepal width
boxplots_creator(versicolor_sw, title= "Iris Versicolor Sepal Width", xlab= "Iris versicolor", ylab="Sepal Width (cm)", grid = True ) # versicolor sepal width
boxplots_creator(virginica_sw, title= "Iris Virginica Sepal Width", xlab= "Iris virginica", ylab="Sepal Width (cm)", grid = True ) # virginica sepal width
boxplots_creator([setosa_sw, versicolor_sw, virginica_sw], ticklabs= ["setosa", "versicolor", "virginica"], xlab= "Species", ylab= "Sepal Width (cm)", title = "Sepal Width of All Species", grid = True) # all species in one plot
