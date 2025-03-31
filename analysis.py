# Iris Dataset Analysis
#
# Author: Anna Lozenko


import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn import datasets

iris = skl.datasets.load_iris()

sl = iris["data"][:, 0] # select sepal length column
sw = iris["data"][:, 1] # select sepal width column
pl = iris["data"][:, 2] # select petal length column
pw = iris["data"][:, 3] # select petal width column


# 1. SUMMARY OF EACH VARIABLE TO A SINGLE TXT FILE

# sepal length analysis:
sl_head = sl[:10] # access the first 10 rows of sepal length 
sl_tail = sl[145:] # access the last 10 rows of sepal length 
sl_mean = np.mean(sl) # compute the mean value of sepal length
sl_sd = np.std(sl) # compute the standard deviation of sepal length
sl_median = np.median(sl) # compute the median of sepal length
sl_min = np.min(sl) # compute the minimimm value of sepal length
sl_max = np.max(sl) # compute the maximum value of sepal length

sepal_length_summary = f"SEPAL LENGTH ANALYSIS:\n\nThese are the first 10 values of sepal length column: \t\t{sl_head}.\n\nThese are the last 10 values of sepal length column: \t\t{sl_tail}.\n\nThis is the mean of sepal length variable: \t\t\t{sl_mean}.\n\nThis is the median of sepal length variable: \t\t\t{sl_median}.\n\nThis is the standard deviation of sepal length variable: \t{sl_sd}.\n\nThis is the minimum value of sepal length: \t\t\t{sl_min}.\n\nThis is the maximum value of sepal length: \t\t\t{sl_max}.\n\n\n\n"


# sepal width analysis:
sw_head = sw[:10] # access the first 10 rows of sepal width 
sw_tail = sw[145:] # access the last 10 rows of sepal width
sw_mean = np.mean(sw) # compute the mean value of sepal width
sw_sd = np.std(sw) # compute the standard deviation of sepal width
sw_median = np.median(sw) # compute the median of sepal width
sw_min = np.min(sw) # compute the minimimm value of sepal width
sw_max = np.max(sw) # compute the maximum value of sepal width

sepal_width_summary = f"SEPAL WIDTH ANALYSIS:\n\nThese are the first 10 values of sepal width column: \t\t{sw_head}.\n\nThese are the last 10 values of sepal width column: \t\t{sw_tail}.\n\nThis is the mean of sepal width variable: \t\t\t{sw_mean}.\n\nThis is the median of sepal width variable: \t\t\t{sw_median}.\n\nThis is the standard deviation of sepal width variable: \t{sw_sd}.\n\nThis is the minimum value of sepal width: \t\t\t{sw_min}.\n\nThis is the maximum value of sepal width: \t\t\t{sw_max}.\n\n\n\n"

# petal length analysis:
pl_head = pl[:10] # access the first 10 rows of petal length 
pl_tail = pl[145:] # access the last 10 rows of petal length 
pl_mean = np.mean(pl) # compute the mean value of petal length
pl_sd = np.std(pl) # compute the standard deviation of petal length
pl_median = np.median(pl) # compute the median of petal length
pl_min = np.min(pl) # compute the minimimm value of petal length
pl_max = np.max(pl) # compute the maximum value of petal length

petal_length_summary = f"PETAL LENGTH ANALYSIS:\n\nThese are the first 10 values of petal length column: \t\t{pl_head}.\n\nThese are the last 10 values of petal length column: \t\t{pl_tail}.\n\nThis is the mean of petal length variable: \t\t\t{pl_mean}.\n\nThis is the median of petal length variable: \t\t\t{pl_median}.\n\nThis is the standard deviation of petal length variable: \t{pl_sd}.\n\nThis is the minimum value of petal length: \t\t\t{pl_min}.\n\nThis is the maximum value of petal length: \t\t\t{pl_max}.\n\n\n\n"

# petal width analysis:
pw_head = pw[:10] # access the first 10 rows of petal width 
pw_tail = pw[145:] # access the last 10 rows of petal width
pw_mean = np.mean(pw) # compute the mean value of petal width
pw_sd = np.std(pw) # compute the standard deviation of petal width
pw_median = np.median(pw) # compute the median of petal width
pw_min = np.min(pw) # compute the minimimm value of petal width
pw_max = np.max(pw) # compute the maximum value of petal width

petal_width_summary = f"PETAL WIDTH ANALYSIS:\n\nThese are the first 10 values of petal width column: \t\t{pw_head}.\n\nThese are the last 10 values of petal width column: \t\t{pw_tail}.\n\nThis is the mean of petal width variable: \t\t\t{pw_mean}.\n\nThis is the median of petal width variable: \t\t\t{pw_median}.\n\nThis is the standard deviation of petal width variable: \t{pw_sd}.\n\nThis is the minimum value of petal width: \t\t\t{pw_min}.\n\nThis is the maximum value of petal width: \t\t\t{pw_max}.\n\n\n\n"

# create a text file to store the output of iris variables analysis
with open ("Iris Dataset Variables Analysis.txt", "w") as f:
    f.write(sepal_length_summary)
    f.write(sepal_width_summary)
    f.write(petal_length_summary)
    f.write(petal_width_summary)


# 2. HISTOGRAM PLOT OF EACH VARIABLE TO PNG FILES

def histogram_creator (x, title = "", xlab = "", ylab = "",  color = "blue", edgecolor = "black", grid = False, save = False, **kwargs):
    plt.xlabel(f"{xlab}")
    plt.ylabel(f"{ylab}")
    title_font= {"family": "Georgia", "color" : "darkblue", "size" : 18}
    plt.title (f"{title}", fontdict= title_font)
    plt.hist(x, edgecolor = edgecolor, color = color, **kwargs)
    if grid == True:
        plt.grid(alpha = 0.2)
    if save == True:
        plt.savefig(f"{title}.png")
    plt.show()

histogram_creator(sl, title = "Sepal Length Histogram", xlab = "Sepal Length", ylab= "Frequency", grid = True, save = True)
histogram_creator(sw, title = "Sepal Width Histogram", xlab = "Sepal Width", ylab = "Frequency", color = "green", grid = True, save = True)
histogram_creator(pl, title = "Petal Length Histogram", xlab = "Petal Length", ylab = "Frequency", color = "red", grid = True, save = True)
histogram_creator(pw, title = "Petal Width Histogram", xlab = "Petal Width", ylab = "Frequency", color = "orange", grid = True, save = True)


# 3. SCATTER PLOT OF EACH PAIR OF VARIABLES

# scatterplots of each pair of variable highlighting the three species in different colors
species = iris["target"] 

def scatterplot_creator (x, y, xlab = "", ylab = "", c = "red", title = "", grid = False, save = False, fitline = False, **kwargs):
    plt.scatter(x, y , c = c, **kwargs)
    plt.xlabel(f"{xlab}")
    plt.ylabel(f"{ylab}")
    title_font= {"family": "Georgia", "color" : "darkblue", "size" : 18}
    plt.title(f"{title}", fontdict= title_font)
    if grid == True:
        plt.grid(alpha = 0.2)
    if save == True:
        plt.savefig(f"{title}.png")
    if fitline == True:
        m, c = np.polyfit(x, y, deg = 1)
        plt.plot(x, m*x + c, color = "red")
        plt.legend (["Data", "Line of Best Fit"])
    plt.show()

scatterplot_creator (sl, sw, xlab= "Sepal Length (cm)", ylab= "Sepal Width (cm)", title= "Sepal Length vs Sepal Width", c = species)
scatterplot_creator(sl, pl, xlab= "Sepal Length (cm)", ylab= "Petal Length (cm)", title= "Sepal Length vs Petal Length", c= species)
scatterplot_creator(sl, pw, xlab= "Sepal Length (cm)", ylab= "Petal Width (cm)", title= "Sepal Length vs Sepal Width", c= species)
scatterplot_creator(pl, sw, xlab= "Petal Length (cm)", ylab= "Sepal Width (cm)", title= "Petal Length vs Sepal Width", c= species)
scatterplot_creator(pw, sw, xlab= "Petal Width", ylab= "Sepal Width (cm)", title= "Petal Width vs Sepal Width", c= species)
scatterplot_creator(pl, pw, xlab= "Petal Length (cm)", ylab= "Petal Width (cm)", title= "Petal Length vs Petal Width", c= species)


# 4. ANY OTHER APPROPRIATE ANALYSIS




