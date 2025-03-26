# Iris Dataset Analysis
#
# Author: Anna Lozenko


import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn import datasets

iris = skl.datasets.load_iris()


sl = iris["data"][:, 0]
sw = iris["data"][:, 1]
pl = iris["data"][:, 2]
pw = iris["data"][:, 3]

# 1. SUMMARY OF EACH VARIABLE TO A SINGLE TXT FILE





# 2. HISTOGRAM PLOT OF EACH VARIABLE TO PNG FILES

# create a histogram plot of Sepal Length
plt.hist(sl, color='blue', edgecolor='black');
plt.title('Sepal Length')
plt.xlabel('Sepal Length')
plt.ylabel('Frequency')
plt.savefig("Sepal Length Histogram.png")
plt.show()

# create a histogram plot of Sepal Width
plt.hist(sw, color='green', edgecolor='black');
plt.title('Sepal Width')
plt.xlabel('Sepal Width')
plt.ylabel('Frequency')
plt.savefig("Sepal Width Histogram.png")
plt.show()

# create a histogram plot of Petal Length
plt.hist(pl, color='red', edgecolor='black');
plt.title('Petal Length')
plt.xlabel('Petal Length')
plt.ylabel('Frequency')
plt.savefig("Petal Length Histogram.png")
plt.show()

# create a histogram plot of Petal Width
plt.hist(pw, color='purple', edgecolor='black');
plt.title('Petal Width')
plt.xlabel('Petal Width')
plt.ylabel('Frequency')
plt.savefig("Petal Width Histogram.png")
plt.show()





# 3. SCATTER PLOT OF EACH PAIR OF VARIABLES




# 4. ANY OTHER APPROPRIATE ANALYSIS