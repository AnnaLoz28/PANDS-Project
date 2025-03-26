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