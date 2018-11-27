from scipy import stats
import numpy as np


# This method computes entropy for information gain
def entropy(class_y):
    # Input:            
    #   class_y         : list of class labels (0's and 1's)
    
    # TODO: Compute the entropy for a list of classes
    #
    # Example:
    #    entropy([0,0,0,1,1,1,1,1,1]) = 0.92

    num_zero = 0
    num_one = 0

    for x in class_y:
        if x == 0:
            num_zero += 1
        else:
            num_one += 1

    if (num_zero == 0) or (num_one == 0):
        entropy = 0
    else:
        part_zero = -(num_zero) / float(len(class_y)) * np.log2(num_zero / float(len(class_y)))
        part_one = -(num_one) / float(len(class_y)) * np.log2(num_one / float(len(class_y)))
        entropy = part_zero + part_one

    return entropy


def partition_classes(X, y, split_attribute, split_val):
    # Inputs:
    #   X               : data containing all attributes
    #   y               : labels
    #   split_attribute : column index of the attribute to split on
    #   split_val       : either a numerical or categorical value to divide the split_attribute
    
    # TODO: Partition the data(X) and labels(y) based on the split value - BINARY SPLIT.
    # 
    # You will have to first check if the split attribute is numerical or categorical    
    # If the split attribute is numeric, split_val should be a numerical value
    # For example, your split_val could be the mean of the values of split_attribute
    # If the split attribute is categorical, split_val should be one of the categories.   
    #
    # You can perform the partition in the following way
    # Numeric Split Attribute:
    #   Split the data X into two lists(X_left and X_right) where the first list has all
    #   the rows where the split attribute is less than or equal to the split value, and the 
    #   second list has all the rows where the split attribute is greater than the split 
    #   value. Also create two lists(y_left and y_right) with the corresponding y labels.
    #
    # Categorical Split Attribute:
    #   Split the data X into two lists(X_left and X_right) where the first list has all 
    #   the rows where the split attribute is equal to the split value, and the second list
    #   has all the rows where the split attribute is not equal to the split value.
    #   Also create two lists(y_left and y_right) with the corresponding y labels.

    '''
    Example:
    
    X = [[3, 'aa', 10],                 y = [1,
         [1, 'bb', 22],                      1,
         [2, 'cc', 28],                      0,
         [5, 'bb', 32],                      0,
         [4, 'cc', 32]]                      1]
    
    Here, columns 0 and 2 represent numeric attributes, while column 1 is a categorical attribute.
    
    Consider the case where we call the function with split_attribute = 0 and split_val = 3 (mean of column 0)
    Then we divide X into two lists - X_left, where column 0 is <= 3  and X_right, where column 0 is > 3.
    
    X_left = [[3, 'aa', 10],                 y_left = [1,
              [1, 'bb', 22],                           1,
              [2, 'cc', 28]]                           0]
              
    X_right = [[5, 'bb', 32],                y_right = [0,
               [4, 'cc', 32]]                           1]

    Consider another case where we call the function with split_attribute = 1 and split_val = 'bb'
    Then we divide X into two lists, one where column 1 is 'bb', and the other where it is not 'bb'.
        
    X_left = [[1, 'bb', 22],                 y_left = [1,
              [5, 'bb', 32]]                           0]
              
    X_right = [[3, 'aa', 10],                y_right = [1,
               [2, 'cc', 28],                           0,
               [4, 'cc', 32]]                           1]
               
    ''' 
    
    X_left = []
    X_right = []
    
    y_left = []
    y_right = []

    if isinstance(X[0][split_attribute], (int, float, complex)) and isinstance(split_val, (int, float, complex)):
        for x in X:
            if x[split_attribute] <= split_val:
                X_left.append(x)
                y_left.append(y[X.index(x)])
            if x[split_attribute] > split_val:
                X_right.append(x)
                y_right.append(y[X.index(x)])

    if isinstance(X[0][split_attribute]) and isinstance(split_val):
        for x in X:
            if x[split_attribute] == split_val:
                X_left.append(x)
                y_left.append(y[X.index(x)])
            else:
                X_right.append(x)
                y_right.append(y[X.index(x)])
    
    return (X_left, X_right, y_left, y_right)

    
def information_gain(previous_y, current_y):
    # Inputs:
    #   previous_y: the distribution of original labels (0's and 1's)
    #   current_y:  the distribution of labels after splitting based on a particular
    #               split attribute and split value
    
    # TODO: Compute and return the information gain from partitioning the previous_y labels
    # into the current_y labels.
    # You will need to use the entropy function above to compute information gain
    # Reference: http://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15381-s06/www/DTs.pdf
    
    """
    Example:
    
    previous_y = [0,0,0,1,1,1]
    current_y = [[0,0], [1,1,1,0]]
    
    info_gain = 0.45915
    """

    info_gain = 0

    info_gain = entropy(previous_y)

    for temp in current_y:
        info_gain -= (float)(len(temp) * entropy(temp)) / len(previous_y)

    return info_gain


def info_best_split_val(data, split_attribute):
    # Function to find the best split value for a certain split_attribute according to the splitting information.

    best_info = 0
    best_val = 0

    y = [e[-1] for e in data]
    X = [e[:-1] for e in data]

    # Find the split_value list for each row of the data
    val_list = []
    for line in X:
        if not (line[split_attribute] in val_list):
            val_list.append(line[split_attribute])

    # Find the largest information gain within the split value list
    for val in val_list:
        y_left = partition_classes(X, y, split_attribute, val)[2]
        y_right = partition_classes(X, y, split_attribute, val)[3]
        current_y = [y_left, y_right]
        new_info = information_gain(y, current_y)

        if new_info > best_info:
            best_info = new_info
            best_val = val

    return (best_info, best_val)


def partition_data(data, split_attribute, split_val):
    # Function to split the data set into left and right part according to its splitting attribute and splitting value.

    left = []
    right = []

    # Numeric split attribute
    if isinstance(data[0][split_attribute], (int, float, complex)) and isinstance(split_val, (int, float, complex)):
        for line in data:
            if line[split_attribute] <= split_val:
                left.append(line)
            else:
                right.append(line)

    # Categorical split attribute
    if isinstance(data[0][split_attribute]) and isinstance(split_val):
        for line in data:
            if line[split_attribute] == split_val:
                left.append(line)
            else:
                right.append(line)

    return (left, right)