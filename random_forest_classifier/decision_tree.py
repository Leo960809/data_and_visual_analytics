from util import *
import numpy as np 
import ast
import copy


class DecisionNode:

    def __init__(self):
        self.label = -1
        self.attribute = 0
        self.best_split_val = 0
        self.depth = 0
        self.leaf = True
        self.left = None
        self.right = None


class DecisionTree(object):

    def __init__(self):
        # Initializing the tree as an empty dictionary or list, as preferred
        #self.tree = []
        self.tree = {}


    def build_tree(self, decision_node, data, feature_list, threshold=0, min_info=1e-8):
        # Build a tree from the given root

        # Initiate the optimal split attribute, split value and information gain
        best_split_val = 0
        best_split_attribute = 0
        best_info_gain = 0

        # Extract the label column. Label_list is not a numpy list
        label_list = [e[-1] for e in data]

        # If all the samples are label 0 or label 1, then the entropy is zero:
        if entropy(label_list):
            decision_node.leaf = True
            decision_node.label = label_list[0]
            return decision_node.label

        # set up a threshold for the minimum number of data being classified to each node
        if threshold > 0 and len(data) <= threshold:
            decision_node.leaf = True
            if len([e for e in label_list if e == 1]) >= len([e for e in label_list if e == 0]):
                decision_node.label = 1
            else:
                decision_node.label = 0
            return decision_node.label

        # In the case when running out of features:
        if not feature_list:
            decision_node.leaf = True
            if len([e for e in label_list if e == 1]) >= len([e for e in label_list if e == 0]):
                decision_node.label = 1
            else:
                decision_node.label = 0
            return decision_node.label

        # # Generates a random sample from self.n_feature given 1-D array,
        # # sampling without replacement

        # Randomly choose the total number of features in each Tree. From root node.
        n_attribute = np.random.choice(self.n_feature, 1, True)

        # Choose which features are used in this tree randomly. in root
        random_attribute = np.random.choice(self.n_feature, n_attribute, False)

        # Find the best attribute and its split value according to the calculated information gain
        for split_attribute in random_attribute:
            info_gain, split_val = info_best_split_val(data, split_attribute)

            if info_gain > best_info_gain:
                best_split_attribute = split_attribute
                best_info_gain = info_gain
                best_split_val = split_val

        # min_info is the min info gain to execute the split operation.
        if best_info_gain > min_info:
            left, right = partition_data(data, best_split_attribute, best_split_val)
            decision_node.leaf = False
            decision_node.attribute = best_split_attribute
            decision_node.best_split_val = best_split_val

            decision_node.left = DecisionNode()  # Left splitted tree
            decision_node.right = DecisionNode()  # Right splitted tree

            # Perform recursion of Build_tree function for both left splitted tree and right splitted tree
            self.build_tree(decision_node.left, left, feature_list, 0)
            self.build_tree(decision_node.right, right, feature_list, 0)

        # The tree stops spitting if the best information gain is equal or smaller than the minimum information gain
        else:
            decision_node.leaf = True
            # Find the Majority Label
            if len([e for e in label_list if e == 1]) >= len([e for e in label_list if e == 0]):
                decision_node.label = 1
            else:
                decision_node.label = 0


    def learn(self, X, y):
        # TODO: Train the decision tree (self.tree) using the the sample X and labels y
        # You will have to make use of the functions in utils.py to train the tree
        
        # One possible way of implementing the tree:
        #    Each node in self.tree could be in the form of a dictionary:
        #       https://docs.python.org/2/library/stdtypes.html#mapping-types-dict
        #    For example, a non-leaf node with two children can have a 'left' key and  a 
        #    'right' key. You can add more keys which might help in classification
        #    (eg. split attribute and split value)

        self.X = X
        self.y = y
        n_feature = np.asarray(X).shape[1]  # Find the number of features
        self.n_feature = n_feature
        feature_list = range(n_feature)  # Create a list for feature number

        # Combine X with the its corresponding label
        data = copy.deepcopy(X)
        count = 0
        for e in data:
            e.append(y[count])
            count = count + 1

        self.tree["root"] = DecisionNode()  # Create the root object
        root = self.tree["root"]

        self.build_tree(root, data, feature_list, 0)  # build the tree
        self.root = root


    def classify(self, record):
        # TODO: classify the record using self.tree and return the predicted label

        node = self.root
        while not node.leaf:
            node_attribute = node.attribute
            if isinstance(record[node_attribute], (int, float, complex)):
                if (record[node_attribute] <= node.best_split_val):
                    node = node.left
                else:
                    node = node.right
            else:
                if (record[node_attribute] == node.best_split_val):
                    node = node.left
                else:
                    node = node.right
        return int(node.label)