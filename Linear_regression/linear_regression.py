# %% [markdown]
# #### 1. Packages
# Import all the packages required.
# 1. __numpy__ - required for scientific computing with Python.
# 2.  __matplotlib__ - to plot graphs in Python.
# 3. __copy__ - to use deepcopy - making a totally different object
# 4. __math__ - to use mathematical functions

# %%
import numpy as np
import matplotlib.pyplot as plt 
import copy
import math

# %% [markdown]
# #### 2. Creating a simple dataset
# | Size (sqft) | Number of Bedrooms  | Number of floors | Age of  Home | Price (1000s dollars)  |   
# | ----------------| ------------------- |----------------- |--------------|-------------- |  
# | 2104            | 5                   | 1                | 45           | 460           |  
# | 1416            | 3                   | 2                | 40           | 232           |  
# | 852             | 2                   | 1                | 35           | 178           |  
# 
# Create the training dataset as shown above:
#    * x_train - creates 3 examples of housing details having size in sqft, no. of bedrooms, floors, age of the house
#    * y_train - corresponding price of the house in $1000's for training purpose
# 

# %%
def create_dataset():
    # create the training dataset
    x_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
    y_train = np.array([460,232,178])

    # get the shape of the training dataset
    shape_X = x_train.shape
    shape_Y = y_train.shape

    # store the size of the training data
    m = x_train.shape[0]

    # print the data properties
    print("Number of training examples " + str(m))
    print("Shape of X = " + str(shape_X))
    print("Shape of Y = " + str(shape_Y))
    print()

    return x_train, y_train


# %% [markdown]
# #### 3. Initializing the weight and bias vectors
# * these are initialized based on the values that are nearer to the optimal values.

# %%
def initialize_weight_and_bias():
    # assigning values
    b_init = 785.1811367994083
    w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

    # getting the shape
    shape_b_init = type(b_init)
    shape_w_init = w_init.shape
    print("Type of b = " + str(shape_b_init))
    print("Shape of w = " + str(shape_w_init))
    print()

    return b_init, w_init

# %% [markdown]
# #### 3. Compute Cost with Multiple Variables
# 1. Find the cost with the multiple variables using the following formula
# $$J(\mathbf{w},b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})^2 \tag{1}$$ 
# 2. Where 
#  $$ f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = \mathbf{w} \cdot \mathbf{x}^{(i)} + b  \tag{2} $$ 
# 

# %%
def compute_cost(X, y, w, b):
    
    #find the size of the Dataset
    m = X.shape[0]

    # initialize the cost to 0
    cost = 0.0

    # using the equation 1 and 2 compute the cost    
    for i in range(m):
        fw_b = np.dot(X[i], w) + b
        cost = cost + (fw_b - y[i])**2

    cost = cost / (2 * m)

    return cost

# %% [markdown]
# #### 4. Gradient Descent with Multiple Variables
# 1. Compute the gradient
# $$\begin{align}
# \frac{\partial J(\mathbf{w},b)}{\partial w_j}  = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})x_{j}^{(i)} \\
# \frac{\partial J(\mathbf{w},b)}{\partial b}  = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)}) \tag{3}
# \end{align}$$
# 
# 2. Perform the gradient descent
# $$\begin{align*} \text{repeat}&\text{ until convergence:} \newline\; \lbrace \newline\;
# & w_j = w_j -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial w_j} \tag{4}  \; & \text{for j = 0..n-1}\newline
# &b\ \ = b -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial b}  \newline \rbrace
# \end{align*}$$
# 

# %%
def compute_gradient(X, y, w, b):
    
    # get the number of examples and number of features
    m,n = X.shape

    # initialize gradients
    dj_dw = np.zeros((n,))
    dj_db = 0.

    # compute the gradient
    for i in range(m):
        error = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + (error * X[i, j])
        dj_db = dj_db + error
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_db, dj_dw

# %%
def gradient_descent(X, y, w_in, b_in, alpha, iterations = 1000):
    
    # array to copy all the cost values
    J_history = []

    # copy to avoid changes in the global variables
    w = copy.deepcopy(w_in)
    b = b_in

    # perform the gradient descent for given iterations 
    # to reduce the cost 
    for i in range(iterations):

        dj_db, dj_dw = compute_gradient(X, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        J_history.append(compute_cost(X, y, w, b))

        if i % math.ceil(iterations / 10) == 0:
            print(f"Iteration {i} : Cost {J_history[-1]}")

    print()

    return w, b, J_history

# %% [markdown]
# #### 5. Creating the Pipeline
# 1. Create the dataset
# 2. Initialize the weights and bias
# 3. Initialize the Hyperparameters
# 4. Perform the Gradient Descent
# 5. Compare the Actual vs Predicted cost of the house

# %%
def main():

    X_train, y_train = create_dataset()

    b_in, w_in = initialize_weight_and_bias()

    alpha = 5.0e-7

    iterations = 1000

    w_final, b_final, J_History = gradient_descent(X_train, y_train, w_in, b_in, alpha, iterations)
    
    print(f"w and b found by gradient descent: {w_final} and {b_final}")
    print()

    m,_ = X_train.shape
    for i in range(m):
        print(f"Actual: {y_train[i]}, Predicted: {np.dot(X_train[i], w_final) + b_final}")
    print()


# %%
if __name__=="__main__":
    main()


