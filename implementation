#formerly a jupyter notebook file


# Load the data and libraries
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

def gaussian_mech_vec(vec, sensitivity, epsilon, delta):
    return [v + np.random.normal(loc=0, scale=sensitivity * np.sqrt(2*np.log(1.25/delta)) / epsilon)
            for v in vec]

food = pd.read_csv('https://raw.githubusercontent.com/fflanagan89/CS211FinalProject/main/indian_food.csv')



#DATA CLEANING

#remove rows that are not spicy or sweet
food = food.drop(food[food.flavor_profile == 'sour'].index)
food = food.drop(food[food.flavor_profile == '-1'].index)
food = food.drop(food[food.flavor_profile == 'bitter'].index)

food = food.drop(food[food.region == '-1'].index)

#drop name, state, and ingredients columns 
food = food.drop(columns=['name', 'state', 'ingredients', 'cook_time', 'prep_time'])

#label encode diet and flavor profile
#vegetarian and sweet are 1 
#non vegetarian and spicy are 0
food['diet'] = food['diet'].map({'vegetarian': 1, 'non vegetarian': 0})
food['flavor_profile'] = food['flavor_profile'].map({'sweet': 1, 'spicy': -1})


#"One Hot encoding"
food = pd.get_dummies(food, columns=["course", "region"], prefix=["course", "region"])




#split dataframe into x and y
#where y is flavor_profile and x is every other column

flavor = food[['flavor_profile']].copy()


food = food.drop(columns=['flavor_profile'])




#DATAFRAMES -> NUMPY
#SPLIT ARRAYS INTO TEST AND TRAINING SETS

X = food.to_numpy(dtype = 'float')


y = flavor.to_numpy(dtype = 'float')

y = y.ravel()
#split the data into training and test sets

training_size = int(X.shape[0] * 0.8)
X_train = X[:training_size]
X_test = X[training_size:]

y_train = y[:training_size]
y_test = y[training_size:]



#FUNCTIONS

# This is the gradient of the logistic loss
# The gradient is a vector that indicates the rate of change of the loss in each direction
def gradient(theta, xi, yi):
    exponent = yi * (xi.dot(theta))
    return - (yi*xi) / (1+np.exp(exponent))

def avg_grad(theta, X, y):
    grads = [gradient(theta, xi, yi) for xi, yi in zip(X, y)]
    return np.mean(grads, axis=0)

# Prediction: take a model (theta) and a single example (xi) and return its predicted label
def predict(xi, theta, bias=0):
    label = np.sign(xi @ theta + bias)
    return label

def accuracy(theta):
    return np.sum(predict(X_test, theta) == y_test)/X_test.shape[0]
    
  
  

#NOISY GRADIENT DESCENT IMPLEMENTATION


def L2_clip(v, b):
    norm = np.linalg.norm(v, ord=2)
    
    if norm > b:
        return b * (v / norm)
    else:
        return v

def noisy_gradient_descent(iterations, epsilon, delta):
     
    theta = np.zeros(X_train.shape[1])

    for i in range(iterations):
        gradients = [gradient(theta, xi, yi) for xi, yi in zip(X, y)]
        

        clipped_gradients = [L2_clip(grad, 5) for grad in gradients]
        
        clipped_mean_gradient = np.mean(clipped_gradients, axis=0)
        
        noisy_g = gaussian_mech_vec(clipped_mean_gradient,5/len(X_train),epsilon, delta)
        
        theta = theta - noisy_g
        

    return theta

theta = noisy_gradient_descent(10, 1.0, 1e-5)
print('Accuracy at 10 iterations epsilon 1.0:', accuracy(theta))

theta = noisy_gradient_descent(100, 1.0, 1e-5)
print('Accuracy at 100 iterations epsilon 1.0:', accuracy(theta))

theta = noisy_gradient_descent(1000, 1.0, 1e-5)
print('Accuracy at 1000 iterations epsilon 1.0:', accuracy(theta))
print("-----------------------------------------------------------")
theta = noisy_gradient_descent(100, .5, 1e-5)
print('Accuracy at 100 iterations epsilon .5:', accuracy(theta))

theta = noisy_gradient_descent(10, .5, 1e-5)
print('Accuracy at 10 iterations epsilon .5:', accuracy(theta))
print("-----------------------------------------------------------")
theta = noisy_gradient_descent(100, .35, 1e-5)
print('Accuracy at 100 iterations epsilon .35:', accuracy(theta))

theta = noisy_gradient_descent(100, .25, 1e-5)
print('Accuracy at 100 iterations epsilon .25:', accuracy(theta))
print("-----------------------------------------------------------")
theta = noisy_gradient_descent(100, .1, 1e-5)
print('Accuracy at 100 iterations epsilon .1:', accuracy(theta))
theta = noisy_gradient_descent(10, .1, 1e-5)
print('Accuracy at 10 iterations epsilon .1:', accuracy(theta))
print("-----------------------------------------------------------")
theta = noisy_gradient_descent(100, .01, 1e-5)
print('Accuracy at 100 iterations epsilon .01:', accuracy(theta))


#COMPARE

from sklearn.linear_model import LogisticRegression

def train_model():
    m = LogisticRegression()
    m.fit(X,y)
    return m

model = train_model()
print('Model coefficients:', model.coef_[0])
print('Model accuracy:', np.sum(model.predict(X_test) == y_test)/X_test.shape[0])

