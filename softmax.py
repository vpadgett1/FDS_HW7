import numpy as np
import autograd.numpy as np
from autograd import grad, jacobian

def softmax(z):
    shifted_z = z - np.max(z)
    exps = np.exp(shifted_z)
    return exps / np.sum(exps)

print(softmax(np.array([5, 3, 0, -1])))

x = np.array(softmax(np.array([5, 3, 0, -1])), dtype=float)

def cost(x):
    return x[0]**2 / x[1] - np.log(x[1])

gradient_cost = grad(cost)
jacobian_cost = jacobian(cost)
gradient_cost(x)
print(jacobian(x))
print(jacobian_cost(x))
print(jacobian_cost(np.array([x, x, x, x], dtype=float)))
