
# coding: utf-8

# # Quadratic Programming
# 
# ## 1. Introduction
# 
# ### 1.1 Libraries Used
# 
# For Quadratic Programming, the packages [quadprog](https://pypi.python.org/pypi/quadprog/0.1.2) and [cvxopt](http://cvxopt.org) were installed:
# 
# ```bash
# pip install quadprog
# pip install cvxopt
# ```
# 
# Help for the appropriate functions are available via
# 
# ```python
# help(quadprog.solve_qp)
# help(cvxopt.solvers.qp)
# ```
# 
# The remaining libraries are loaded in the code below:

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import scipy
import cvxopt
import quadprog
from numpy.random import permutation
from sklearn import linear_model
from sympy import var, diff, exp, latex, factor, log, simplify
from IPython.display import display, Math, Latex
np.set_printoptions(precision=4,threshold=400)
get_ipython().magic('matplotlib inline')


# ### 1.2 Theory
# 
# #### 1.2.1 Lagrange Multipliers
# 
# The Lagrangian is given by:
# 
# $$\mathcal{L}\left(\mathbf{w},b,\mathbf{\alpha}\right) = \frac{1}{2}\mathbf{w^T w} - \sum\limits_{n=1}^N \alpha_n\left[y_n\left(\mathbf{w^T x_n + b}\right) - 1\right]$$
# 
# The Lagrangian may be simplified by making the following substitution:
# 
# $$\mathbf{w} = \sum\limits_{n=1}^N \alpha_n y_n \mathbf{x_n}, \quad \sum\limits_{n=1}^N \alpha_n y_n = 0$$
# 
# whereby we obtain:
# 
# $$\mathcal{L}\left(\mathbf{\alpha}\right) = \sum\limits_{n=1}^N \alpha_n - \frac{1}{2}\sum\limits_{n=1}^N \sum\limits_{m=1}^N y_n y_m \alpha_n \alpha_m \mathbf{x_n^T x_m}$$

# We wish to maximize the Lagrangian with respect to $\mathbf{\alpha}$ subject to the conditions: $\alpha_n \ge 0$ for:
# 
# $$n = 1, \dots, N \quad\text{and}\quad \sum\limits_{n=1}^N \alpha_n y_n = 0$$
# 
# To do this, we convert the Lagrangian to match a form that can be used with quadratic programming software packages.
# 
# $$\min\limits_\alpha \frac{1}{2}\alpha^T \left[\begin{array}{cccc}
# y_1 y_1 \mathbf{x_1^T x_1} & y_1 y_2 \mathbf{x_1^T x_2} & \cdots & y_1 y_N \mathbf{x_1^T x_N}\\
# y_2 y_1 \mathbf{x_2^T x_1} & y_2 y_2 \mathbf{x_2^T x_2} & \cdots & y_2 y_N \mathbf{x_2^T x_N}\\
# \vdots & \vdots & & \vdots\\
# y_N y_1 \mathbf{x_N^T x_1} & y_N y_2 \mathbf{x_N^T x_2} & \cdots & y_N y_N \mathbf{x_N^T x_N}\end{array}\right]\alpha + \left(-\mathbf{1^T}\right)\mathbf{\alpha}$$
# 
# i.e.
# 
# $$\min\limits_\alpha \frac{1}{2}\alpha^T \mathbf{Q} \alpha + \left(-\mathbf{1^T}\right)\mathbf{\alpha}$$
# 
# Subject to the linear constraint: $\mathbf{y^T \alpha} = 0$ and $0 \le \alpha \le \infty$.

# #### 1.2.2 Quadratic Programming
# 
# In [Quadratic Programming](https://en.wikipedia.org/wiki/Quadratic_programming), the objective is to find the value of $\mathbf{x}$ that minimizes the function:
# 
# $$\frac{1}{2}\mathbf{x^T Q x + c^T x}$$
# 
# subject to the constraint:
# 
# $$\mathbf{Ax \le b}$$
# 
# The support vectors are $\mathbf{x_n}$ where $\alpha_n > 0$.

# The solution to the above is calculated using a subroutine such as `solve_qp(G, a, C, b)`,
# which finds the $\alpha$'s that minimize:
# 
# $$\frac{1}{2}\mathbf{x^T G x} - \mathbf{a^T x}$$
# 
# subject to the condition:
# 
# $$\mathbf{C^T x} \ge \mathbf{b}$$
# 
# The quadratic programming solver is implemented in [`solve.QP.c`](https://github.com/rmcgibbo/quadprog/blob/master/quadprog/solve.QP.c), with a Cython wrapper [`quadprog.pyx`](https://github.com/rmcgibbo/quadprog/blob/master/quadprog/quadprog.pyx).  The unit tests are in [`test_1.py`](https://github.com/rmcgibbo/quadprog/blob/master/quadprog/tests/test_1.py) which compares the solution from quadprog's `solve_qp()` with that obtained from `scipy.optimize.minimize`, and [`test_factorized.py`](https://github.com/rmcgibbo/quadprog/blob/master/quadprog/tests/test_factorized.py).

# ## 2. Validation
# 
# ### 2.1 Is there a Validation Bias when choosing the minimum of two random variables?
# 
# Let $\text{e}_1$ and $\text{e}_2$ be independent random variables, distributed uniformly over the interval [0, 1]. Let $\text{e} = \min\left(\text{e}_1, \text{e}_2\right)$. What is the expected values of $\left(\text{e}_1, \text{e}_2, \text{e}\right)$:

# In[2]:

n_samples = 1000
e1 = np.random.random(n_samples)
e2 = np.random.random(n_samples)
e = np.vstack((e1,e2))
e = np.min(e, axis=0)
print("E(e1) = {}".format(np.mean(e1)))
print("E(e2) = {}".format(np.mean(e2)))
print("E(e ) = {}".format(np.mean(e)))


# ### 2.2 Leave-one-out Cross-Validation Example

# In[3]:

from sympy import Matrix, Rational, Eq, sqrt
var('x1 x2 x3 rho')


# The [ordinary least squares](https://en.wikipedia.org/wiki/Ordinary_least_squares) (OLS) estimator for the weights is:
# 
# $$w = \left(\mathbf{X^T X}\right)^{-1}\mathbf{X^T y} = \mathbf{X^\dagger}y$$
# 
# When $\mathbf{X}$ is invertible, $\mathbf{X^\dagger} = \mathbf{X^{-1}}$, so:
# 
# $$w = \mathbf{X^{-1}}y$$
# 
# Lastly, the error is given by
# 
# $$e = \left[h(x) - y\right]^2 = \left|\mathbf{w^T x - y}\right|$$

# Linear model

# In[4]:

def linear_model_cv_err(x1,y1,x2,y2,x3,y3):
    X_train1 = Matrix((x2,x3))
    X_train2 = Matrix((x1,x3))
    X_train3 = Matrix((x1,x2))
    display(Math('X_1^{train} = ' + latex(X_train1) + ', ' +
                 'X_2^{train} = ' + latex(X_train2) + ', ' +
                 'X_3^{train} = ' + latex(X_train3)))
    display(Math('(X_1^{train})^{-1} = ' + latex(X_train1.inv()) + ', ' +
                 '(X_2^{train})^{-1} = ' + latex(X_train2.inv()) + ', ' +
                 '(X_3^{train})^{-1} = ' + latex(X_train3.inv()) ))
    y_train1 = Matrix((y2,y3))
    y_train2 = Matrix((y1,y3))
    y_train3 = Matrix((y1,y2))
    display(Math('y_1^{train} = ' + latex(y_train1) + ', ' +
                 'y_2^{train} = ' + latex(y_train2) + ', ' + 
                 'y_3^{train} = ' + latex(y_train3)))
    w1 = X_train1.inv() * y_train1
    w2 = X_train2.inv() * y_train2
    w3 = X_train3.inv() * y_train3
    display(Math('w_1 = ' + latex(w1) + ', ' +
                 'w_2 = ' + latex(w2) + ', ' +
                 'w_3 = ' + latex(w3)))
    y_pred1 = w1.T*Matrix(x1)
    y_pred2 = w2.T*Matrix(x2)
    y_pred3 = w3.T*Matrix(x3)
    display(Math('y_1^{pred} = ' + latex(y_pred1) + ', ' +
                 'y_2^{pred} = ' + latex(y_pred2) + ', ' +
                 'y_3^{pred} = ' + latex(y_pred3)))
    e1 = (y_pred1 - Matrix([y1])).norm()**2
    e2 = (y_pred2 - Matrix([y2])).norm()**2
    e3 = (y_pred3 - Matrix([y3])).norm()**2
    display(Math('e_1 = ' + latex(e1) + ', ' +
                 'e_2 = ' + latex(e2) + ', ' +
                 'e_3 = ' + latex(e3)))
    return (e1 + e2 + e3)/3


# In[5]:

x1 = 1,-1
x2 = 1,rho
x3 = 1,1
y1 = 0
y2 = 1
y3 = 0
e_linear = linear_model_cv_err(x1,y1,x2,y2,x3,y3)
display(Math('e_{linear\;model} = ' + latex(e_linear)))


# Constant model (inverse does not work here as the matrix is not square)

# In[6]:

def const_model_cv_err(x1,y1,x2,y2,x3,y3):
    X_train1 = Matrix((x2,x3))
    X_train2 = Matrix((x1,x3))
    X_train3 = Matrix((x1,x2))
    y_train1 = Matrix((y2,y3))
    y_train2 = Matrix((y1,y3))
    y_train3 = Matrix((y1,y2))
    w1 = Rational(y2+y3,2)
    w2 = Rational(y1+y3,2)
    w3 = Rational(y1+y2,2)
    e1 = (w1 * Matrix([x1]) - Matrix([y1])).norm()**2
    e2 = (w2 * Matrix([x2]) - Matrix([y2])).norm()**2
    e3 = (w3 * Matrix([x3]) - Matrix([y3])).norm()**2
    return Rational(e1 + e2 + e3,3)


# In[7]:

x1 = 1
x2 = 1
x3 = 1
y1 = 0
y2 = 1
y3 = 0
e_const = const_model_cv_err(x1,y1,x2,y2,x3,y3)
display(Math('e_{constant\;model} = ' + latex(e_const)))


# In[8]:

rho1 = sqrt(sqrt(3)+4)
rho2 = sqrt(sqrt(3)-1)
rho3 = sqrt(9+4*sqrt(6))
rho4 = sqrt(9-sqrt(6))
ans1 = e_linear.subs(rho,rho1).simplify()
ans2 = e_linear.subs(rho,rho2).simplify()
ans3 = e_linear.subs(rho,rho3).simplify()
ans4 = e_linear.subs(rho,rho4).simplify()
display(Math(latex(ans1) + '=' + str(ans1.evalf())))
display(Math(latex(ans2) + '=' + str(ans2.evalf())))
display(Math(latex(ans3) + '=' + str(ans3.evalf())))
display(Math(latex(ans4) + '=' + str(ans4.evalf())))


# Here, we can see that the 3rd expression gives the same leave-one-out cross validation error as the constant model.

# In[9]:

Math(latex(Eq(6*(e_linear-e_const),0)))


# ## 3. Quadratic Programming
# 
# ### 3.1 Background
# In the notation of `help(solve_qp)`, we wish to minimize:
# 
# $$\frac{1}{2}\mathbf{x^T G x - a^T x}$$
# 
# subject to the constraint
# 
# $$\mathbf{C^T x} \ge \mathbf{b}$$

# The matrix, `Q`, (also called `Dmat`) is:
# 
# $$G = \left[\begin{array}{cccc}
# y_1 y_1 \mathbf{x_1^T x_1} & y_1 y_2 \mathbf{x_1^T x_2} & \cdots & y_1 y_N \mathbf{x_1^T x_N}\\
# y_2 y_1 \mathbf{x_2^T x_1} & y_2 y_2 \mathbf{x_2^T x_2} & \cdots & y_2 y_N \mathbf{x_2^T x_N}\\
# \vdots & \vdots & & \vdots\\
# y_N y_1 \mathbf{x_N^T x_1} & y_N y_2 \mathbf{x_N^T x_2} & \cdots & y_N y_N \mathbf{x_N^T x_N}\end{array}\right]$$
# 
# The calculation of the above matrix is implemented in the code below:

# In[10]:

def get_Dmat(X,y):
    n = len(X)
    K = np.zeros(shape=(n,n))
    for i in range(n):
        for j in range(n):
            K[i,j] = np.dot(X[i], X[j])
    Q = np.outer(y,y)*K
    return(Q)


# Calculation of `dvec`:
# 
# $$-\mathbf{a^T x} = \left(-\mathbf{1^T}\right)\mathbf{\alpha} = \begin{pmatrix} -1 & -1 & \dots & -1\end{pmatrix}\mathbf{\alpha}$$
# 
# is implemented as:
# 
# ```python
# a = np.ones(n)
# ```

# Calculation of Inequality constraint:
# 
# $$\mathbf{C^T x} \ge \mathbf{b}$$
# 
# via
# 
# $$\mathbf{y^T x} \ge \mathbf{0}$$
# 
# $$\mathbf{\alpha} \ge \mathbf{0}$$
# 
# where the last two constraints are implemented as:
# 
# $$\mathbf{C^T} = \begin{pmatrix}y_1 & y_2 & \dots & y_n\\
# 1 & 0 & \cdots & 0\\
# 0 & 1 & \cdots & 0\\
# \vdots & \vdots & \ddots & \vdots\\
# 0 & 0 & \cdots & 1\end{pmatrix}$$
# 
# $$\mathbf{b} = \begin{pmatrix}0 \\ 0 \\ \vdots \\ 0\end{pmatrix}$$
# 
# ```python
# C = np.vstack([y,np.eye(n)])
# b = np.zeros(1+n)
# ```

# In[11]:

def get_GaCb(X,y, verbose=False):
    n = len(X)
    assert n == len(y)
    G = get_Dmat(X,y)
    a = np.ones(n)
    C = np.vstack([y,np.eye(n)]).T
    b = np.zeros(1+n)
    I = np.eye(n, dtype=float)
    assert G.shape == (n,n)
    assert y.shape == (n,)
    assert a.shape == (n,)
    assert C.shape == (n,n+1)
    assert b.shape == (1+n,)
    assert I.shape == (n,n)
    if verbose is True:
        print(G)
        print(C.astype(int).T)
    return G,a,C,b,I


# In[12]:

def solve_cvxopt(P, q, G, h, A, b):
    P = cvxopt.matrix(P)
    q = cvxopt.matrix(q)
    G = cvxopt.matrix(G)
    h = cvxopt.matrix(h) 
    A = cvxopt.matrix(A)
    b = cvxopt.matrix(b)
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)
    return solution


# In[13]:

def create_toy_problem_1():
    X = np.array(
         [[ 1.0],
          [ 2.0],
          [ 3.0]])
    y = np.array([-1,-1,1], dtype=float)
    return X,y

def create_toy_problem_2():
    X = np.array(
         [[ 1.0, 0.0],
          [ 2.0, 0.0],
          [ 3.0, 0.0]])
    y = np.array([-1,-1,1], dtype=float)
    return X,y

def create_toy_problem_3():
    X = np.array(
         [[ 0.0, 0.0],
          [ 2.0, 2.0],
          [ 2.0, 0.0],
          [ 3.0, 0.0]])
    y = np.array([-1,-1,1,1], dtype=float)
    return X,y

def create_toy_problem_4():
    X = np.array(
         [[ 0.78683463, 0.44665934],
          [-0.16648517,-0.72218041],
          [ 0.94398266, 0.74900882],
          [ 0.45756412,-0.91334759],
          [ 0.15403063,-0.75459915],
          [-0.47632360, 0.02265701],
          [ 0.53992470,-0.25138609],
          [-0.73822772,-0.50766569],
          [ 0.92590792,-0.92529239],
          [ 0.08283211,-0.15199064]])
    y = np.array([-1,1,-1,1,1,-1,1,-1,1,1], dtype=float)
    G,a,C,b,I = get_GaCb(X,y)
    assert np.allclose(G[0,:],np.array([0.818613299,0.453564930,1.077310034,0.047927947,
        0.215852131,-0.364667935,-0.312547506,-0.807616753,-0.315245922,0.002712864]))
    assert np.allclose(G[n-1,:],np.array([0.002712864,0.095974341,0.035650250,0.176721283,
        0.127450687,0.042898544,0.082931435,-0.016011470,0.217330687,0.029962312]))
    return X,y


# In[14]:

def solve_quadratic_programming(X,y,tol=1.0e-8,method='solve_qp'):
    n = len(X)
    G,a,C,b,I = get_GaCb(X,y)
    eigs = np.linalg.eigvals(G + tol*I)
    pos_definite = np.all(eigs > 0)
    if pos_definite is False:
        print("Warning! Positive Definite(G+tol*I) = {}".format(pos_definite))
    
    if method=='solve_qp':
        try:
            alphas, f, xu, iters, lagr, iact = quadprog.solve_qp(G + tol*I,a,C,b,meq=1)
            print("solve_qp(): alphas = {} (f = {})".format(alphas,f))
            return alphas
        except:
            print("solve_qp() failed")
    else:
        #solution = cvxopt.solvers.qp(G, a, np.eye(n), np.zeros(n), np.diag(y), np.zeros(n))
        solution = solve_cvxopt(P=G, q=-np.ones(n),
                                G=-np.eye(n), h=np.zeros(n),
                                A=np.array([y]), b=np.zeros(1))  #A=np.diag(y), b=np.zeros(n))
        if solution['status'] != 'optimal':
            print("cvxopt.solvers.qp() failed")
            return None
        else:
            alphas = np.ravel(solution['x'])
            print("cvxopt.solvers.qp(): alphas = {}".format(alphas))
        #ssv = alphas > 1e-5
        #alphas = alphas[ssv]
        #print("alphas = {}".format(alphas))
        return alphas


# Here, the quadratic programming code is tested on a handful of 'toy problems'

# In[15]:

#X, y = create_toy_problem_1()
X, y = create_toy_problem_3()
G,a,C,b,I = get_GaCb(X,y,verbose=True)


# In[16]:

X, y = create_toy_problem_3()
alphas1 = solve_quadratic_programming(X,y,method='cvxopt')
alphas2 = solve_quadratic_programming(X,y,method='solve_qp')


# In[17]:

#h = np.hstack([
#    np.zeros(n),
#    np.ones(n) * 999999999.0])
#A = np.array([y])    #A = cvxopt.matrix(y, (1,n))
#b = np.array([0.0])  #b = cvxopt.matrix(0.0)


# In[18]:

def solve_cvxopt(n,P,y_output):
    # Generating all the matrices and vectors
    # P = cvxopt.matrix(np.outer(y_output, y_output) * K)
    q = cvxopt.matrix(np.ones(n) * -1)
    G = cvxopt.matrix(np.vstack([
        np.eye(n) * -1,
        np.eye(n)
        ]))
    h = cvxopt.matrix(np.hstack([
        np.zeros(n),
        np.ones(n) * 999999999.0
        ])) 
    A = cvxopt.matrix(y_output, (1,n))
    b = cvxopt.matrix(0.0)
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)
    return solution


# In[19]:

G = np.eye(3, 3)
a = np.array([0, 5, 0], dtype=np.double)
C = np.array([[-4, 2, 0], [-3, 1, -2], [0, 0, 1]], dtype=np.double)
b = np.array([-8, 2, 0], dtype=np.double)
xf, f, xu, iters, lagr, iact = quadprog.solve_qp(G, a, C, b)


# In[20]:

#https://github.com/rmcgibbo/quadprog/blob/master/quadprog/tests/test_1.py
def solve_qp_scipy(G, a, C, b, meq=0):
    # Minimize     1/2 x^T G x - a^T x
    # Subject to   C.T x >= b
    def f(x):
        return 0.5 * np.dot(x, G).dot(x) - np.dot(a, x)

    if C is not None and b is not None:
        constraints = [{
            'type': 'ineq',
            'fun': lambda x, C=C, b=b, i=i: (np.dot(C.T, x) - b)[i]
        } for i in range(C.shape[1])]
    else:
        constraints = []

    result = scipy.optimize.minimize(f, x0=np.zeros(len(G)), method='COBYLA',
        constraints=constraints, tol=1e-10)
    return result

def verify(G, a, C=None, b=None):
    xf, f, xu, iters, lagr, iact = quadprog.solve_qp(G, a, C, b)
    result = solve_qp_scipy(G, a, C, b)
    np.testing.assert_array_almost_equal(result.x, xf)
    np.testing.assert_array_almost_equal(result.fun, f)
    
def test_1():
    G = np.eye(3, 3)
    a = np.array([0, 5, 0], dtype=np.double)
    C = np.array([[-4, 2, 0], [-3, 1, -2], [0, 0, 1]], dtype=np.double)
    b = np.array([-8, 2, 0], dtype=np.double)
    xf, f, xu, iters, lagr, iact = quadprog.solve_qp(G, a, C, b)
    np.testing.assert_array_almost_equal(xf, [0.4761905, 1.0476190, 2.0952381])
    np.testing.assert_almost_equal(f, -2.380952380952381)
    np.testing.assert_almost_equal(xu, [0, 5, 0])
    np.testing.assert_array_equal(iters, [3, 0])
    np.testing.assert_array_almost_equal(lagr, [0.0000000, 0.2380952, 2.0952381])

    verify(G, a, C, b)
    
def test_2():
    G = np.eye(3, 3)
    a = np.array([0, 0, 0], dtype=np.double)
    C = np.ones((3, 1))
    b = -1000 * np.ones(1)
    verify(G, a, C, b)
    verify(G, a)
    
def test_3():
    random = np.random.RandomState(0)
    G = scipy.stats.wishart(scale=np.eye(3,3), seed=random).rvs()
    a = random.randn(3)
    C = random.randn(3, 2)
    b = random.randn(2)
    verify(G, a, C, b)
    verify(G, a)


# In[21]:

test_1()
test_2()
test_3()


# In[22]:

#https://gist.github.com/zibet/4f76b66feeb5aa24e124740081f241cb
from cvxopt import solvers
from cvxopt import matrix

def toysvm():
    def to_matrix(a):
        return matrix(a, tc='d')    
    X = np.array([
        [0,2],
        [2,2],
        [2,0],
        [3,0]], dtype=float)
    y = np.array([-1,-1,1,1], dtype=float)
    Qd = np.array([
        [0,0,0,0],
        [0,8,-4,-6],
        [0,-4,4,6],
        [0,-6,6,9]], dtype=float)
    Ad = np.array([
        [-1,-1,1,1],
        [1,1,-1,-1],
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1]], dtype=float)
    N = len(y)
    P = to_matrix(Qd)
    q = to_matrix(-(np.ones((N))))
    G = to_matrix(-Ad)
    h = to_matrix(np.array(np.zeros(N+2)))
    sol = solvers.qp(P,q,G,h)
    print(sol['x'])
    
    #xf, f, xu, iters, lagr, iact = solve_qp(Qd, y, Ad, X)


# In[23]:

toysvm()

