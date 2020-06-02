import numpy as np
import matplotlib.pyplot as plt

# context_dim = 3  # 摩擦系数
# action_dim = 10  # Kp and Kd
# action_bound = 1  # Kp和Kd的范围暂定为1
# D = [[[0.20, 0.007, 0.007], [0.03, 0.0001, 0.0001, 0.005, 0.01, 0.01, 0.05, 0.0001, 0.0001, 0.005], [-0.9]],
#      [[0.30, 0.007, 0.007], [0.03, 0.0001, 0.0001, 0.005, 0.01, 0.01, 0.05, 0.0001, 0.0001, 0.005], [-1.2]],
#      [[0.40, 0.007, 0.007], [0.03, 0.0001, 0.0001, 0.005, 0.01, 0.01, 0.05, 0.0001, 0.0001, 0.005], [-1.5]],
#      [[0.15, 0.007, 0.007], [0.03, 0.0001, 0.0001, 0.005, 0.01, 0.01, 0.05, 0.0001, 0.0001, 0.005], [-0.7]],
#      [[0.10, 0.007, 0.007], [0.03, 0.0001, 0.0001, 0.005, 0.01, 0.01, 0.05, 0.0001, 0.0001, 0.005], [-0.4]],
#      [[0.18, 0.007, 0.007], [0.03, 0.0001, 0.0001, 0.005, 0.01, 0.01, 0.05, 0.0001, 0.0001, 0.005], [-0.7]],
#      [[0.17, 0.007, 0.007], [0.03, 0.0001, 0.0001, 0.005, 0.01, 0.01, 0.05, 0.0001, 0.0001, 0.005], [-0.8]],
#      [[0.14, 0.007, 0.007], [0.03, 0.0001, 0.0001, 0.005, 0.01, 0.01, 0.05, 0.0001, 0.0001, 0.005], [-1.3]]]
# S_ = np.array([[0.18, 0.007, 0.006]])

E = 0.25
T = 0.2  # 更新步长为0.1
Tolent = 0.000001  # 误差容许范围为0.0001
MAX_EP_STEPS = 3000


def argmin_g(D, s_, context_dim):
    eta = 1
    theta = np.ones((context_dim, 1))
    G =[]
    for j in range(MAX_EP_STEPS):
        deta, dtheta, g = gradient(eta, theta, s_, D, context_dim)
        deta *= T
        dtheta *= T
        eta -= deta
        theta -= dtheta.transpose()
        if eta <= 0:
            eta = np.array([[0]])
        G.append(g[0][0])
        if j >= 2:
            if abs(G[j] - G[j - 1]) <= Tolent:
                print('Argmin eta and theta calculated, using ', j, 'episodes.', 'eta:', eta, 'theta:', theta)
                break
            if j == MAX_EP_STEPS:
                print('Argmin eta and theta calculated, using 3000 episodes.', 'eta:', eta, 'theta:', theta)
        # print('eta:', eta, 'theta:', theta)
    # plt.plot(G)
    # plt.show()
    return eta, theta


# calculate the gradient at point (eta, theta)
def gradient(eta, theta, s_, memory, context_dim):
    m = len(memory)
    ze = 0.  # sum of log((r-theta*s)/eta))
    zr = 0.  # sum of log((r-theta*s)/eta))*(r-theta*s)
    zs = 0.  # sum of log((r-theta*s)/eta))*s
    for i in range(m):
        s, a, r = memory[i]
        s = np.array([s])
        a = np.array([a])
        r = np.array([r])
        R = np.dot(s, theta)  # s*theta
        # print('theta*s:', R, 'r:', r)
        z = np.exp((r-R)/eta)
        # print('Z:', z)
        ze += z
        zr += z*(r-R)
        zs += z*s
    # print('ze:', ze, 'zr:', zr, 'zs:', zs)
    gradient_g_eta = E + np.log(ze/m) - zr/(eta*ze)
    gradient_g_theta = s_ - zs/ze
    g = eta * np.log(ze/m) + eta * E + np.dot(s_, theta)
    # print('gradient_eta:', gradient_g_eta)
    # print('gradient_theta:', gradient_g_theta)
    # print ('G(eta, theta):', g)
    return gradient_g_eta, gradient_g_theta, g

#
# eta, theta = argmin_g(D, S_, context_dim)



