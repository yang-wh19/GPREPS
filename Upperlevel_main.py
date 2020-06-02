import numpy as np
from envs.env_ import ArmEnv
from context_model import CONTEXT as S_MODEL  # unfinished yet
from Lowerlevel_gp_model import R_MODEL
from argmin import argmin_g as argmin
import torch
import torch.nn as nn
import torch.nn.functional as F

np.set_printoptions(precision=3, suppress=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


###########################  GAC  ############################
class GPREPS(object):
    def __init__(self, a_dim, s_dim, memory_dim, a_bound):
        # initialize parameters
        self.memory = []
        self.pointer = 0
        self.a_dim, self.s_dim, self.memory_dim, self.a_bound = a_dim, s_dim, memory_dim, a_bound

        # build actor
        self.a = np.ones((a_dim, 1), dtype=np.float32)
        self.A = np.ones((a_dim, s_dim), dtype=np.float32)
        self.COV = np.ones((a_dim, a_dim), dtype=np.float32)

    def choose_action(self, s):
        s = np.array([s])
        u = self.a + np.dot(self.A, s.transpose())
        u = u.transpose()[0]
        return np.random.multivariate_normal(mean=u, cov=self.COV, size=1)

    def learn(self, s_):
        eta, theta = argmin(self.memory, s_, self.s_dim)
        p = 0.
        P_ = []
        S = []
        B = []
        for i in range(self.memory_dim):
            s, w, r = self.memory[i]
            s = np.array([s])
            w = np.array([w])
            r = np.array([r])
            p = np.exp((r - np.dot(s, theta)) / eta)
            s_ = np.c_[np.array([1.]), s]
            S.append(s_[0])
            B.append(w[0])
            P_.append(p[0])
        p, B, S = np.array(p), np.array(B), np.array(S)
        P = P_ * np.eye(self.memory_dim)

        # calculate mean action
        target1 = np.linalg.inv(np.dot(np.dot(S.transpose(), P), S))
        target2 = np.dot(np.dot(S.transpose(), P), B)
        target = np.dot(target1, target2).transpose()
        self.a = target[:, :1]
        self.A = target[:, 1:]

        # calculate the COV
        Err = 0
        for i in range(self.memory_dim):
            s, w, r = self.memory[i]
            s = np.array([s])
            w = np.array([w])
            err = w - self.a - np.dot(self.A, s.transpose())
            Err += np.dot(err, err.transpose()) * P_[i]
        self.COV = Err / np.sum(P_)
        # print(COV)

    def store_data(self, s, a, r):
        transition = [s, a, [r]]
        if len(self.memory) == self.memory:
            index = self.pointer % self.memory_dim  # replace the old memory with new memory
            self.memory[index] = transition
        else:
            self.memory.append(transition)
        self.pointer += 1


######################  GAC Parameters  ######################
K = 1000  # number of policy training cycles
N = 10  # number of episodes during a single DDPG training
M = 10  # number of artificial dataset collected per policy training cycle
L = 10  # number of artificial trajectories per prediction
MAX_EP_STEPS = 200
context_dim = 1  # 摩擦系数
action_dim = 12  # Kp and Kd
action_bound = 0.1  # Kp和Kd的范围暂定为0.1

env = ArmEnv()
gpreps = GPREPS(action_dim, context_dim, M, action_bound)
r_model = R_MODEL(env, context_dim, action_dim)
s_model = S_MODEL()

###########################  Main  ###########################
for k in range(K):
    # Collect Data
    r_model.train_state_model(gpreps, N)
    s_model.train()

    # Predict Rewards and Store Data
    D = np.zeros((M, context_dim + action_dim + 1), dtype=np.float32)
    for j in range(M):
        # Predict Rewards
        R = 0
        S = s_model.predict()
        Action = gpreps.choose_action(S)

        # Predict L Trajectories
        for l in range(L):
            R += r_model.trajectory(S, Action)
        reward = R / L

        # Construct Artiﬁcial Dataset D
        gpreps.store_data(S, Action, R)

    # Sample and Update Policy
    S_ = s_model.average()
    gpreps.learn(S_)
