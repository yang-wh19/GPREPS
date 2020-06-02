import numpy as np
from envs.env_ import ArmEnv
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from ddpg import DDPG
from Upperlevel_main import GPREPS
from envs.env_ import ArmEnv
import algorithms.calculations as cal


################################  R_MODEL  ###################################

class R_MODEL(object):
    def __init__(self, env, context_dim, pd_dim=12):
        self.env = env
        self.context_dim = context_dim
        self.pd_dim = pd_dim
        self.observation_dim = 12  # single-hole assembly 中为12，分别为Px, Py, Pz, Ox, Oy, Oz, Fx, Fy, Fz, Tx, Ty, Tz
        self.action_dim = 6  # DDPG输出动作的维度，此任务中为6，分别为Px, Py, Pz, Ox, Oy, Oz
        self.action_bound = 1  # DDPG输出动作的上下界
        self.MAX_EP = 5 # 对于一个context，训练DDPG的episode数
        self.MAX_EP_STEPS = 400 # maximized DDPG step number
        self.var = 0.6  # control exploration

        # New gpr model
        self.kernel = DotProduct() + WhiteKernel()
        # the GP model between dA and dX
        self.state_transfer_model = GaussianProcessRegressor(kernel=self.kernel, random_state=0)
        # the GP model between POS and F
        self.contact_model = GaussianProcessRegressor(kernel=self.kernel, random_state=0)

        # New DDPG
        self.ddpg = DDPG(self.observation_dim, self.action_dim, self.action_bound)

    # run DDPG training to collect data and train state transfer model
    def train_state_model(self, gpreps, n):
        # run DDPG training and collect data for model learning
        memory = self.__run_ddpg(gpreps, n)

        # process the memory for forward training
        I, J, X, Y, U = [], [], [], [], []
        dA, dX, POS, F = [], [], [], []
        # dA: action in real space of each step
        # dX: position change during each step
        # POS: position of each step
        # F: contact force of each step
        uk = np.array([0, 0, 0, 0, 0, 0])
        uk_1 = np.array([0, 0, 0, 0, 0, 0])

        for t in range(len(memory)):
            i, j, kp, kd, x, y, u, r, d = memory[t]  # i: episode number, j: step number
            I.append(i)
            J.append(j)
            X.append(x)
            Y.append(y)
            U.append(u)
            pos = X[t][: 6]
            force = X[t][-6:]
            POS.append(pos)
            F.append(force)

            if t >= 2:
                if I[t] == I[t-1] and I[t-1] == I[t-2] and J[t] == J[t-1]+1 and J[t-1] == J[t-2]+1:  # if the consecutive 3 data come from 3 consecutive steps in a single episode:
                    ds = (Y[t]-X[t])[: 6]
                    dX.append(ds)
                    # reproduce the PD control
                    rk = np.array([0, 0, 15, 0, 0, 0])
                    yk = np.array(X[t][-6:])
                    ek = rk - yk
                    yk = np.array(X[t-1][-6:])
                    ek_1 = rk - yk
                    yk = np.array(X[t-2][-6:])
                    ek_2 = rk - yk
                    # discrete PD algorithm
                    uk = uk_1 + kp * (ek - ek_1) + kd * (
                                ek - 2 * ek_1 + ek_2)
                    uk_1 = uk
                    da = uk
                    for i in range(6):
                        da[i] = round(da[i], 4)
                    da = da + da * U[t]
                    dA.append(da)
                else:  # if the episode ends
                    # renew variables
                    uk = np.array([0, 0, 0, 0, 0, 0])
                    uk_1 = np.array([0, 0, 0, 0, 0, 0])
        dA, dX, POS, F = np.array(dA), np.array(dX), np.array(POS), np.array(F)

        # reward model training
        self.state_transfer_model = GaussianProcessRegressor(kernel=self.kernel, random_state=0).fit(dA, dX)
        self.contact_model = GaussianProcessRegressor(kernel=self.kernel, random_state=0).fit(POS, F)

    def __run_ddpg(self, gpreps, n):
        # n training cycles
        for j in range(n):
            self.env.reset()
            # Choose pd parameters
            s = [1, 2]  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            w = gpreps.choose_action(s)  # Kp = action[:, :6], Kd = action[:, 6:]
            kp = w[:, :6][0]
            kd = w[:, 6:][0]
            self.env.pd_control(kd, kp)

            # Start DDPG training
            for i in range(self.MAX_EP):
                self.env.restart()
                observation = self.env.init_state
                ep_reward = 0
                for j in range(self.MAX_EP_STEPS):
                    # Add exploration noise
                    action = self.ddpg.select_action(observation)
                    # print(j, 'th step: ', action)
                    action = np.clip(np.random.normal(action, self.var), -self.action_bound, self.action_bound)

                    # Observe and store
                    observation_, uncode_observation, reward, done, safe = self.env.step(action)
                    self.ddpg.store_transition(i, j, kp, kd, observation, observation_, action, reward, done)

                    # Sample and learn
                    if self.ddpg.pointer > 200:
                        self.ddpg.train()
                        self.var *= .9995  # decay the action randomness

                    # update data
                    observation = observation_
                    ep_reward += reward

                    # 判别结果种类
                    if not safe:
                        print('Episode', i+1, 'Assembly Failed', 'step', j, 'reward', ep_reward)
                        break
                    if done:
                        print('Episode', i+1, 'Assembly Finished', 'step', j, 'reward', ep_reward)
                        break
                    if j == self.MAX_EP_STEPS - 1:
                        print('Episode:', i+1, ' Assembly Unfinished', 'reward', ep_reward)
                        # if ep_reward > -300:RENDER = True
                        break
        return self.ddpg.replay_buffer

    # get an artificial trajectory and compute the reward
    def trajectory(self, context, w):
        # set pd parameters
        kp = w[:, :6][0]
        kd = w[:, 6:][0]
        self.env.pd_control(kd, kp)

        # Start artificial trajectory
        observation = np.array([0., -0.327, -53.77, 0., 0., 0., -0.001, 0., -0.604, 0., 0.001, 0.])  # init observation
        ep_reward = 0
        for j in range(self.MAX_EP_STEPS):
            action = self.ddpg.select_action(observation)
            action = np.clip(np.random.normal(action, self.var), -self.action_bound, self.action_bound)
            action = cal.actions(observation, action, True)
            ds = self.state_transfer_model.predict(np.array([action]), return_std=0, return_cov=0)[0]
            new_pos = ds + observation[: 6]
            new_force = self.contact_model.predict(np.array([new_pos]), return_std=0, return_cov=0)[0]
            observation_ = np.hstack((new_pos, new_force))

            # judge the safety and done, then calculate the reward
            reward = -0.01
            ep_reward += reward
            if observation_[6] >= 50 or observation_[7] >= 50 or observation_[8] >= 200 or observation_[9] >= 3 or observation_[10] >= 3 or observation_[11] >= 3:
                reward = (-1 + (observation_[2] + 52.7) / 40)
                ep_reward += reward
                break
            if observation_[2] > -12:
                reward = 1 - j / self.MAX_EP_STEPS
                ep_reward += reward
                break
        return ep_reward



# 调试程序
env = ArmEnv()
gpreps = GPREPS(12, 2, 10, 0.1)
r_model = R_MODEL(env, 2, 12)
r_model.train_state_model(gpreps, 1)
print('calculating forward model')
R = r_model.trajectory([1, 2], np.array([[0.005, 0.005, 0.005, 0.0001, 0.0001, 0.005, 0.007, 0.007, 0.03, 0.0001, 0.0001, 0.005]]))
print(R)


