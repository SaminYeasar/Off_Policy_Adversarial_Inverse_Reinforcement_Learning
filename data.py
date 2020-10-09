import numpy as np



class ReplayBuffer(object):
    def __init__(self):
        self.storage = []

    # Expects tuples of (state, next_state, action, reward, done)
    def add(self, data):
        self.storage.append(data)


    def save_traj(self, filename='trajectory', dirr=None):
        if dirr==None:
            print('saving at common folder')
            np.save('{}.npy'.format(filename), self.storage)
        else:
            np.save('{}/{}.npy'.format(dirr,filename), self.storage)

    def sample(self, batch_size=100):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(
            -1, 1), np.array(d).reshape(-1, 1)


class ReplayBufferIRL(object):
    def __init__(self):
        self.storage = []

    # Expects tuples of (state, next_state, action, reward, done)
    def add(self, data):
        self.storage.append(data)

    def sample(self, batch_size=100):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d ,e = [], [], [], [], [], []

        for i in ind:
            X, Y, U, R, D ,E = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))
            e.append(np.array(E, copy=False))
        # state, next_state, action, lprob, reward, done
        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(
            -1, 1), np.array(d).reshape(-1, 1), np.array(e).reshape(-1, 1)

