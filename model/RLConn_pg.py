from neural_model import NeuralModel
import matplotlib.pyplot as plt
import numpy as np
from pg import PolicyGradient

# target dynamics
Gg = np.array([[0, 8, 5],
                 [8, 0, 2],
                 [5, 2, 0]])
Gs = np.array([[0, 2, 8],
                 [7, 0, 3],
                 [7, 7, 0]])
is_inhibitory = np.array([1, 0 ,0])
I_ext = 100000 * np.array([0, 0.03 ,0])
model = NeuralModel(
            N = 3,
            Gg = Gg,
            Gs = Gs,
            is_inhibitory = is_inhibitory,
            I_ext = I_ext)
(v_mat, s_mat, v_normalized_mat) = model.run(1000)
true_v = v_normalized_mat[250:,:].flatten()

# get random commectome network of size N
def get_random_connectome(N):
    l = int(N*(N-1)/2)
    Gg = np.zeros((N,N))
    Gg[np.logical_not(np.tri(N,dtype=bool))] = np.random.randint(0,10,l)
    Gg += Gg.T
    Gs = np.zeros((N,N))
    Gs[np.tri(N,dtype=bool,k=-1)] = np.random.rand(l)
    Gs[np.logical_not(np.tri(N,dtype=bool))] = np.random.randint(0,10,l)
    return Gg, Gs

# get reward given Gg and Gs using 2-norm error of dynamics
def get_reward(Gg, Gs):
    '''get v_normalized with 1000 time_step
    '''
    model = NeuralModel(
        N = Gg.shape[0],
        Gg = Gg,
        Gs = Gs,
        is_inhibitory = is_inhibitory,
        I_ext = I_ext)
    (v_mat, s_mat, v_normalized_mat) = model.run(1000)
    actual = v_normalized_mat[250:].flatten()
    error = np.linalg.norm(actual-true_v)
    np.random.seed() #reset seed
    return -error

# get the components that we control in Gg and Gs in vectro form
def get_observe(Gg, Gs):
    N = Gg.shape[0]
    return np.concatenate((Gg[np.triu_indices(N,k=1)],Gs[np.triu_indices(N,k=1)],Gs[np.tril_indices(N,k=-1)]))

# change Gg and Gs at some "location" by some "action"
def move(Gg, Gs, action, location, dx = 0.1):
    N = Gg.shape[0]
    size = int(N*(N-1)/2)
    change = dx*(action == 0) + (-dx)*(action == 2)
    if location < size:
        coordinate = np.array(np.triu_indices(N,k=1)).T[location]
        Gg[coordinate[0], coordinate[1]] += change
        Gg[coordinate[1], coordinate[0]] += change
    elif location < 2*size:
        coordinate = np.array(np.triu_indices(N,k=1)).T[location - size]
        Gs[coordinate[0], coordinate[1]] += change
    else: 
        coordinate = np.array(np.tril_indices(N,k=-1)).T[location - 2*size]
        Gs[coordinate[0], coordinate[1]] += change
    return Gg, Gs

# get a random action: (action, location)
def get_rand_action(N):
    size = int(N*(N-1)/2)
    return np.random.randint(2), np.random.randint(size)

# update the observables x given the Gg, Gs, and reward at new time step
def update_x(x, Gg, Gs, reward):
    obs = get_observe(Gg, Gs)
    x = np.concatenate((x[len(obs)+1:], obs))
    x = np.append(x,reward)
    return x


###################################################################################################
# training

# initialize
N = 3
size = int(3*N*(N-1)/2)
RL = PolicyGradient(
    n_actions=3,
    n_features=(size+1)*10)

for i_episode in range(3000):
    #initialize x
    x = np.array([])
    Gg, Gs = get_random_connectome(N)
    reward = get_reward(Gg, Gs)
    x = np.concatenate((x, get_observe(Gg, Gs)))
    x = np.append(x,reward)
    for i in range(9):
        action, location = get_rand_action(N)
        Gg, Gs = move(Gg, Gs, action, location, dx=0.1)
        reward = get_reward(Gg, Gs)
        x = np.concatenate((x, get_observe(Gg, Gs)))
        x = np.append(x,reward)    
    
    #choose action using nn
    for step in range(300):
        action, location = RL.choose_action(x)
        Gg, Gs = move(Gg, Gs, action, location, dx=0.1)
        reward = get_reward(Gg, Gs)
        RL.store_transition(x, action, location, reward)
        x = update_x(x, Gg, Gs, reward)
    ep_rs_sum = sum(RL.ep_rs)
    if 'running_reward' not in globals():
        running_reward = ep_rs_sum
    else:
        running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
    print("episode:", i_episode, "  reward:", int(running_reward))
    print("Gg:", Gg.flatten(), "  Gs:", Gs.flatten())
    vt = RL.learn()
    # print(vt)










