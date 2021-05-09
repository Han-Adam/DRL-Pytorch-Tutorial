import numpy as np
import time
from make_env import make_env
import torch
import torch.nn.functional as F
from Network import Actor, Critic
import cv2

ENV_NAME = 'simple_push'
PATH = './agent'
NUM = 2
RANDOM = False
FPS = 24
FOURCC = cv2.VideoWriter_fourcc(*'XVID')
WINDOW_SIZE = (700, 700)
HIDDEN = 64
# define the environment
env = make_env(scenario_name=ENV_NAME)
NUM = env.n
state_shape_n = [env.observation_space[i].shape[0] for i in range(env.n)]
action_shape_n = [env.action_space[i].n for i in range(env.n)]
# define the actor
actors = [None for _ in range(NUM)]
critics = [None for _ in range(NUM)]
for i in range(NUM):
    actors[i] = Actor(state_n=state_shape_n[i],
                      action_n=action_shape_n[i],
                      hidden=HIDDEN)
    actors[i].load_state_dict(torch.load(PATH+str(i)+'.pt',map_location=torch.device('cpu')))
# define the video writer.
videoWriter = cv2.VideoWriter('./TestVideo.avi',FOURCC,FPS,WINDOW_SIZE,True)
# start to evaluate
s_n = env.reset()
for i in range(100):
    image = env.render(mode='rgb_array')
    image = np.array(image)
    image = np.reshape(image,image.shape[1:])
    r,g,b = cv2.split(image)  # 分解Opencv里的标准格式B、G、R
    image = cv2.merge([b,g,r])
    videoWriter.write(image)

    a_n = []
    if RANDOM:
        a_n = [actor(torch.FloatTensor(s).to(device)).detach().tolist()
               for actor, s in zip(actors, s_n)]
    else:
        a_n = []
        for j in range(NUM):
            a, policy = actors[j](torch.FloatTensor(s_n[j]), model_original_out=True)
            a_n.append(F.softmax(a,dim=-1).detach().tolist())
    # a_n = [actor(torch.FloatTensor(s).to(device)).detach().tolist()
    #        for actor, s in zip(actors, s_n)]

    s_n_, r_n, done_n, infor = env.step(a_n)
    print(i)
    if all(done_n):
        s_n = env.reset()
    else:
        s_n = s_n_

videoWriter.release()
cv2.destroyAllWindows()