from actor_critic import A2C_Network

env_id = "BreakoutNoFrameskip-v4"

# hyperparameters
GAMMA = 0.99
TOTAL_UPDATES = int(1.5e5)
LEARNING_RATE = 7e-4
INPUT_DIM = (4, 84, 84)
CONV1_FILTER = 32
CONV2_FILTER = 64
CONV3_FILTER = 128
HIDDEN_DIM = 512
ENTROPY_COEF = 3e-3
CRITIC_COEF = 0.5
NUM_ENVS = 16
N_STEPS = 5
RANDOM_SEED = 0
CLIP_GRAD = 0.5

agent = A2C_Network(env_id, GAMMA, TOTAL_UPDATES, LEARNING_RATE, INPUT_DIM,
                    CONV1_FILTER, CONV2_FILTER, CONV3_FILTER, HIDDEN_DIM,
                    ENTROPY_COEF, CRITIC_COEF, CLIP_GRAD)

agent.training(NUM_ENVS,N_STEPS,RANDOM_SEED)

agent.save_policy_video(episodes=1)
