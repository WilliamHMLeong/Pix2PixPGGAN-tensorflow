import tensorflow as tf
import numpy as np

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
event_acc = EventAccumulator('E:/dp/data/result/403-pgan-celeba-feature-pose-preset-v2-1gpu-fp32/')
event_acc.Reload()
# Show all tags in the log file
print(event_acc.Tags())

EPS = 0.0001

wgan_lambda     = 10.0     # Weight for the gradient penalty term.
wgan_epsilon    = 0.001    # Weight for the epsilon term, \epsilon_{drift}.
wgan_target     = 1.0     # Target value for gradient magnitudes.
cond_weight     = 1.0

# E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
w_times, step_nums, real_scores_out = zip(*event_acc.Scalars('Loss/real_scores'))
print("real_scores")
print(real_scores_out)

w_times, step_nums, fake_scores_out = zip(*event_acc.Scalars('Loss/fake_scores'))
print("fake_scores_out")
print(fake_scores_out)

w_times, step_nums, mixed_scores = zip(*event_acc.Scalars('Loss/mixed_scores'))
print("mixed_scores")
print(mixed_scores)

w_times, step_nums, mixed_norms = zip(*event_acc.Scalars('Loss/mixed_norms'))
print("mixed_norm")
print(mixed_norms)

w_times, step_nums, epsilon_penalty = zip(*event_acc.Scalars('Loss/epsilon_penalty'))
print("epsilon_penalty")
print(epsilon_penalty)

w_times, step_nums, L1_loss = zip(*event_acc.Scalars('Loss/L1_loss'))
print("L1_loss")
print(L1_loss)

num = 1

#LP_loss = (-(np.log(np.abs(real_scores_out[num]/100 + EPS)) + np.log(1 - np.abs(fake_scores_out[num]/100) + EPS)))
#print(LP_loss)

gradient_penalty = np.square(mixed_norms[num] - wgan_target)

Dloss = fake_scores_out[num] - real_scores_out[num] + gradient_penalty * (10.0 / (1.0**2)) + epsilon_penalty[num] * 0.001

Gloss = -1 * fake_scores_out[num] + 1*L1_loss[num]

print(fake_scores_out[num])
print(L1_loss[num])

print("Gloss")
print(Gloss)


print("Dloss")
print(Dloss)

