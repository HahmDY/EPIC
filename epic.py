import pickle
import numpy as np

pkl_path = '/home/dongyoon/epic/lamp.pkl'
with open(pkl_path, 'rb') as f: data = pickle.load(f)

# canonicalize sparse reward

reward = data['reward']
reward = np.array(reward)
canon_reward = reward - (0.99-1) * np.mean(reward)
data['canon_reward'] = canon_reward.tolist()
with open(pkl_path, 'wb') as f: pickle.dump(data, f)

# canonicalize step reward

reward = data['reward_step']
reward = np.array(reward)
canon_reward = reward + (0.99-1) * np.mean(reward)
data['canon_reward_step'] = canon_reward.tolist()
with open(pkl_path, 'wb') as f: pickle.dump(data, f)

# canonicalize diffusion_reward

canon_diff_reward = 0.99 * np.array(data['reward_diff_next'])
data['canon_reward_diff'] = canon_diff_reward.tolist()
with open(pkl_path, 'wb') as f: pickle.dump(data, f)

# sparse vs diff
print("sparse vs diff")
pearson_coff = np.corrcoef(data['canon_reward'], data['canon_reward_diff'])[0,1]
print(pearson_coff)
pearson_dist = np.sqrt(1-pearson_coff) / np.sqrt(2)
print(pearson_dist)

# step vs diff
print("step vs diff")
pearson_coff = np.corrcoef(data['canon_reward_step'], data['canon_reward_diff'])[0,1]
print(pearson_coff)
pearson_dist = np.sqrt(1-pearson_coff) / np.sqrt(2)
print(pearson_dist)

# step vs sparse
print("step vs sparse")
pearson_coff = np.corrcoef(data['canon_reward_step'], data['canon_reward'])[0,1]
print(pearson_coff)
pearson_dist = np.sqrt(1-pearson_coff) / np.sqrt(2)
print(pearson_dist)