import pickle
import numpy as np

pkl_path = '/home/dongyoon/epic/lamp.pkl'
with open(pkl_path, 'rb') as f: data = pickle.load(f)

reward = data['reward']
reward = np.array(reward)

# canon_reward = reward - (0.99-1) * np.mean(reward)
# data['canon_reward'] = canon_reward.tolist()
# with open(pkl_path, 'wb') as f: pickle.dump(data, f)

pearson_corr = np.corrcoef(np.array(data['reward_diff']), data['canon_reward'])[0, 1]
pearson_dist = np.sqrt(1-pearson_corr)/np.sqrt(2)
print(pearson_corr, pearson_dist)