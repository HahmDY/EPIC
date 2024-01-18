import numpy as np
import pickle
import os
from PIL import Image
from pathlib import Path

pkl_dir = '/home/dongyoon/FB_dataset/raw/low/one_leg/val'
output = '/home/dongyoon/epic/coverage_one_leg.pkl'

pkl_dir_Path = Path(pkl_dir)
files = list(pkl_dir_Path.glob(r"[0-9]*success.pkl"))
len_files = len(files)
time_cnt = 0

###############################################
# index of below lists are treated as time step
imgs_ = []

obs_ = []
obs_viper_ = []
obs_diff_ = []

next_obs_ = []
next_obs_viper_ = []
next_obs_diff_ = []

action_ = []

reward_ = []
reward_viper_ = []
reward_diff_ = []
reward_diff_next_ = []

done_ = []
###############################################
    
for i, file_path in enumerate(files):
    print(f"Loading [{i+1}/{len_files}] {file_path}...")
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        
    length = len(data["observations"])
    
    for i in range(length): # 0 ~ T-1 (t=1~T)
        frame = np.transpose(data["observations"][i]["color_image2"], (1, 2, 0))
        img = Image.fromarray(frame)
        resized_img = img.resize((64, 64))
        frame = np.array(resized_img)
        imgs_.append(frame)
        
        obs_.append(i + time_cnt)
        obs_viper_.append(data['viper_stacked_timesteps_16'][i] + time_cnt)
        obs_diff_.append(data['diffusion_stacked_timesteps_16'][i] + time_cnt)
        
        next_obs_.append(min(i + 1 + time_cnt, length - 2))
        next_obs_viper_.append(data['viper_stacked_timesteps_16'][min(i + 1, length - 1)] + time_cnt)
        next_obs_diff_.append(data['diffusion_stacked_timesteps_16'][min(i + 1, length - 1)] + time_cnt)
        
        action_.append(data['actions'][i])
        
        reward_.append(data['rewards'][i])
        reward_viper_.append(data['viper_reward_16'][i])
        reward_diff_.append(data['diffusion_reward_16'][i])
        reward_diff_next_.append(data['diffusion_reward_16'][min(i + 1, length - 1)])
        
        done_.append(1 if i == length - 1 else 0)
        
    time_cnt += length
    
coverage = {
    'imgs': imgs_,
    
    'obs': obs_,
    'obs_viper': obs_viper_,
    'obs_diff': obs_diff_,
    
    'next_obs': next_obs_,
    'next_obs_viper': next_obs_viper_,
    'next_obs_diff': next_obs_diff_,
    
    'action': action_,
    
    'reward': reward_,
    'reward_viper': reward_viper_,
    'reward_diff': reward_diff_,
    'reward_diff_next': reward_diff_next_,
    
    'done': done_
}

path = Path(output)
path.parent.mkdir(exist_ok=True, parents=True)
with Path(path).open("wb") as f:
    pickle.dump(coverage, f)
    print(f"Saved at {path}")