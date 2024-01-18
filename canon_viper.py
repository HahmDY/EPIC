from pathlib import Path

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from omegaconf import OmegaConf

from diffusion_reward.models.video_models.videogpt.transformer import VideoGPTTransformer
import pickle
from types import SimpleNamespace
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import scipy
import tqdm

"""
canonicalize VIPER
"""

pkl_path = '/home/dongyoon/epic/coverage_one_leg.pkl'
config_path = '/home/dongyoon/diffusion_reward/dongyoon/config/viper_lamp_4_16.yaml'

class CustomVIPER(nn.Module):
    def __init__(self, cfg):
        super(CustomVIPER, self).__init__()

        # load video models
        self.model_cfg = OmegaConf.load(cfg.cfg_path)
        self.model = VideoGPTTransformer(self.model_cfg)
        self.model.load_state_dict(torch.load(cfg.ckpt_path))
        self.model.eval()
        for param in self.model.parameters(): 
            param.requires_grad = False

        # set attribute
        for attr_name, attr_value in vars(cfg).items():
            setattr(self, attr_name, attr_value)
        
    def imgs_to_batch(self, x, reward_type='likelihood'):
        '''
        input:
            imgs: B * T * H * W * C
            (mostly): 1 * T * ...
        '''
        seq_len = x.shape[1] # T
        num_frames = self.model_cfg.num_frames + 1 # num_frame+1 = 4
        n_skip = self.model_cfg.frame_skip
        subseq_len = num_frames * n_skip # (num_frame+1) * skip = 80

        x = x.permute(0, 1, 4, 2 ,3) # B * T * C * H * W
        embs, indices = self.model.encode_to_z(x) 
        indices = indices.reshape(indices.shape[0], seq_len, -1)
        embs = embs.reshape(embs.shape[0], seq_len, indices.shape[-1], -1)
        
        if reward_type == 'likelihood':
            post_idxes = list(range(seq_len - subseq_len)) # T - 64
            
            #batch_indices = [indices[:, idx:idx+subseq_len+n_skip:n_skip] for idx in post_idxes] # T, T+16, T+32, T+48, T+64, post_idxes:0~T-64
            batch_indices = []
            for idx in post_idxes:
                idx_idx = [idx+n_skip, idx+2*n_skip, idx+3*n_skip, idx+4*n_skip, min(idx+5*n_skip, seq_len-1)]
                batch_indices.append(indices[:, idx_idx])
            # [1, 5, 256]
            batch_indices = torch.stack(batch_indices, dim=0)
            batch_indices = batch_indices.squeeze(1).reshape(batch_indices.shape[0], -1) # [1280]
            
            #batch_embs = [embs[:, idx:idx+subseq_len+n_skip:n_skip] for idx in post_idxes]
            batch_embs = []
            for idx in post_idxes:
                idx_idx = [idx+n_skip, idx+2*n_skip, idx+3*n_skip, idx+4*n_skip, min(idx+5*n_skip, seq_len-1)]
                batch_embs.append(embs[:, idx_idx])
            # [64, 1280]
            batch_embs = torch.stack(batch_embs, dim=0) # [36, 1, 5, 256, 64]
            batch_embs = batch_embs.squeeze(1).reshape(batch_embs.shape[0], -1, batch_embs.shape[-1]) # [36, 1280, 64]
            
            pre_batch_indices = []
            for idx in range(subseq_len):
                idx_idx = [max(idx-3*n_skip, 0),max(idx-2*n_skip, 0),max(idx-1*n_skip, 0),idx,min(idx+n_skip, seq_len-1)]
                pre_batch_indice = torch.cat([indices[:, idx_idx[0]], indices[:, idx_idx[1]], indices[:, idx_idx[2]], indices[:, idx_idx[3]], indices[:, idx_idx[4]]], dim=1)
                pre_batch_indices.append(pre_batch_indice)
            #pre_batch_indices = [indices[:, min(idx+n_skip, seq_len-1)].tile((1, num_frames)) for idx in range(subseq_len)] # 64 * [1280]
            pre_batch_indices = torch.concat(pre_batch_indices, dim=0) # [64, 1280]
            batch_indices = torch.concat([pre_batch_indices, batch_indices], dim=0) # [128, 1280]

            pre_batch_embs = []
            for idx in range(subseq_len):
                idx_idx = [max(idx-3*n_skip, 0),max(idx-2*n_skip, 0),max(idx-1*n_skip, 0),idx,min(idx+n_skip, seq_len-1)]
                pre_batch_emb = torch.cat([embs[:, idx_idx[0]], embs[:, idx_idx[1]], embs[:, idx_idx[2]], embs[:, idx_idx[3]], embs[:, idx_idx[4]]], dim=1)
                pre_batch_embs.append(pre_batch_emb)
            #pre_batch_embs = [embs[:, min(idx+n_skip, seq_len-1)].tile((1, num_frames, 1)) for idx in range(subseq_len)]
            pre_batch_embs = torch.concat(pre_batch_embs, dim=0)
            batch_embs = torch.concat([pre_batch_embs, batch_embs], dim=0)
        elif reward_type == 'entropy':
            post_idxes = list(range(seq_len - subseq_len + 2))
            batch_indices = [indices[:, idx:idx+subseq_len-n_skip:n_skip] for idx in post_idxes]
            batch_indices = torch.stack(batch_indices, dim=0)
            batch_indices = batch_indices.squeeze(1).reshape(batch_indices.shape[0], -1)
            batch_embs = [embs[:, idx:idx+subseq_len-n_skip:n_skip] for idx in post_idxes]
            batch_embs = torch.stack(batch_embs, dim=0)
            batch_embs = batch_embs.squeeze(1).reshape(batch_embs.shape[0], -1, batch_embs.shape[-1])

            pre_batch_indices = [indices[:, idx].tile((1, num_frames-1)) for idx in range(subseq_len-2)]
            pre_batch_indices = torch.concat(pre_batch_indices, dim=0)
            batch_indices = torch.concat([pre_batch_indices, batch_indices], dim=0)

            pre_batch_embs = [embs[:, idx].tile((1, num_frames-1, 1)) for idx in range(subseq_len-2)]
            pre_batch_embs = torch.concat(pre_batch_embs, dim=0)
            batch_embs = torch.concat([pre_batch_embs, batch_embs], dim=0)
        else:
            raise NotImplementedError

        return batch_embs, batch_indices
    
    @torch.no_grad()
    def calc_reward(self, imgs):
        batch_embs, batch_indices = self.imgs_to_batch(imgs, self.reward_type)
        batch_embs = batch_embs.to('cuda:7')
        batch_indices = batch_indices.to('cuda:7')
        sos_tokens = self.model.calc_sos_tokens(imgs, batch_embs).tile((batch_embs.shape[0], 1, 1))
        sos_tokens = sos_tokens.to('cuda:7')
        rewards = self.cal_log_prob(batch_embs, batch_indices, sos_tokens, target_indices=batch_indices, reward_type=self.reward_type)
        return rewards  
    
    @torch.no_grad()
    def cal_log_prob(self, embs, x, c, target_indices=None, reward_type='likelihood'):
        self.model.eval()
        # x: batch_indices
        # c: sos_tokens
        if not self.model.use_vqemb:
            x = torch.cat((c, x), dim=1) if x is not None else c   
        else:
            x = torch.cat((c, embs), dim=1) if x is not None else c
            
        logits, _ = self.model.transformer(x[:, :-1])
        probs = F.log_softmax(logits, dim=-1)

        if reward_type == 'likelihood':
            target = F.one_hot(target_indices, num_classes=self.model_cfg.codec.num_codebook_vectors)
            if self.compute_joint:
                rewards = (probs * target).sum(-1).sum(-1, keepdim=True)
            else:
                num_valid_logits = int(logits.shape[1] // (self.model_cfg.num_frames + 1))
                rewards = (probs * target).sum(-1)[:, -num_valid_logits:].sum(-1, keepdim=True)
        elif reward_type == 'entropy':
            num_valid_logits = int(logits.shape[1] // (self.model_cfg.num_frames))
            entropy = (- probs * probs.exp()).sum(-1)[:, -num_valid_logits:].sum(-1, keepdim=True)
            rewards = - entropy
        else:
            raise NotImplementedError

        # if self.use_std:
        #     rewards_std = (rewards - self.stat[0]) / self.stat[1]
        # scaled_rewards = (1 - self.expl_scale) * rewards_std
        return rewards

    def update(self, batch):
        metrics = dict()

        if self.use_expl_reward:
            metrics.update(self.expl_reward.update(batch))
        return metrics
    
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
    config = SimpleNamespace(**config)
reward_model = CustomVIPER(config)
if torch.cuda.is_available():
    reward_model = reward_model.to('cuda:7')
    reward_model.model = reward_model.model.to('cuda:7')
    reward_model.model.transformer = reward_model.model.transformer.to('cuda:7')
    reward_model.model.vqgan = reward_model.model.vqgan.to('cuda:7')

def process_frames(frames):
    frames = np.expand_dims(frames, axis=0) # dim 0 for batch
    frames = frames.astype(np.float32)
    frames = frames / 127.5 - 1 # normalize to [-1, 1]
    frames = torch.from_numpy(frames).float().to('cuda:7')
    return frames

def batch_reward(imgs, obs, next_obs_list, reward_model):
    """
    output list of rewards
    """
    reward_output = np.empty(0)
    remain_len = next_obs_list.shape[0]
    
    start_idx = 0
    #trigger = True
    while remain_len > 0:
        if remain_len >= 16:
            frames_to_process = 16
        else:
            frames_to_process = remain_len
        input = np.zeros((96, 64, 64, 3))
        input[0:16]  = imgs[obs[0]]
        input[16:32] = imgs[obs[1]]
        input[32:48] = imgs[obs[2]]
        input[48:64] = imgs[obs[3]]
        input[64:64+frames_to_process] = imgs[next_obs_list[:,3][start_idx:start_idx+frames_to_process]] # next_obs_list[:,3]->x_t+1
        
        frames = process_frames(input)
        reward = reward_model.calc_reward(frames)
        reward = reward.cpu().numpy().squeeze()
        reward = reward[48:48+frames_to_process]
        reward_output = np.concatenate((reward_output, reward))
        
        start_idx += frames_to_process
        remain_len -= frames_to_process
        #trigger = False
    #print(reward_output.shape)
    assert reward_output.shape[0] == next_obs_list.shape[0]
    
    return reward_output

with open(pkl_path, 'rb') as f: data = pickle.load(f)

imgs = np.array(data['imgs'])
next_obs_list = np.array(data['next_obs_viper'])
batch_size = 100
batch_index = np.random.choice(range(next_obs_list.shape[0]), batch_size, replace=False)
next_obs_list = next_obs_list[batch_index]

canon_reward_viper = []
for i in range(len(data['obs'])):
    print("processing {}th reward".format(i+1))
    second_term = batch_reward(imgs, data['next_obs_viper'][i], next_obs_list, reward_model)
    third_term = batch_reward(imgs, data['obs_viper'][i], next_obs_list, reward_model)
    canon_reward_i = data['reward_viper'][i] + 0.99 * np.mean(second_term) - np.mean(third_term)
    canon_reward_viper.append(canon_reward_i)
    
data['canon_reward_viper'] = canon_reward_viper
with open(pkl_path, 'wb') as f: pickle.dump(data, f)