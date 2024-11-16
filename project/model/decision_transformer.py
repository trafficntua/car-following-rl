import os
import random
from dataclasses import dataclass
import numpy as np
import torch
from datasets import load_dataset
from transformers import DecisionTransformerConfig, DecisionTransformerModel, Trainer, TrainingArguments


import pickle

class TrainableDT(DecisionTransformerModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        # add the DT loss
        action_preds = output[1]
        action_targets = kwargs["actions"]
        attention_mask = kwargs["attention_mask"]
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_targets = action_targets.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = torch.mean((action_preds - action_targets) ** 2)

        return {"loss": loss}

    def original_forward(self, **kwargs):
        return super().forward(**kwargs)


# Function that gets an action from the model using autoregressive prediction with a window of the previous 20 timesteps.
def get_action(model: TrainableDT, states, actions, rewards, returns_to_go, timesteps):
    # This implementation does not condition on past rewards

    states = states.reshape(1, -1, model.config.state_dim)
    actions = actions.reshape(1, -1, model.config.act_dim)
    returns_to_go = returns_to_go.reshape(1, -1, 1)
    timesteps = timesteps.reshape(1, -1)

    states = states[:, -model.config.max_length :]
    actions = actions[:, -model.config.max_length :]
    returns_to_go = returns_to_go[:, -model.config.max_length :]
    timesteps = timesteps[:, -model.config.max_length :]
    padding = model.config.max_length - states.shape[1]
    # pad all tokens to sequence length
    attention_mask = torch.cat([torch.zeros(padding), torch.ones(states.shape[1])])
    attention_mask = attention_mask.to(dtype=torch.long).reshape(1, -1)
    states = torch.cat([torch.zeros((1, padding, model.config.state_dim)), states], dim=1).float()
    actions = torch.cat([torch.zeros((1, padding, model.config.act_dim)), actions], dim=1).float()
    returns_to_go = torch.cat([torch.zeros((1, padding, 1)), returns_to_go], dim=1).float()
    timesteps = torch.cat([torch.zeros((1, padding), dtype=torch.long), timesteps], dim=1)

    _, action_preds, _ = model.original_forward(
        states=states,
        actions=actions,
        rewards=rewards,
        returns_to_go=returns_to_go,
        timesteps=timesteps,
        attention_mask=attention_mask,
        return_dict=False,
    )

    return action_preds[0, -1]

def predict(traj, TARGET_RETURN):
    states = torch.tensor(traj['observations'], device=device, dtype=torch.float32)
    actions =  torch.tensor(traj['actions'], device=device, dtype=torch.float32)
    # dones =  torch.tensor(traj['dones'], device=device, dtype=torch.float32)
    rewards =  torch.tensor(traj['rewards'], device=device, dtype=torch.float32)
    timesteps = torch.arange(0, len(traj['observations']), device=device, dtype=torch.long).reshape(1, -1)
    # for gamma == 1
    target_return = torch.tensor(TARGET_RETURN, device=device, dtype=torch.float32).reshape(1, 1)
    target_return = target_return - torch.cumsum(rewards, dim=0).reshape(1, -1)
    next_action = get_action(
        model,
        (states - state_mean) / state_std,
        actions,
        rewards,
        target_return,
        timesteps,
    )
    return next_action.item()

# weights
model = TrainableDT.from_pretrained('/project/model/weights/')

device = "cpu"
#state_mean = np.array([28.62879043, 7.06839248, 7.12274401, 0.], dtype=np.float32)
#state_std = np.array([33.54715525, 4.09763968, 4.10320888, 1.], dtype=np.float32)
state_mean = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
state_std = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
state_dim = model.config.state_dim
act_dim = model.config.act_dim

state_mean = torch.from_numpy(state_mean).to(device=device)
state_std = torch.from_numpy(state_std).to(device=device)
model = model.to(device)