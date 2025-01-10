import numpy as np
import torch
from model.decision_transformer import TrainableDT


# Updated get_action for batch processing
def get_action(model: TrainableDT, states, actions, rewards, returns_to_go, timesteps):
    batch_size = states.shape[0]
    sequence_length = states.shape[1]
    #returns_to_go = returns_to_go.reshape(1, -1, 1)
    # Truncate sequences to max_length and pad if necessary
    max_length = model.config.max_length
    if sequence_length > max_length:
        states = states[:, -max_length:]
        actions = actions[:, -max_length:]
        returns_to_go = returns_to_go[:, -max_length:]
        timesteps = timesteps[:, -max_length:]

    padding = max_length - states.shape[1]

    # Pad all tensors to max_length
    attention_mask = torch.cat(
        [torch.zeros((batch_size, padding)), torch.ones((batch_size, states.shape[1]))],
        dim=1,
    ).to(dtype=torch.long)

    states = torch.cat(
        [torch.zeros((batch_size, padding, model.config.state_dim)), states], dim=1
    ).float()
    actions = torch.cat(
        [torch.zeros((batch_size, padding, model.config.act_dim)), actions], dim=1
    ).float()
    returns_to_go = torch.cat(
        [torch.zeros((batch_size, padding, 1)), returns_to_go], dim=1
    ).float()
    timesteps = torch.cat(
        [torch.zeros((batch_size, padding), dtype=torch.long), timesteps], dim=1
    )

    # Perform batch inference
    _, action_preds, _ = model.original_forward(
        states=states,
        actions=actions,
        rewards=rewards,
        returns_to_go=returns_to_go,
        timesteps=timesteps,
        attention_mask=attention_mask,
        return_dict=False,
    )

    # Return the last predicted action for each sequence in the batch
    return action_preds[:, -1]


# Updated predict for batch processing
def batch_predict(batch_trajectories, TARGET_RETURN):
    batch_size = len(batch_trajectories)

    # Determine the maximum sequence length across all trajectories
    max_seq_len = max(len(traj["observations"]) for traj in batch_trajectories)

    # Prepare padded tensors
    states = torch.zeros(
        (batch_size, max_seq_len, state_dim), device=device, dtype=torch.float32
    )
    actions = torch.zeros(
        (batch_size, max_seq_len, act_dim), device=device, dtype=torch.float32
    )
    rewards = torch.zeros(
        (batch_size, max_seq_len), device=device, dtype=torch.float32
    )
    timesteps = torch.zeros(
        (batch_size, max_seq_len), device=device, dtype=torch.long
    )

    # Populate the padded tensors with actual data, padding at the beginning
    for i, traj in enumerate(batch_trajectories):
        seq_len = len(traj["observations"])
        states[i, -seq_len:] = torch.tensor(traj["observations"], device=device)
        actions[i, -seq_len:] = torch.tensor(traj["actions"], device=device)
        rewards[i, -seq_len:] = torch.tensor(traj["rewards"], device=device)
        timesteps[i, -seq_len:] = torch.arange(seq_len, device=device, dtype=torch.long)

    # Calculate returns-to-go using all rewards
    target_return = torch.tensor(
        [TARGET_RETURN] * batch_size, device=device, dtype=torch.float32
    ).reshape(batch_size, 1)
    target_return = target_return.expand(-1, max_seq_len).unsqueeze(-1)
    target_return = target_return - torch.cumsum(rewards, dim=1).unsqueeze(-1)

    # Normalize states
    states = (states - state_mean) / state_std

    # Get batch predictions
    next_actions = get_action(model, states, actions, rewards, target_return, timesteps)

    # Convert to list of predicted actions
    return next_actions.tolist()


# weights
model = TrainableDT.from_pretrained("/project/model/weights/")

device = "cpu"
state_mean = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
state_std = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
state_dim = model.config.state_dim
act_dim = model.config.act_dim

state_mean = torch.from_numpy(state_mean).to(device=device)
state_std = torch.from_numpy(state_std).to(device=device)
model = model.to(device)
