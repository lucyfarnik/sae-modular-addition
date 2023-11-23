#%%
import os
import sys
from tqdm import tqdm

import torch as T
from torch import nn
from torch.utils.data import DataLoader
from transformer_lens.utils import get_device
import wandb


sys.path.append('.')
from progress_measures_paper.transformers import Config, Transformer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = get_device()
print(f"Using device: {device}")  # You will need a GPU
# %%
model_info = T.load("mod_addition_no_wd.pth", map_location=T.device('cpu'))
# %%
config = Config()
model = Transformer(config).to(device)
model.load_state_dict(model_info['model'])
# %%
my_cache = dict()
model.cache_all(my_cache)
model(T.tensor([[69, 42, 113]]))
my_cache.keys()
# %%
# keys_that_should_be_in_config = HookedTransformerConfig(config.num_layers,
#                                                         config.d_model,
#                                                         config.n_ctx,
#                                                         config.d_head,
#                                                         act_fn=config.act_type.lower()).__dict__.keys()
# config_dict = config.__dict__
# keys_to_del = []
# for k in config_dict:
#     if k not in keys_that_should_be_in_config:
#         keys_to_del.append(k)
# for k in keys_to_del:
#     del config_dict[k]
# if 'device' in config_dict:
#     del config_dict['device']
# config_dict['n_layers'] = config.num_layers
# config_dict['d_head'] = config.d_head
# config_dict['act_fn'] = config.act_type.lower()
# hooked_model = HookedTransformer(config_dict)
# hooked_model.load_and_process_state_dict(model_info['model'])
# %%
all_data = T.tensor([(i, j, config.p) for i in range(config.p) for j in range(config.p)]).to(device)
cache = dict()
model.cache_all(cache)
model(all_data);
#%%
# define the autoencoder
class SAE(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()
        self.encoder = nn.Linear(n_input, n_hidden)
        self.decoder = nn.Linear(n_hidden, n_output)
    
    def forward(self, input):
        sae_activations = self.encoder(input)
        output = self.decoder(sae_activations)
        return sae_activations, output

#%%
def loss_function(input, sae_activations, output, l1_weight=1e-3):
    """Return reconstruction loss plus L1 penalty on activations."""
    reconstruction_loss = nn.MSELoss()(input, output)
    l1_penalty = T.norm(sae_activations, p=1)
    return reconstruction_loss + l1_weight*l1_penalty
# %%
hparams = {
    'batch_size': 64, #! AAAAA
    'lr': 1e-3,
    'l1_weight': 1e-3,
    'num_epochs': 100,
}

expansion_ratio = 4
sae = SAE(config.d_model, n_hidden=expansion_ratio*config.d_model,
          n_output=config.d_model).to(device)
dataloader = DataLoader(cache['blocks.0.hook_resid_post'],
                        batch_size=hparams['batch_size'],
                        shuffle=True)
optimizer = T.optim.Adam(model.parameters(), lr=hparams['lr'])

for epoch in tqdm(range(hparams['num_epochs'])):
    for batch in dataloader:
        optimizer.zero_grad()

        batch_last_pos = batch[:, -1, :]

        sae_activations, output = sae(batch_last_pos)

        loss = loss_function(batch_last_pos, sae_activations, output, l1_weight=hparams['l1_weight'])

        loss.backward()
        optimizer.step()

# %%