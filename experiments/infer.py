"""
Reproduce Omniglot results of Snell et al Prototypical networks.
"""
from torch.optim import Adam
from torch.utils.data import DataLoader
import argparse

from few_shot.datasets import OmniglotDataset, MiniImageNet, FashionDataset
from few_shot.models import get_few_shot_encoder
from few_shot.core import NShotTaskSampler, EvaluateFewShot, prepare_nshot_task
from few_shot.proto import proto_net_episode
from few_shot.train import fit
from few_shot.callbacks import *
from few_shot.utils import setup_dirs
from config import PATH

setup_dirs()
assert torch.cuda.is_available()
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True


##############
# Parameters #
##############
parser = argparse.ArgumentParser()
parser.add_argument('--dataset')
parser.add_argument('--distance', default='l2')
#parser.add_argument('--n-train', default=1, type=int)
parser.add_argument('--n-test', default=1, type=int)
#parser.add_argument('--k-train', default=60, type=int)
parser.add_argument('--k-test', default=5, type=int)
#parser.add_argument('--q-train', default=5, type=int)
parser.add_argument('--q-test', default=1, type=int)
parser.add_argument('--exp_name', default='test', type=str)
parser.add_argument('--model_path', default='test', type=str)
args = parser.parse_args()

param_str = f' {args.exp_name}_experiment_name_{args.dataset}_nt={args.n_test}_kt={args.k_test}_qt={args.q_test}_' \
            f'nv={args.n_test}_kv={args.k_test}_qv={args.q_test}'

print(param_str)
if args.dataset == 'omniglot':
    n_epochs = 40
    dataset_class = OmniglotDataset
    num_input_channels = 1
    drop_lr_every = 20
elif args.dataset == 'miniImageNet':
    n_epochs = 80
    dataset_class = MiniImageNet
    num_input_channels = 3
    drop_lr_every = 40
elif args.dataset == 'fashion-dataset':
    n_epochs = 80
    dataset_class = FashionDataset
    num_input_channels = 3
    drop_lr_every = 40
else:
    # add for fashion here
    raise(ValueError, 'Unsupported dataset')


###################
# Create datasets #
###################
background = dataset_class('background')
# no batch size for proto nets
# background_taskloader = DataLoader(
#     background,
#     batch_sampler=NShotTaskSampler(background, episodes_per_epoch, args.n_train, args.k_train, args.q_train),
#     num_workers=4
# )
evaluation = dataset_class('evaluation')
evaluation_taskloader = DataLoader(
    evaluation,
    batch_sampler=NShotTaskSampler(evaluation, episodes_per_epoch, args.n_test, args.k_test, args.q_test), # why is qtest needed for protonet i think its not rquired for protonet check it
    num_workers=4
)


#########
# Model #
#########
model = get_few_shot_encoder(num_input_channels)
model.to(device, dtype=torch.double)


model.load_state_dict(torch.load(args.model_path))
import pdb; pdb.set_trace()


