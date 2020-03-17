"""
Script for domain transfer experiments

"""
from torch.optim import Adam
from torch.utils.data import DataLoader
import argparse

from few_shot.models import FewShotClassifier
from few_shot.datasets import OmniglotDataset, MiniImageNet, FashionDataset
from few_shot.models import get_few_shot_encoder
#from  experiments.maml import prepare_meta_batch
from few_shot.core import NShotTaskSampler, EvaluateFewShot, prepare_nshot_task, create_nshot_task_label
from few_shot.proto import proto_net_episode
from few_shot.matching import matching_net_episode
from few_shot.maml import meta_gradient_step

from few_shot.train import fit
from few_shot.callbacks import *
from few_shot.utils import setup_dirs
from config import PATH

setup_dirs()
assert torch.cuda.is_available()
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True

#def prepare_meta_batch(n, k, q, meta_batch_size):
#    def prepare_meta_batch_(batch):
#        x, y = batch
        # Reshape to `meta_batch_size` number of tasks. Each task contains
        # n*k support samples to train the fast model on and q*k query samples to
        # evaluate the fast model on and generate meta-gradients
#        x = x.reshape(meta_batch_size, n*k + q*k, num_input_channels, x.shape[-2], x.shape[-1])
        # Move to device
#        x = x.double().to(device)
        # Create label
#        y = create_nshot_task_label(k, q).cuda().repeat(meta_batch_size)
#       return x, y

#return prepare_meta_batch_
def prepare_meta_batch(n, k, q, meta_batch_size):
    def prepare_meta_batch_(batch):
        x, y = batch
        # Reshape to `meta_batch_size` number of tasks. Each task contains
        # n*k support samples to train the fast model on and q*k query samples to
        # evaluate the fast model on and generate meta-gradients
        x = x.reshape(meta_batch_size, n*k + q*k, num_input_channels, x.shape[-2], x.shape[-1])
        # Move to device
        x = x.double().to(device)
        # Create label
        y = create_nshot_task_label(k, q).cuda().repeat(meta_batch_size)
        return x, y
    return prepare_meta_batch_

def infer(callbacks, model, loss_fn, optimiser):
#    import pdb; pdb.set_trace()
    num_batches = 1
#    batch_size = dataloader.batch_size

    # default call back averages the bach accuracy and loss
    callbacks = CallbackList((callbacks or []))
    # model and all other information has been passed to call back nothing else ot be done during function calls
    callbacks.set_model(model)
    callbacks.set_params({
        'loss_fn': loss_fn,
        'optimiser': optimiser
    })

   # creates a csv logger file
    callbacks.on_train_begin()
    callbacks.on_epoch_begin(1)
    epoch_logs = {}
    callbacks.on_epoch_end(1, epoch_logs)
#    import pdb; pdb.set_trace()


##############
# Parameters #
##############
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='fashion-dataset')
parser.add_argument('--distance', default='l2')
parser.add_argument('--n-train', default=5, type=int)
parser.add_argument('--n-test', default=5, type=int)
parser.add_argument('--k-train', default=5, type=int)
parser.add_argument('--k-test', default=5, type=int)
parser.add_argument('--q-train', default=5, type=int)
parser.add_argument('--q-test', default=5, type=int)
parser.add_argument('--exp_name', default='test', type=str)
parser.add_argument('--model_path', default='test', type=str)
parser.add_argument('--eval_classes', default='None' , type=str)
parser.add_argument('--network', default='proto', type=str)
args = parser.parse_args()

param_str = f'{args.network}_network_{args.exp_name}_experiment_name_{args.dataset}_nt={args.n_test}_kt={args.k_test}_qt={args.q_test}_' \
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

##############
# Create datasets #
###################
background = dataset_class('background')
# no batch size for proto nets

evaluation = dataset_class('evaluation')

if args.eval_classes == 'low_class':
    eval_classes = ['Shirts','Kurta Sets', 'Sweaters' , 'Sweatshirts', 'Night suits' ]
elif args.eval_classes == 'high_class':
    import random
    eval_classes = ['Bracelet', 'Tracksuits', 'Mask and Peel', 'Scarves', 'Sports Shoes']
else:
    eval_classes = None

evaluation_taskloader = DataLoader(
    evaluation,
   num_workers=4
)

#########
# Model #
#########
evaluation_episodes = 1000
episodes_per_epoch = 100

if args.network == 'proto':
    optimiser = Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.NLLLoss().cuda()

    n_epochs = 80
    dataset_class = FashionDataset
    num_input_channels = 3
    drop_lr_every = 40
    model = get_few_shot_encoder(num_input_channels)
    eval_fn = proto_net_episode
    evaluation_taskloader = DataLoader(evaluation, batch_sampler=NShotTaskSampler(evaluation, episodes_per_epoch, args.n_test, args.k_test, args.q_test, eval_classes=None), # why is qtest needed for protonet i think its not rquired for protonet check it
    num_workers=4
)
    callbacks = [
        EvaluateFewShot(
            eval_fn=eval_fn,
            num_tasks=evaluation_episodes,
            n_shot=args.n_test,
            k_way=args.k_test,
            q_queries=args.q_test,
            taskloader=evaluation_taskloader,
            prepare_batch=prepare_nshot_task(args.n_test, args.k_test, args.q_test), # n shot task is a simple function that maps classes to [0-k]
        distance=args.distance
    ),
]
elif args.network == 'matching':
    from few_shot.models import MatchingNetwork
    n_epochs = 200
    dataset_class = FashionDataset
    num_input_channels = 3
    lstm_input_size = 1600

    evaluation_taskloader = DataLoader(evaluation, batch_sampler=NShotTaskSampler(evaluation, episodes_per_epoch, args.n_test, args.k_test, args.q_test, eval_classes=eval_classes), # why is qtest needed for protonet i think its not rquired for protonet check it
    num_workers=4
)
    model = MatchingNetwork(args.n_train, args.k_train, args.q_train, True, num_input_channels,
                        lstm_layers=1,
                        lstm_input_size=lstm_input_size,
                        unrolling_steps=2,
                        device=device)
    optimiser = Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.NLLLoss().cuda()


    eval_fn = matching_net_episode

    callbacks = [    EvaluateFewShot(
        eval_fn=matching_net_episode,
        num_tasks=evaluation_episodes,
        n_shot=args.n_test,
        k_way=args.k_test,
        q_queries=args.q_test,
        taskloader=evaluation_taskloader,
        prepare_batch=prepare_nshot_task(args.n_test, args.k_test, args.q_test),
        fce= True,
        distance=args.distance)]
elif args.network == 'maml':
#    import pdb;pdb.set_trace()
    dataset_class = FashionDataset
    fc_layer_size = 1600
    num_input_channels = 3
    meta_lr = 0.001
    model = FewShotClassifier(num_input_channels, args.k_train, fc_layer_size).to(device, dtype=torch.double)
    optimiser = torch.optim.Adam(model.parameters(), lr=meta_lr)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    optimiser = Adam(model.parameters(), lr=1e-3)
    eval_batches=20
    meta_batch_size = 32
    batch_sampler=NShotTaskSampler(evaluation, eval_batches, n=args.n_test, k=args.k_test, q=args.q_test,
                                   num_tasks=meta_batch_size, eval_classes=eval_classes)
    evaluation_taskloader = DataLoader(evaluation, batch_sampler= batch_sampler, num_workers=8 )
    prepare_batch = prepare_meta_batch(args.n_test, args.k_test, args.q_test, meta_batch_size)
    callbacks = [
        EvaluateFewShot(
            eval_fn=meta_gradient_step,
            num_tasks=eval_batches,
            n_shot=args.n_test,
            k_way=args.k_test,
            q_queries=args.q_test,
            taskloader=evaluation_taskloader,
            prepare_batch=prepare_meta_batch(args.n_test, args.k_test, args.q_test, meta_batch_size),
            # MAML kwargs
            inner_train_steps=1,
            inner_lr=0.4,
            device=device,
            order=1,
            )
        ]

model.load_state_dict(torch.load(args.model_path))
model.to(device, dtype=torch.double)

############
# Training #
############
#print(f'Training Prototypical network on {args.dataset}...')

def lr_schedule(epoch, lr):
    # Drop lr every 2000 episodes
    if epoch % drop_lr_every == 0:
        return lr / 2
    else:
        return lr
infer(callbacks, model, loss_fn, optimiser)

