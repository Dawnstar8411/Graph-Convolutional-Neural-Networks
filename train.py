import csv
import time
import shutil
import datetime
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lr_scheduler

from path import Path
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter

import models
from config.args_train import *
#from utils import custom_transforms
from utils.utils import AverageMeter, save_path_formatter
from utils.utils import load_data,accuracy

warnings.filterwarnings('ignore')
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda") if args.cuda else torch.device("cpu")

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("1. Path to save the output.")
save_path = Path(save_path_formatter(args))
args.save_path = 'checkpoints' / save_path
args.save_path.makedirs_p()
print("=> will save everything to {}".format(args.save_path))

print("2.Data Loading...")
adj, features, labels, idx_train, idx_val, idx_test = load_data(path=args.data_path)

adj = adj.to(device)
features = features.to(device)
labels = labels.to(device)
idx_train = idx_train.to(device)
idx_val = idx_val.to(device)
idx_test = idx_test.to(device)

print("3.Creating Model")

gc_net = models.GCN(
    nfeat=features.shape[1],
    nhid=args.hidden,
    nclass=labels.max().item()+1,
    dropout=args.dropout).to(device)
# if args.pretrained:
#     print('=> using pre-trained weights for PoseNet')
#     weights = torch.load(args.pretrained)
#     gc_net.load_state_dict(weights['state_dict'], strict=False)
# else:
#     gc_net.init_weights()

print("4. Setting Optimization Solver")
optimizer = torch.optim.Adam(gc_net.parameters(), lr=args.lr, betas=(args.momentum, args.beta),
                             weight_decay=args.weight_decay)

#exp_lr_scheduler_R = lr_scheduler.StepLR(optimizer, step_size=100, gamma=5)

print("5. Start Tensorboard ")
# tensorboard --logdir=/path_to_log_dir/ --port 6006
#training_writer = SummaryWriter(args.save_path)

print("6. Create csvfile to save log information")

with open(args.save_path / args.log_summary, 'w') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter='\t')
    csv_writer.writerow(['train_loss', 'validation_loss'])

with open(args.save_path / args.log_full, 'w') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter='\t')
    csv_writer.writerow(['loss_train', 'acc_train', 'loss_val','acc_val'])

print("7. Start Training!")


def main():

    best_error = -1
    for epoch in range(args.epochs):
        start_time = time.time()
        losses, loss_names = train(gc_net, optimizer)
        errors, error_names = validate(gc_net)

        decisive_error = errors[1]
        if best_error < 0:
            best_error = decisive_error

        is_best = decisive_error < best_error
        best_error = min(best_error, decisive_error)

        torch.save({
            'epoch': epoch + 1,
            'state_dict': gc_net.state_dict()
        }, args.save_path / 'gc_net{}.pth.tar'.format(epoch))

        if is_best:
            shutil.copyfile(args.save_path / 'gc_net_{}.pth.tar'.format(epoch),
                            args.save_path / 'gc_net_best.pth.tar')

        # for loss, name in zip(losses, loss_names):
        #     training_writer.add_scalar(name, loss, epoch)
        #     training_writer.flush()
        # for error, name in zip(errors, error_names):
        #     training_writer.add_scalar(name, error, epoch)
        #     training_writer.flush()

        with open(args.save_path / args.log_summary, 'a') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter='\t')
            csv_writer.writerow([losses[0], decisive_error])

        with open(args.save_path / args.log_full, 'a') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter='\t')
            csv_writer.writerow([losses[0], losses[1], errors[0],errors[1]])

        print("\n---- [Epoch {}/{}] ----".format(epoch, args.epochs))
        print("Train---loss train:{}, acc train:{}".format(losses[0],losses[1]))
        print("Valid---loss val:{}, acc val:{}".format(errors[0],errors[1]))

        epoch_left = args.epochs - (epoch + 1)
        time_left = datetime.timedelta(seconds=epoch_left * (time.time() - start_time))
        print("----ETA {}".format(time_left))


def train(gc_net, optimizer):
    loss_names = ['loss train', 'acc train']
    losses = AverageMeter(i=len(loss_names), precision=4)

    gc_net.train()

    optimizer.zero_grad()
    output = gc_net(features,adj)
    loss_train = F.nll_loss(output[idx_train],labels[idx_train])
    acc_train = accuracy(output[idx_train],labels[idx_train])
    loss_train.backward()
    optimizer.step()


    losses.update([loss_train.item(), acc_train.item()])

    return losses.avg, loss_names


@torch.no_grad()
def validate(gc_net):
    error_names = ['loss val', 'acc val']
    losses = AverageMeter(i=len(error_names), precision=4)

    gc_net.eval()
    output = gc_net(features,adj)

    loss_val = F.nll_loss(output[idx_test],labels[idx_test])
    acc_val = accuracy(output[idx_test],labels[idx_test])

    losses.update([loss_val.item(), acc_val.item()])

    return losses.avg, error_names


if __name__ == '__main__':
    main()
