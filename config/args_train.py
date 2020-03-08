import argparse

parser = argparse.ArgumentParser(description="xxx", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# 模型信息
parser.add_argument('--dataset_name', type=str, default="Cora")
parser.add_argument('--model_name', type=str, default="GCNet")
parser.add_argument('--seed', default=1024, type=int, help="seef for random function and network initialization.")

# 读取与保存
parser.add_argument('--data_path', default='./data/cora/', metavar='DIR', help='path to dataset')
parser.add_argument('--pretrained', default=None, metavar='PATH')
parser.add_argument('--log_summary', default='progress_log_summary.csv', metavar='PATH')
parser.add_argument('--log_full', default='progress_log_full.csv', metavar='PATH')

# 网络训练
parser.add_argument('--no_cuda',default=False,type=bool)
parser.add_argument('--workers', '-j', default=8, type=int, metavar='N', help="number of data loading workers")
parser.add_argument('--epochs', default=200, type=int, metavar='N', help="number of total epochs to run")
parser.add_argument('-epoch_size', default=1000, type=int, metavar='N', help="manual epoch size")
parser.add_argument('--lr', default=0.01, type=float, metavar='LR', help="initial learning rate")
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help="momentum for sgd, alpha for adam")
parser.add_argument('--beta', default=0.999, type=float, metavar='M', help='beta parameters for adam')
parser.add_argument('--weight_decay', default=5e-4, type=float, metavar='W', help='weight decay')

# 模型超参（网络结构与损失函数）
parser.add_argument('--hidden',type=int, default=16,help="Number of hidden units")
parser.add_argument('--dropout',type=float,default=0.5,help="Dropout rate")


# 具体算法相关


# 是否为debug模式
parser.add_argument('--is_debug', type=bool, default=True)

args = parser.parse_args()
