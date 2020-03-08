import argparse

parser=argparse.ArgumentParser(description='xxx',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--pretrained',default='',type=str,help="pretrained model path")




args = parser.parse_args()