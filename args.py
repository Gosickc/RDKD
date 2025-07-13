import argparse
from easydict import EasyDict as edict
import json

parser = argparse.ArgumentParser(description='DMKD')
parser.add_argument('--Train', default=True, help='train or test', type=bool)
parser.add_argument('--Config', default='./config/MIRFlickr.json', help='Configure path MIRFlickr.json or NUSWIDE or MSCOCO', type=str)
parser.add_argument('--Dataset', default='MIRFlickr', help='MIRFlickr or NUSWIDE or MSCOCO', type=str)
# parser.add_argument('--Checkpoint', default='flickr_128bit.pth', help='checkpoint name', type=str)
#  parser.add_argument('--Bit', default=128, help='hash bit', type=int)

args = parser.parse_args()

# load basic settings
with open(args.Config, 'r', encoding='utf-8' ) as f:
    config = edict(json.load(f))

# update settings
config.TRAIN = args.Train
config.DATASET = args.Dataset
# config.CHECKPOINT = args.Checkpoint
# config.HASH_BIT = args.Bit
