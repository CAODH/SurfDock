
import os
from argparse import ArgumentParser
import torch
from tqdm import tqdm


parser = ArgumentParser()
parser.add_argument('--esm_embeddings_path', type=str, default='~/esm_embedding/esm_embedding_output_pocket_new', help='')
parser.add_argument('--output_path', type=str, default='~/esm_embedding/esm_embedding_pocket_for_train_new/esm2_3billion_embeddings.pt', help='')
args = parser.parse_args()
os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
# args = parser.parse_args()
dict = {}
for filename in tqdm(os.listdir(args.esm_embeddings_path)):
    try:
        dict[os.path.splitext(os.path.basename(filename))[0]] = torch.load(os.path.join(args.esm_embeddings_path,filename))
    except:
        print(filename)
torch.save(dict,args.output_path)