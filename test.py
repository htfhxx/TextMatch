# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 13:30:07 2020

@author: zhaog
"""
import torch
from sys import platform
from torch.utils.data import DataLoader
from projects.ABCNN_demo.util.data import LCQMC_Dataset, load_embeddings
from projects.ABCNN_demo.util.model import ABCNN
from projects.ABCNN_demo.util.utils import test
import argparse

def test_model(test_file, vocab_file, embeddings_file, pretrained_file, max_length, gpu_index, batch_size):

    device = torch.device("cuda:{}".format(gpu_index) if torch.cuda.is_available() else "cpu")
    print(20 * "=", " Preparing for testing ", 20 * "=")
    if platform == "linux" or platform == "linux2":
        checkpoint = torch.load(pretrained_file)
    else:
        checkpoint = torch.load(pretrained_file, map_location=device)
    # Retrieving model parameters from checkpoint.
    embeddings = load_embeddings(embeddings_file)
    print("\t* Loading test data...")
    test_data = LCQMC_Dataset(test_file, vocab_file, max_length)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
    print("\t* Building model...")
    model = ABCNN(embeddings, num_layer=1, linear_size=300, max_length=max_length, device=device).to(device)
    model.load_state_dict(checkpoint["model"])
    print(20 * "=", " Testing ABCNN model on device: {} ".format(device), 20 * "=")
    batch_time, total_time, accuracy, auc = test(model, test_loader)
    print("\n-> Average batch processing time: {:.4f}s, total test time: {:.4f}s, accuracy: {:.4f}%, auc: {:.4f}\n".format(batch_time, total_time, (accuracy*100), auc))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", default="data/BQ/test.tsv")
    parser.add_argument("--embedding_dir", default="util/token_vec_300.bin")
    parser.add_argument("--vocab_dir", default="util/vocab.txt")
    parser.add_argument("--laod_model_dir", default="checkpoints/epoch_best.tar")
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gpu_index", type=int, default=0)
    args = parser.parse_args()

    test_model(args.test_dir, args.vocab_dir, args.embedding_dir, args.laod_model_dir, max_length=args.max_len, gpu_index=args.gpu_index, batch_size=args.batch_size)


if __name__ == "__main__":
    main()