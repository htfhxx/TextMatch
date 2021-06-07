# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 13:27:48 2020

@author: zhaog
"""
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from projects.ABCNN_demo.util.data import LCQMC_Dataset, load_embeddings
from projects.ABCNN_demo.util.utils import train, validate
from projects.ABCNN_demo.util.model import ABCNN
import argparse

def train_model(train_file, dev_file, embeddings_file, vocab_file, target_dir, max_length, epochs, batch_size, lr, patience, max_grad_norm, gpu_index, checkpoint):
    device = torch.device("cuda:{}".format(gpu_index) if torch.cuda.is_available() else "cpu")
    print(20 * "=", " Preparing for training ", 20 * "=")
    # 保存模型的路径
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    # -------------------- Data loading ------------------- #
    print("\t* Loading training data...")
    train_data = LCQMC_Dataset(train_file, vocab_file, max_length)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    print("\t* Loading validation data...")
    dev_data = LCQMC_Dataset(dev_file, vocab_file, max_length)
    dev_loader = DataLoader(dev_data, shuffle=True, batch_size=batch_size)
    # -------------------- Model definition ------------------- #
    print("\t* Building model...")
    embeddings = load_embeddings(embeddings_file)
    model = ABCNN(embeddings, num_layer=1, linear_size=300, max_length=max_length, device=device).to(device)
    # -------------------- Preparation for training  ------------------- #
    criterion = nn.CrossEntropyLoss()
    # 过滤出需要梯度更新的参数
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    # optimizer = optim.Adadelta(parameters, params["LEARNING_RATE"])
    optimizer = torch.optim.Adam(parameters, lr=lr)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", 
                                                           factor=0.85, patience=0)
    best_score = 0.0
    start_epoch = 1
    # Data for loss curves plot
    epochs_count = []
    train_losses = []
    valid_losses = []
    # Continuing training from a checkpoint if one was given as argument
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]
        print("\t* Training will continue on existing model from epoch {}...".format(start_epoch))
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epochs_count = checkpoint["epochs_count"]
        train_losses = checkpoint["train_losses"]
        valid_losses = checkpoint["valid_losses"]
     # Compute loss and accuracy before starting (or resuming) training.
    _, valid_loss, valid_accuracy, auc = validate(model, dev_loader, criterion)
    print("\t* Validation loss before training: {:.4f}, accuracy: {:.4f}%, auc: {:.4f}".format(valid_loss, (valid_accuracy*100), auc))
    # -------------------- Training epochs ------------------- #
    print("\n", 20 * "=", "Training ABCNN model on device: {}".format(device), 20 * "=")
    patience_counter = 0
    for epoch in range(start_epoch, epochs + 1):
        epochs_count.append(epoch)
        print("* Training epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = train(model, train_loader, optimizer,
                                                       criterion, epoch, max_grad_norm)
        train_losses.append(epoch_loss)
        print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%"
              .format(epoch_time, epoch_loss, (epoch_accuracy*100)))
        print("* Validation for epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy , epoch_auc= validate(model, dev_loader, criterion)
        valid_losses.append(epoch_loss)
        print("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%, auc: {:.4f}\n"
              .format(epoch_time, epoch_loss, (epoch_accuracy*100), epoch_auc))
        # Update the optimizer's learning rate with the scheduler.
        scheduler.step(epoch_accuracy)
        # Early stopping on validation accuracy.
        if epoch_accuracy < best_score:
            patience_counter += 1
        else:
            best_score = epoch_accuracy
            patience_counter = 0
            torch.save({"epoch": epoch, 
                        "model": model.state_dict(),
                        "best_score": best_score,
                        "epochs_count": epochs_count,
                        "train_losses": train_losses,
                        "valid_losses": valid_losses},
                        os.path.join(target_dir, "epoch_"+str(epoch)+".tar"))
            torch.save({"epoch": epoch,
                        "model": model.state_dict(),
                        "best_score": best_score,
                        "epochs_count": epochs_count,
                        "train_losses": train_losses,
                        "valid_losses": valid_losses},
                        os.path.join(target_dir, "epoch_best.tar"))
        if patience_counter >= patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default="data/BQ/train.tsv")
    parser.add_argument("--dev_dir", default="data/BQ/dev.tsv")
    parser.add_argument("--embedding_dir", default="util/token_vec_300.bin")
    parser.add_argument("--vocab_dir", default="util/vocab.txt")
    parser.add_argument("--checkpoints_dir", default="checkpoints")
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max_grad_norm", type=float, default=10.0)
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()



    train_model(args.train_dir, args.dev_dir, args.embedding_dir, args.vocab_dir, args.checkpoints_dir,
          max_length = args.max_len,
          epochs=args.epochs,
          batch_size=args.batch_size,
          lr=args.learning_rate,
          patience=args.patience,
          max_grad_norm=args.max_grad_norm,
          gpu_index=args.gpu_index,
          checkpoint=args.checkpoint)



if __name__ == "__main__":
    main()

