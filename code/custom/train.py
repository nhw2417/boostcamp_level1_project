import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset
from loss import create_criterion

import torchmetrics.functional as tmf

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(
        figsize=(12, 18 + 2)
    )  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(
        top=0.8
    )  # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = int(np.ceil(n**0.5))
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join(
            [
                f"{task} - gt: {gt_label}, pred: {pred_label}"
                for gt_label, pred_label, task in zip(
                    gt_decoded_labels, pred_decoded_labels, tasks
                )
            ]
        )

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"

###
'''
def calculate_f1_score(y_true, y_pred):
    epsilon = 1e-7  # 0으로 나누는 것을 방지하기 위한 작은 값

    # True Positives, False Positives, False Negatives 계산
    tp = (y_true * y_pred).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    # F1 Score 계산
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return f1
'''
###

def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(
        import_module("dataset"), args.dataset
    )  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir,
    )
    num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform_module = getattr(
        import_module("dataset"), args.augmentation
    )  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)

    # -- data_loader
    train_set, val_set = dataset.split_dataset()

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        #num_workers=multiprocessing.cpu_count() // 2,
        num_workers=0,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    model = model_module(num_classes=num_classes).to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4,
    )
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)
    
    ###
    #scaler = torch.cuda.amp.GradScaler()
    ###

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_acc = 0
    best_val_loss = np.inf
    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_value = 0
        matches = 0

        ###
        all_labels = []
        all_preds = []
        ###

        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            ###
            '''
            with torch.cuda.amp.autocast():
                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)
                loss = criterion(outs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            '''
            ###

            outs = model(inputs)
            preds = torch.argmax(outs, dim=-1)
            loss = criterion(outs, labels)

            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            matches += (preds == labels).sum().item()
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )
                logger.add_scalar(
                    "Train/loss", train_loss, epoch * len(train_loader) + idx
                )
                logger.add_scalar(
                    "Train/accuracy", train_acc, epoch * len(train_loader) + idx
                )

                loss_value = 0
                matches = 0

            ###
            all_labels.append(labels)
            all_preds.append(preds)
            ###
        ###
        all_labels = torch.cat(all_labels)
        all_preds = torch.cat(all_preds)
        #f1_score = calculate_f1_score(all_labels, all_preds)
        train_f1_micro = tmf.f1_score(all_preds, all_labels, "multiclass", num_classes=num_classes, average='micro')
        train_f1_macro = tmf.f1_score(all_preds, all_labels, "multiclass", num_classes=num_classes, average='macro')
        print(f"F1 Score on training set: micro {train_f1_micro} || macro {train_f1_macro}")
        
        ###

        scheduler.step()

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            figure = None
            
            ###
            all_labels = []
            all_preds = []
            ###
            
            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)
                
                ###
                all_labels.append(labels)
                all_preds.append(preds)
                ###

                loss_item = criterion(outs, labels).item()
                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

                if figure is None:
                    inputs_np = (
                        torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    )
                    inputs_np = dataset_module.denormalize_image(
                        inputs_np, dataset.mean, dataset.std
                    )
                    figure = grid_image(
                        inputs_np,
                        labels,
                        preds,
                        n=16,
                        shuffle=args.dataset != "MaskSplitByProfileDataset",
                    )

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            best_val_loss = min(best_val_loss, val_loss)
            if val_acc > best_val_acc:
                print(
                    f"New best model for val accuracy : {val_acc:4.2%}! saving the best model.."
                )
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_acc = val_acc
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )

            ###
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)
            #f1_score = calculate_f1_score(all_labels, all_preds)
            val_f1_micro = tmf.f1_score(all_preds, all_labels, "multiclass", num_classes=num_classes, average='micro')
            val_f1_macro = tmf.f1_score(all_preds, all_labels, "multiclass", num_classes=num_classes, average='macro')
            print(f"F1 Score on validation set: {val_f1_micro} || macro {val_f1_macro}")
            ###

            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            
            ###
            logger.add_scalar("F1 Score micro", val_f1_micro, epoch)
            logger.add_scalar("F1 Score macro", val_f1_macro, epoch)
            ###

            logger.add_figure("results", figure, epoch)
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="number of epochs to train (default: 1)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MaskBaseDataset",
        help="dataset augmentation type (default: MaskBaseDataset)",
    )
    parser.add_argument(
        "--augmentation",
        type=str,
        default="BaseAugmentation",
        help="data augmentation type (default: BaseAugmentation)",
    )
    parser.add_argument(
        "--resize",
        nargs=2,
        type=int,
        default=[128, 96],
        help="resize size for image when training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--valid_batch_size",
        type=int,
        default=1000,
        help="input batch size for validing (default: 1000)",
    )
    parser.add_argument(
        "--model", type=str, default="BaseModel", help="model type (default: BaseModel)"
    )
    parser.add_argument(
        "--optimizer", type=str, default="SGD", help="optimizer type (default: SGD)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="ratio for validaton (default: 0.2)",
    )
    parser.add_argument(
        "--criterion",
        type=str,
        default="cross_entropy",
        help="criterion type (default: cross_entropy)",
    )
    parser.add_argument(
        "--lr_decay_step",
        type=int,
        default=20,
        help="learning rate scheduler deacy step (default: 20)",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=40,
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--name", default="exp", help="model save at {SM_MODEL_DIR}/{name}"
    )

    # Container environment
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "/data/ephemeral/home/train/images"),
    )
    parser.add_argument(
        "--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./model")
    )

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)

