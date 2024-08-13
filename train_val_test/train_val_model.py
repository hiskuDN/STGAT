import torch
from torch.autograd import Variable
import torch.nn as nn
from tqdm import tqdm
from utility.log import IteratorTimer

# import torchvision
import numpy as np
import time
import random
import pickle
import cv2
from sklearn.metrics import confusion_matrix


def to_onehot(num_class, label, alpha):
    return (
        torch.zeros((label.shape[0], num_class))
        .fill_(alpha)
        .scatter_(1, label.unsqueeze(1), 1 - alpha)
    )


def mixup(input, target, gamma):
    # target is onehot format!
    perm = torch.randperm(input.size(0))
    perm_input = input[perm]
    perm_target = target[perm]
    return input.mul_(gamma).add_(1 - gamma, perm_input), target.mul_(gamma).add_(
        1 - gamma, perm_target
    )


def cutmix(input, target, beta=1.0):
    """
    Apply CutMix augmentation to the input data.

    Parameters:
    - input (torch.Tensor): The input data of shape (batch_size, channels, sequence_length, joints, 2).
    - target (torch.Tensor): The target labels in one-hot format.
    - beta (float): The parameter for the beta distribution used to sample the mix ratio.

    Returns:
    - input (torch.Tensor): The input data with CutMix applied.
    - target (torch.Tensor): The mixed target labels.
    """
    batch_size, channels, sequence_length, joints, _ = input.size()
    
    # Sample lambda from beta distribution
    lam = np.random.beta(beta, beta)
    
    # Randomly select another sample in the batch
    rand_index = torch.randperm(batch_size)
    
    # Get coordinates for the cutout box
    cutout_length = int(sequence_length * lam)
    cutout_joints = int(joints * lam)
    cutout_sequence_start = random.randint(0, sequence_length - cutout_length)
    cutout_joints_start = random.randint(0, joints - cutout_joints)
    
    # Create the new input and target
    input_cutmix = input.clone()
    for i in range(batch_size):
        # CutMix the input
        input_cutmix[i, :, cutout_sequence_start:cutout_sequence_start + cutout_length, 
                     cutout_joints_start:cutout_joints_start + cutout_joints, :] = \
                     input[rand_index[i], :, cutout_sequence_start:cutout_sequence_start + cutout_length, 
                           cutout_joints_start:cutout_joints_start + cutout_joints, :]
        
        # Mix the target labels
        target[i] = target[i] * lam + target[rand_index[i]] * (1 - lam)
    
    return input_cutmix, target


def random_cutout(input, cutout_length):
    """
    Apply random cutout to the input data.

    Parameters:
    - input (torch.Tensor): The input data of shape (batch_size, channels, sequence_length, joints, 2).
    - cutout_length (int): The length of the cutout segment.

    Returns:
    - input (torch.Tensor): The input data with random cutout applied.
    - target (torch.Tensor): The target labels unchanged.
    """
    batch_size, channels, sequence_length, joints, _ = input.size()

    for i in range(batch_size):
        # Randomly choose the start index for the cutout
        start_idx = random.randint(0, sequence_length - cutout_length)
        end_idx = start_idx + cutout_length

        # Apply cutout by setting the selected segment to zero
        input[i, :, start_idx:end_idx, :, :] = 0

    return input


def clip_grad_norm_(parameters, max_grad):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p[1].grad is not None, parameters))
    max_grad = float(max_grad)

    for name, p in parameters:
        grad = p.grad.data.abs()
        if grad.isnan().any():
            ind = grad.isnan()
            p.grad.data[ind] = 0
            grad = p.grad.data.abs()
        if grad.isinf().any():
            ind = grad.isinf()
            p.grad.data[ind] = 0
            grad = p.grad.data.abs()
        if grad.max() > max_grad:
            ind = grad > max_grad
            p.grad.data[ind] = p.grad.data[ind] / grad[ind] * max_grad  # sign x val


def train_classifier(
    data_loader, model, loss_function, optimizer, global_step, args, writer
):
    process = tqdm(IteratorTimer(data_loader), desc="Train: ", dynamic_ncols=True)
    loss_values = []
    for index, (inputs, labels) in enumerate(process):
        # if index == 0:
        # label_onehot = to_onehot(args.class_num, labels, args.label_smoothing_num)
        if args.mix_up_num > 0:
            # self.print_log('using mixup data: ', self.arg.mix_up_num)
            
            # mixup
            # targets = to_onehot(args.class_num, labels, args.label_smoothing_num)
            # inputs, targets = mixup(
            #     inputs, targets, np.random.beta(args.mix_up_num, args.mix_up_num)
            # )
            
            # random cutout
            aug_inputs = random_cutout(inputs, 1)
            aug_targets = labels
        elif args.label_smoothing_num != 0 or args.loss == "cross_entropy_naive":
            targets = to_onehot(args.class_num, labels, args.label_smoothing_num)
        else:
            targets = labels

        # random cutout, merge inputs with augmented inputs
        inputs = torch.cat((inputs, aug_inputs), dim=0)
        targets = torch.cat((targets, aug_targets), dim=0)
        
        # inputs, labels = Variable(inputs.cuda(non_blocking=True)), Variable(labels.cuda(non_blocking=True))
        inputs, targets, labels = (
            inputs.cuda(non_blocking=True),
            targets.cuda(non_blocking=True),
            labels.cuda(non_blocking=True),
        )
        # net = torch.nn.DataParallel(model, device_ids=args.device_id)
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip:
            clip_grad_norm_(model.named_parameters(), args.grad_clip)
        optimizer.step()
        global_step += 1
        if len(outputs.data.shape) == 3:  # T N cls
            _, predict_label = torch.max(outputs.data[:, :, :-1].mean(0), 1)
        else:
            _, predict_label = torch.max(outputs.data, 1)
        loss = loss_function(outputs, targets)
        ls = loss.data.item()
        loss_values.append(ls)
        acc = torch.mean((predict_label == labels.data).float()).item()
        # ls = loss.data[0]
        # acc = torch.mean((predict_label == labels.data).float())
        # lr = optimizer.param_groups[0]['lr']
        process.set_description(
            "           Train Acc: {:.4f}, batch time: {:.4f}".format(
                acc, process.iterable.last_duration
            )
        )

        # 每个batch记录一次
        if args.mode == "train_val":
            writer.add_scalar("acc", acc, global_step)
            writer.add_scalar("loss", ls, global_step)
            writer.add_scalar("batch_time", process.iterable.last_duration, global_step)
            # if len(inputs.shape) == 5:
            #     if index % 500 == 0:
            #         img = inputs.data.cpu().permute(2, 0, 1, 3, 4)
            #         # NCLHW->LNCHW
            #         img = torchvision.utils.make_grid(img[0::4, 0][0:4], normalize=True)
            #         writer.add_image('img', img, global_step=global_step)
            # elif len(inputs.shape) == 4:
            #     if index % 500 == 0:
            #         writer.add_image('img', ((inputs.cpu().numpy()[0] + 128) * 1).astype(np.uint8).transpose(1, 2, 0),
            #                          global_step=global_step)
    mean_loss = np.mean(loss_values)
    process.close()
    return global_step, mean_loss


def val_classifier(data_loader, model, loss_function, global_step, args, writer):
    right_num_total = 0
    total_num = 0
    loss_total = 0
    step = 0
    process = tqdm(IteratorTimer(data_loader), desc="Val: ", dynamic_ncols=True)
    # s = time.time()
    # t=0
    score_frag = []
    all_pre_true = []
    wrong_path_pre_ture = []
    video_labels = []
    video_pred = []
    for index, (inputs, labels, path) in enumerate(process):
        # label_onehot = to_onehot(args.class_num, labels, args.label_smoothing_num)
        video_labels.extend(labels)
        if args.loss == "cross_entropy_naive":
            targets = to_onehot(args.class_num, labels, args.label_smoothing_num)
        else:
            targets = labels

        with torch.no_grad():
            inputs, targets, labels = (
                inputs.cuda(non_blocking=True),
                targets.cuda(non_blocking=True),
                labels.cuda(non_blocking=True),
            )
            outputs = model(inputs)
            if len(outputs.data.shape) == 3:  # T N cls
                _, predict_label = torch.max(outputs.data[:, :, :-1].mean(0), 1)
                score_frag.append(outputs.data.cpu().numpy().transpose(1, 0, 2))
            else:
                _, predict_label = torch.max(outputs.data, 1)
                score_frag.append(outputs.data.cpu().numpy())
            loss = loss_function(outputs, targets)

        predict = list(predict_label.cpu().numpy())
        video_pred.extend(predict)
        true = list(labels.data.cpu().numpy())
        for i, x in enumerate(predict):
            all_pre_true.append(str(x) + "," + str(true[i]) + "\n")
            if x != true[i]:
                wrong_path_pre_ture.append(
                    str(path[i]) + "," + str(x) + "," + str(true[i]) + "\n"
                )

        right_num = torch.sum(predict_label == labels.data).item()
        # right_num = torch.sum(predict_label == labels.data)
        batch_num = labels.data.size(0)
        acc = right_num / batch_num
        ls = loss.data.item()
        # ls = loss.data[0]

        right_num_total += right_num
        total_num += batch_num
        loss_total += ls
        step += 1

        process.set_description(
            "           Val Acc: {:.4f}, time: {:.4f}".format(
                acc, process.iterable.last_duration
            )
        )
        # process.set_description_str(
        #     'Val: acc: {:4f}, loss: {:4f}, time: {:4f}'.format(t, t, t), refresh=False)
        # if len(inputs.shape) == 5:
        #     if index % 50 == 0 and (writer is not None) and args.mode == 'train_val':
        #         # NCLHW->LNCHW
        #         img = inputs.data.cpu().permute(2, 0, 1, 3, 4)
        #         img = torchvision.utils.make_grid(img[0::4, 0][0:4], normalize=True)
        #         writer.add_image('img', img, global_step=global_step)
        # elif len(inputs.shape) == 4:
        #     if index % 50 == 0 and (writer is not None) and args.mode == 'train_val':
        #         writer.add_image('img', ((inputs.cpu().numpy()[0] + 128) * 1).astype(np.uint8).transpose(1, 2, 0),
        #                          global_step=global_step)
    # t = time.time()-s
    # print('time: ', t)
    score = np.concatenate(score_frag)
    score_dict = dict(zip(data_loader.dataset.sample_name, score))
    cf = confusion_matrix(video_labels, video_pred).astype(float)
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_acc = cls_hit / cls_cnt

    process.close()
    loss = loss_total / step
    accuracy = right_num_total / total_num
    # print('Accuracy: ', accuracy)
    if args.mode == "train_val" and writer is not None:
        writer.add_scalar("loss", loss, global_step)
        writer.add_scalar("acc", accuracy, global_step)
        writer.add_scalar("batch time", process.iterable.last_duration, global_step)

    return loss, accuracy, score_dict, all_pre_true, wrong_path_pre_ture, cls_acc
