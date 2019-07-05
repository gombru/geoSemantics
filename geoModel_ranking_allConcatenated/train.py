import time
import torch
import torch.nn.parallel
import glob
import os
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

def save_checkpoint(model, filename, prefix_len):
    print("Saving Checkpoint")
    # for cur_filename in glob.glob(filename[:-prefix_len] + '*'):
    #     print(cur_filename)
    #     os.remove(cur_filename)
    torch.save(model.state_dict(), filename + '.pth.tar')

def train(train_loader, model, criterion, optimizer, epoch, print_freq, plot_data, gpu):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    correct_pairs = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (img_p, tag_p, lat_p, lon_p, img_n, tag_n, lat_n, lon_n) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        img_p_var = torch.autograd.Variable(img_p)
        tag_p_var = torch.autograd.Variable(tag_p)
        lat_p_var = torch.autograd.Variable(lat_p)
        lon_p_var = torch.autograd.Variable(lon_p)
        img_n_var = torch.autograd.Variable(img_n)
        tag_n_var = torch.autograd.Variable(tag_n)
        lat_n_var = torch.autograd.Variable(lat_n)
        lon_n_var = torch.autograd.Variable(lon_n)

        # compute output
        s_p, s_n, correct = model(img_p_var, tag_p_var, lat_p_var, lon_p_var, img_n_var, tag_n_var, lat_n_var, lon_n_var)

        y = torch.ones(img_p_var.size()[0]).cuda(gpu, async=True)
        # If `y == 1` then it assumed the first input should be ranked higher (have a larger value) than the second input, and vice-versa for `y == -1`.
        loss = criterion(s_p, s_n, y)

        # measure and record loss
        loss_meter.update(loss.data.item(), img_p_var.size()[0])
        correct_pairs.update(torch.sum(correct))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if correct_pairs.avg >= len(img_p) - 1 and i % 5000 == 0:
            print("Correct pairs avg: " + str(correct_pairs.avg) + " --> Saving model")
            filename = '../../../datasets/YFCC100M/' + '/models/' + 'geoModel_ranking_allConcatenated_randomTriplets' + '_iter_' + str(i) + '_TrainLoss_' + str(round(loss.data.item(), 2))
            prefix_len = len(str(i) + '_TrainLoss_' + str(round(loss.data.item(), 2)))
            save_checkpoint(model, filename, prefix_len)


        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Correct Pairs {correct_pairs.val:.3f} ({correct_pairs.avg:.3f})\t'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=loss_meter, correct_pairs=correct_pairs))

    plot_data['train_loss'][plot_data['epoch']] = loss_meter.avg
    plot_data['train_correct_pairs'][plot_data['epoch']] = correct_pairs.avg


    return plot_data


def validate(val_loader, model, criterion, print_freq, plot_data, gpu):
    with torch.no_grad():

        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        correct_pairs = AverageMeter()

        # switch to evaluate mode
        model.eval()

        end = time.time()
        for i, (img_p, tag_p, lat_p, lon_p, img_n, tag_n, lat_n, lon_n) in enumerate(val_loader):

            img_p_var = torch.autograd.Variable(img_p)
            tag_p_var = torch.autograd.Variable(tag_p)
            lat_p_var = torch.autograd.Variable(lat_p)
            lon_p_var = torch.autograd.Variable(lon_p)
            img_n_var = torch.autograd.Variable(img_n)
            tag_n_var = torch.autograd.Variable(tag_n)
            lat_n_var = torch.autograd.Variable(lat_n)
            lon_n_var = torch.autograd.Variable(lon_n)

            # compute output
            s_p, s_n, correct = model(img_p_var, tag_p_var, lat_p_var, lon_p_var, img_n_var, tag_n_var, lat_n_var, lon_n_var)

            y = torch.ones(img_p_var.size()[0]).cuda(gpu, async=True)  # Flag to indicate that all are positive pairs
            loss = criterion(s_p, s_n, y)

            # measure and record loss
            # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            loss_meter.update(loss.data.item(), img_p_var.size()[0])
            correct_pairs.update(torch.sum(correct))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Correct Pairs {correct_pairs.val:.3f} ({correct_pairs.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=loss_meter,
                       correct_pairs=correct_pairs))

        plot_data['val_loss'][plot_data['epoch']] = loss_meter.avg
        plot_data['val_correct_pairs'][plot_data['epoch']] = correct_pairs.avg

    return plot_data



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count