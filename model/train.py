import time
import torch
import torch.nn.parallel
import glob
import os
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

def train(train_loader, model, criterion, optimizer, epoch, print_freq, plot_data, gpu):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    correct_triplets = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (image, tag_pos, tag_neg, lat, lon) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(gpu, async=True)
        image_var = torch.autograd.Variable(image)
        tag_pos_var = torch.autograd.Variable(tag_pos)
        tag_neg_var = torch.autograd.Variable(tag_neg)

        # compute output
        i_e, tp_e, tn_e, correct = model(image_var, tag_pos_var, tag_neg_var)
        loss = criterion(i_e, tp_e, tn_e)

        # measure and record loss
        loss_meter.update(loss.data.item(), image.size()[0])
        correct_triplets.update(torch.sum(correct))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'GPU: {gpu}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Correct Triplets {correct_triplets.val.data[0]:.3f} ({correct_triplets.avg.data[0]:.3f})\t'
                  .format(
                   epoch, i, len(train_loader), gpu=str(gpu), batch_time=batch_time,
                   data_time=data_time, loss=loss_meter, correct_triplets=correct_triplets))

    plot_data['train_loss'][plot_data['epoch']] = loss_meter.avg
    plot_data['train_correct_triplets'][plot_data['epoch']] = correct_triplets.avg


    return plot_data


def validate(val_loader, model, criterion, print_freq, plot_data, gpu):
    with torch.no_grad():

        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        correct_triplets = AverageMeter()

        # switch to evaluate mode
        model.eval()

        end = time.time()
        for i, (image, tag_pos, tag_neg, lat, lon) in enumerate(val_loader):

            target = target.cuda(gpu, async=True)
            image_var = torch.autograd.Variable(image)
            tag_pos_var = torch.autograd.Variable(tag_pos)
            tag_neg_var = torch.autograd.Variable(tag_neg)

            # compute output
            i_e, tp_e, tn_e, correct = model(image_var, tag_pos_var, tag_neg_var)
            loss = criterion(i_e, tp_e, tn_e)

            # measure and record loss
            # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            loss_meter.update(loss.data.item(), image.size()[0])
            correct_triplets.update(torch.sum(correct))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Correct Triplets {correct_triplets.val.data[0]:.3f} ({correct_triplets.avg.data[0]:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=loss_meter,
                       correct_triplets=correct_triplets))

        plot_data['val_loss'][plot_data['epoch']] = loss_meter.avg
        plot_data['val_correct_triplets'][plot_data['epoch']] = correct_triplets.avg

    return plot_data


def save_checkpoint(model, filename, prefix_len):
    print("Saving Checkpoint")
    for cur_filename in glob.glob(filename[:-prefix_len] + '*'):
        print(cur_filename)
        os.remove(cur_filename)
    torch.save(model.state_dict(), filename + '.pth.tar')

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