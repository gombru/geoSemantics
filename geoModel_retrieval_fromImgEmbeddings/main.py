import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import YFCC_dataset
import train
import model
from pylab import zeros, arange, subplots, plt, savefig

# Configg
training_id = 'geoModel_retrieval_fromEm_MCLL_NotNorm_WithLoc_SoftTriplets_smallTrain_lr0_02_NormLocEmGN'


dataset = '../../../hd/datasets/YFCC100M/'
split_train = 'train_filtered.txt'
split_val = 'val.txt'

img_embeddings_folder = 'YFCC_MCLL_2ndtraining_epoch_5_ValLoss_6.55'

margin = 1
norm_degree = 2

gpus = [0]
gpu = 0
workers = 0  # 8 Num of data loading workers
epochs = 100000
start_epoch = 0 # Useful on restarts
batch_size = 1024 # 1024 # Batch size
print_freq = 1 # An epoch are 60000 iterations. Print every 100: Every 40k images
resume = None # dataset + 'models/saved/' + 'geoModel_retrieval_fromEm_NCSLTr2_M2_NotNorm_LocTh750_lr0_001_LocEm_GN_WithLoc_smallTrain_epoch_37_ValLoss_0.37.pth.tar'  # Path to checkpoint top resume training
plot = True
best_epoch = 0
best_correct_pairs = 0
best_loss = 1000

train_iters = 2000 # 10000 # 5000
val_iters = 200 # 500  # 100

# Optimizer (SGD)
lr = 0.02 # 0.05 # 0.05 seems best # 0.001
lr_mult = 1
momentum = 0.9
weight_decay = 1e-4

# Loss
criterion = nn.TripletMarginLoss(margin=margin, p=norm_degree).cuda(gpu)
# Model
print("Initializing model")
model = model.Model(margin, norm_degree).cuda(gpu)
model = torch.nn.DataParallel(model, device_ids=gpus)

optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

# Optionally resume from a checkpoint
if resume:
    print("Loading pretrained model")
    print("=> loading checkpoint '{}'".format(resume))
    checkpoint = torch.load(resume, map_location={'cuda:1':'cuda:0', 'cuda:1':'cuda:0', 'cuda:3':'cuda:0'})
    model.load_state_dict(checkpoint, strict=False)
    print("Checkpoint loaded")

cudnn.benchmark = True

# Data loading code (pin_memory allows better transferring of samples to GPU memory)
train_dataset = YFCC_dataset.YFCC_Dataset(
    dataset,split_train,img_embeddings_folder)

val_dataset = YFCC_dataset.YFCC_Dataset(
    dataset, split_val,img_embeddings_folder)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=workers, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

# Plotting config
plot_data = {}
plot_data['train_loss'] = zeros(epochs)
plot_data['train_correct_triplets'] = zeros(epochs)
plot_data['val_loss'] = zeros(epochs)
plot_data['val_correct_triplets'] = zeros(epochs)
plot_data['epoch'] = 0
it_axes = arange(epochs)
_, ax1 = subplots()
ax2 = ax1.twinx()
ax1.set_xlabel('epoch')
ax1.set_ylabel('train loss (r), val loss (y)')
ax2.set_ylabel('train correct pairs (b), val correct pairs (g)')
ax2.set_autoscaley_on(False)
ax1.set_ylim([0, 1.02])
ax2.set_ylim([0, batch_size + 0.2])
prev_val_loss = 0

print("Dataset and model ready. Starting training ...")

for epoch in range(start_epoch, epochs):
    plot_data['epoch'] = epoch

    # Train for one epoch
    plot_data = train.train(train_loader, model, criterion, optimizer, epoch, print_freq, plot_data, train_iters)

    # Evaluate on validation set
    print("Validating ... (using train loader) ")
    plot_data = train.validate(val_loader, model, criterion, epoch, print_freq, plot_data, val_iters)


    # Remember best model and save checkpoint
    is_best = plot_data['val_loss'][epoch] < best_loss
    if is_best:
        print("New best model by loss. Val Loss = " + str(plot_data['val_loss'][epoch]))
        best_loss = plot_data['val_loss'][epoch]
        filename = dataset +'/models/' + training_id + '_epoch_' + str(epoch) + '_ValLoss_' + str(round(plot_data['val_loss'][epoch],2))
        prefix_len = len('_epoch_' + str(epoch) + '_ValLoss_' + str(round(plot_data['val_loss'][epoch],2)))
        train.save_checkpoint(model, filename)

    elif epoch > 0 and plot_data['val_loss'][epoch] > prev_val_loss:
        print("Model didn't improve Val Loss --> Multiplying lr by " + str(lr_mult))
        lr = lr * lr_mult
        print("Cur lr: " + str(lr))
        for g in optimizer.param_groups:
            g['lr'] = g['lr'] * lr_mult

    else:
        print("Continuing with lr " + str(lr) + "  - Not best model but improved previous epoch - ")

    prev_val_loss = plot_data['val_loss'][epoch]

    if plot:
        ax1.plot(it_axes[0:epoch+1], plot_data['train_loss'][0:epoch+1], 'r')
        ax2.plot(it_axes[0:epoch+1], plot_data['train_correct_triplets'][0:epoch+1], 'b')

        ax1.plot(it_axes[0:epoch+1], plot_data['val_loss'][0:epoch+1], 'y')
        ax2.plot(it_axes[0:epoch+1], plot_data['val_correct_triplets'][0:epoch+1], 'g')

        ax1.set_xlabel('epoch, lr ' + str(lr))

        plt.title(training_id)
        plt.ion()
        plt.grid(True)
        plt.show()
        plt.pause(0.001)

        # Save graph to disk
        if epoch % 1 == 0 and epoch != 0:
            title = dataset +'/training/' + training_id + '_epoch_' + str(epoch) + '.png'
            savefig(title, bbox_inches='tight')


print("Finished Training, saving checkpoint")
filename = dataset +'/models/' + training_id + '_epoch_' + str(epoch)
prefix_len = len('_epoch_' + str(epoch) + '_ValLoss_' + str(round(plot_data['val_loss'][epoch],2)))
train.save_checkpoint(model, filename)

