from dataset import *
import os, sys, numpy as np, argparse, time, gc, datetime, pickle as pkl, random
from tqdm import tqdm, trange
import torch, torch.nn as nn
import network as netlib
import auxiliaries as aux

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

device = torch.device('cuda')
def train_one_epoch(opt, epoch, net, optimizer, criterion, dataloader, Metrics):
    start = time.time()

    epoch_coll_loss, epoch_coll_acc = [],[]

    data_iter = tqdm(dataloader, desc='Training...')
    for iter_idx, file_dict in enumerate(data_iter):
        
        prediction  = net(file_dict['Tiles'].type(torch.FloatTensor).to(device))
        target = file_dict['Target']
        #print("Prediction : ",prediction.shape, "Target : ",target.shape)
        #print(torch.argmax(prediction,axis=1))
        #print(target)
        loss        = criterion(prediction, file_dict['Target'].type(torch.LongTensor).to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (np.argmax(prediction.cpu().detach().numpy(),axis=1)==target.cpu().detach().numpy()).reshape(-1)

        #--- Get Scores ---
        epoch_coll_acc.extend(list(acc))
        epoch_coll_loss.append(float(loss.item()))

        if iter_idx==len(dataloader)-1:
            data_iter.set_description('Epoch {0}: Loss [{1:.5f}] | Acc [{2:.3f}]'.format(epoch, np.mean(epoch_coll_loss), np.mean(epoch_coll_acc)))

    # Empty GPU cache
    torch.cuda.empty_cache()

    # Save Training Epoch Metrics
    Metrics['Train Loss'].append(np.round(np.mean(epoch_coll_loss),4))
    Metrics['Train Acc'].append(np.round(np.mean(epoch_coll_acc),4))
    Metrics['Train Time'].append(np.round(time.time()-start,4))



############## VALIDATOR ###############
def evaluate(opt, epoch, net, criterion, dataloader, Metrics, save_path):
    global best_val_acc
    start = time.time()

    epoch_coll_loss, epoch_coll_acc = [],[]

    data_iter = tqdm(dataloader, desc='Evaluating...')
    for iter_idx, file_dict in enumerate(data_iter):

        prediction  = net(file_dict['Tiles'].type(torch.FloatTensor).to(device))
        loss      = criterion(prediction, file_dict['Target'].type(torch.LongTensor).to(device))

        acc = (np.argmax(prediction.cpu().detach().numpy(), axis=1)==file_dict['Target'].cpu().detach().numpy()).reshape(-1)

        #--- Get Scores ---
        epoch_coll_acc.extend(list(acc))
        epoch_coll_loss.append(float(loss.item()))

        if iter_idx==len(dataloader)-1:
            data_iter.set_description('Epoch {0}: Loss [{1:.5f}] | Acc [{2:.5f}]'.format(epoch, np.mean(epoch_coll_loss), np.mean(epoch_coll_acc)))

    # Empty GPU cache
    torch.cuda.empty_cache()

    if np.mean(epoch_coll_acc)>best_val_acc:
        set_checkpoint(net, epoch, opt, save_path)
        best_val_acc = np.mean(epoch_coll_acc)

    # Save Training Epoch Metrics
    Metrics['Val Loss'].append(np.round(np.mean(epoch_coll_loss),4))
    Metrics['Val Acc'].append(np.round(np.mean(epoch_coll_acc),4))
    Metrics['Val Time'].append(np.round(time.time()-start,4))


############## CHECKPOINT SETTER ###############
def set_checkpoint(model, epoch, opt, save_path, progress_saver = ""):
    torch.save({'epoch': epoch+1, 'state_dict':model.state_dict(),
                'progress':progress_saver}, save_path+'/checkpoint_ep'+str(epoch) + '.pth.tar')
    
    

def main():
    
    jigsaw_config = read_config("configs/jigsaw_gold_panels.yaml")
    opt           = read_config("configs/training_configs.yaml")
    train_dataset  = GoldenPanels_Jigsaw_Dataset(images_path = jigsaw_config.panel_path,
                         annotation_path = jigsaw_config.panels_annotation,
                         permutation_path = jigsaw_config.permutation_path,                
                         panel_dim = jigsaw_config.panel_dim ,
                         num_panels = jigsaw_config.num_panels,
                         train_test_ratio = jigsaw_config.train_test_ratio,
                         normalize = False,
                         train_mode = True,
                         limit_size = -1)
    
    val_dataset  = GoldenPanels_Jigsaw_Dataset(images_path = jigsaw_config.panel_path,
                         annotation_path = jigsaw_config.panels_annotation,
                         permutation_path = jigsaw_config.permutation_path,                
                         panel_dim = jigsaw_config.panel_dim ,
                         num_panels = jigsaw_config.num_panels,
                         train_test_ratio = jigsaw_config.train_test_ratio,
                         normalize = False,
                         train_mode = False,
                         limit_size = -1)
    
    bcolor = aux.bcolors
    # Data Loader
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.bs, num_workers=opt.kernels, shuffle=True, pin_memory=False)
    val_dataloader   = torch.utils.data.DataLoader(val_dataset,   batch_size= int(opt.bs/2), num_workers=opt.kernels, pin_memory=False)
    
    
    date = datetime.datetime.now()
    time_string = '{}-{}-{}-{}-{}-{}'.format(date.year, date.month, date.day, date.hour, date.minute, date.second)
    checkfolder = os.getcwd() + "/" + opt.save_path+'/{}_JigsawNetwork_'.format(opt.dataset)+time_string
    counter     = 1
    while os.path.exists(checkfolder):
        checkfolder = opt.save_path+'_'+str(counter)
        counter += 1
    os.makedirs(checkfolder)
    save_path = checkfolder
    
    
    #################### SAVE OPTIONS TO TXT ################
    with open(save_path+'/Parameter_Info.txt','w') as f:
        f.write(aux.gimme_save_string(opt))
    #pkl.dump(opt,open(save_path+"/hypa.pkl","wb"))
    
    
    #################### CREATE LOGGING FILES ###############
    InfoPlotter   = aux.InfoPlotter(save_path+'/InfoPlot.svg')
    full_log      = aux.CSV_Writer(save_path +'/log_epoch.csv', ['Epoch', 'Train Loss', 'Val Loss', 'Val Acc'])
    Progress_Saver= {'Train Loss':[], 'Val NMI':[], 'Val Recall Sum':[]}
   
    
     #################### SETUP JIGSAW NETWORK ###################
    device = torch.device('cuda')
    model = netlib.NetworkSelect(opt)
    print('JIGSAW Setup for [{}] complete with #weights: {}'.format(opt.arch, aux.gimme_params(model)))
    _ = model.to(device)

    global best_val_acc
    best_val_acc = 0


    ################### SET OPTIMIZATION SETUP ##################
    criterion  = nn.CrossEntropyLoss()
    optimizer  = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, opt.tau, gamma=opt.gamma)
    print('Done.')
    
############## START TRAINING ###########################
    print(bcolor.BOLD+bcolor.WARNING+'Starting Jigsaw Network Training!\n'+bcolor.ENDC+bcolor.ENDC)

    Metrics = {}
    Progress_Saver = {}

    Metrics['Train Loss'] = []
    Metrics['Val Loss']   = []
    Metrics["Train Acc"]  = []
    Metrics["Val Acc"]    = []
    Metrics['Train Time'] = []
    Metrics['Val Time']   = []

    Progress_Saver['Train Loss'] = []
    Progress_Saver['Val Loss']   = []
    Progress_Saver["Train Acc"]  = []
    Progress_Saver["Val Acc"]    = []
    Progress_Saver['Train Time'] = []
    Progress_Saver['Val Time']   = []
    

    for epoch in range(opt.n_epochs):
        scheduler.step()

        ### Training ###
        train_one_epoch(opt, epoch, model, optimizer, criterion, train_dataloader, Metrics)
        
        
        
        ### Validating ###
        evaluate(opt, epoch, model, criterion, val_dataloader, Metrics, save_path)
        
        
        #Save Model
        set_checkpoint(model, epoch, opt,save_path)
        
        ###### Logging Epoch Data ######
        full_log.log([len(Metrics['Train Loss']), Metrics["Train Loss"][-1], Metrics["Train Acc"][-1], Metrics["Val Acc"][-1]])


        ###### Generating Summary Plots #######
        InfoPlotter.make_plot(range(epoch+1), Metrics['Train Loss'], Metrics['Val Acc'], ['Train Loss', 'Val Acc'])



if __name__ == "__main__":
    main()
