import yaml
from collections import namedtuple
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def panel_sqrtize(pw, ph, w_h_ratio = 1):
    if pw / ph >= w_h_ratio:
        w, h = int(ph * w_h_ratio), int(ph)
    elif pw / ph < w_h_ratio:
        w, h = int(pw), int(pw / w_h_ratio)
        
    area = [pw/2 - w/2, ph/2 - h/2, pw/2 + w/2, ph/2 + h/2]
    return area

def panel_transforms(panel, panel_dim, augment):
    # Calculate square coordinates
    p_area = panel_sqrtize(*panel.size)
    if augment:
        panel = distort_color(panel)
    # Crop to square size
    panel = TF.crop(panel, p_area[1], p_area[0], p_area[3]-p_area[1], p_area[2]-p_area[0])
    # Scale 0-1
    panel = transforms.ToTensor()(panel).unsqueeze(0)
    #Resizes
    panel = TF.resize(panel, [panel_dim[1], panel_dim[0]])
    return panel

### MISC
def color_jitter(x):
    x = np.array(x, 'int32')
    for ch in range(x.shape[-1]):
        x[:,:,ch] += np.random.randint(-2,2)
    x[x>255] = 255
    x[x<0]   = 0
    return x.astype('uint8')


def read_config(path):
    with open(path) as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)
        configs = namedtuple("Config", configs.keys())(*configs.values())
    return configs

def plot_resorted_image(dataset,index ):
    data = dataset[index]
    #print(data["path"])
    tiles = data["Tiles"]
    _, h, w = tiles[0].shape
    target = data["Target"]
    mean = data["tile_mean"]
    std = data["tile_std"]
    cur_perm = dataset.permutations[target]
    print("Current Permutation",cur_perm)
    resorted = [tiles[np.where(cur_perm == ind)[0][0]].unsqueeze(0) for (ind,per) in enumerate(cur_perm)]
    recon_images = torch.zeros(1,3,h*3, w*3)
    for i in range(len(resorted)):
        col = i//3
        row = i%3
        recon_images[0][:,row*h : row*h + h, col*w:col*w+w ] = resorted[i]
        
    
    means_CelebA   = torch.tensor(mean).unsqueeze(1).unsqueeze(2)
    stds_CelebA    = torch.tensor(std).unsqueeze(1).unsqueeze(2)
    unnormalized = recon_images.squeeze(0)*stds_CelebA + means_CelebA

    unnormalized = transforms.ToPILImage()(unnormalized)
    
    return unnormalized,tiles



    

def getPILImage(tile, mean=[0.5, 0.5, 0.5], std= [0.5, 0.5, 0.5]):
    means   = torch.tensor(mean).unsqueeze(1).unsqueeze(2)
    stds    = torch.tensor(std).unsqueeze(1).unsqueeze(2)
    
    unnormalized = tile*stds + means
    
    return transforms.ToPILImage()(unnormalized)




def plot_tiles(dataset, index):
    data = dataset[index]
    #print(data["path"])
    tiles = data["Tiles"]  #torch.Size([9, 3, 100, 100])
    num_tiles, _, h, w = tiles.shape
    target = data["Target"] 
    #mean = data["tile_mean"]
    #std = data["tile_std"]
    
    
    #num_tiles = tiles.shape[0] # 3*3
    #print("Num Tiles", num_tiles)
    wsize, hsize = 3,3
    w = (w + 100) * wsize
    h = (h + 50) * hsize
    px = 1/plt.rcParams['figure.dpi']
    f, ax = plt.subplots(hsize, wsize)
    f.set_size_inches(w*px, h*px)
    
    #print(ax)
    for i in range(num_tiles):
        ax[i%3][i//3].imshow(getPILImage(tiles[i]))
        #ax[i//3][i%3].title.set_text("Patch" + str(i+1))
    
    plt.show()
    
    
    
def plot_resorted_image_saliency(data,dataset, saliency_list):
    
    #print(data["path"])
    tiles = data["Tiles"]
    _,num_tiles,_, h, w = tiles.shape
    tiles = data["Tiles"][0]
    target = data["Target"]
    cur_perm = dataset.permutations[target]
    #print("Current Permutation",cur_perm)
    
    resorted = []
    #print("Tiles.shape",tiles.shape)
    
    resorted = [saliency_list[np.where(cur_perm == ind)[0][0]].unsqueeze(0) for (ind,per) in enumerate(cur_perm)]
    recon_images = torch.zeros(1,h*3, w*3)
    for i in range(len(resorted)):
        col = i//3
        row = i%3
        recon_images[0][row*h : row*h + h, col*w:col*w+w ] = resorted[i]
        
    
    return recon_images


def draw_jigsaw_saliency(model, batch, dataset):
    bs, num_patch,ch, h, w = batch["Tiles"].shape
    px = 1/plt.rcParams["figure.dpi"]
    fig, ax = plt.subplots(3, 9)
    fig.set_size_inches(18.5, 10.5)
    
    imgs = batch["Tiles"]
    
    imgs.requires_grad_()
    outs = model(imgs.cuda())
    out = outs.sum()
    out.backward()
    
    g_imgs = imgs.grad
    
    saliencies = []

    for i in range(9):
        
        saliency, _ = torch.max(g_imgs[0,i,:,:,:].data.abs(), dim=0) 
        saliency = saliency.reshape(h, w)
        #print(saliency.shape)
        init_img = getPILImage(imgs[0,i,:,:,:])
        # Visualize the image and the saliency map
        #ax[i//3][i%3]
        ax[0, i].imshow(init_img)
        ax[0, i].axis('off')
        ax[0, i].title.set_text("Panel" + str(i+1))
        ax[1, i].imshow(saliency.cpu(), cmap='hot')
        ax[1, i].axis('off')
        ax[2, i].imshow(init_img)
        ax[2, i].imshow(saliency.cpu(), cmap='hot', alpha=0.7)
        ax[2, i].axis('off')
        saliencies.append(saliency.cpu())
    
    
    
    #resorted_saliency = plot_resorted_image_saliency(batch, dataset, saliencies)[0]
    #ax[3,0].imshow(plot_resorted_image(batch,dataset))
    #ax[3,1].imshow(resorted_saliency, cmap='hot')  
    #ax[3,2].imshow(plot_resorted_image(batch,dataset))
    #ax[3,2].imshow(resorted_saliency, cmap='hot', alpha=0.7)
    
    

    
def newPlot(model, batch, dataset):
    bs, num_patch,ch, h, w = batch["Tiles"].shape
    px = 1/plt.rcParams["figure.dpi"]
    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(18.5, 10.5)
    
    imgs = batch["Tiles"]
    
    imgs.requires_grad_()
    outs = model(imgs.cuda())
    out = outs.sum()
    out.backward()
    
    g_imgs = imgs.grad
    
    saliencies = []

    for i in range(9):
        
        saliency, _ = torch.max(g_imgs[0,i,:,:,:].data.abs(), dim=0) 
        saliency = saliency.reshape(h, w)
        #print(saliency.shape)
        init_img = getPILImage(imgs[0,i,:,:,:])
        # Visualize the image and the saliency map
        #ax[i//3][i%3]
        saliencies.append(saliency.cpu())
    
    
    
    resorted_saliency = plot_resorted_image_saliency(batch, dataset, saliencies)[0]
    ax[0].imshow(plot_resorted_image(batch,dataset))
    ax[1].imshow(resorted_saliency, cmap='hot')  
    ax[2].imshow(plot_resorted_image(batch,dataset))
    ax[2].imshow(resorted_saliency, cmap='hot', alpha=0.7)
    
    
def plot_resorted_image(data,dataset):
    
    #print(data["path"])
    tiles = data["Tiles"]
    _,num_tiles,_, h, w = tiles.shape
    tiles = data["Tiles"][0]
    target = data["Target"]
    cur_perm = dataset.permutations[target]
    print("Current Permutation",cur_perm)
    
    resorted = []
    #print("Tiles.shape",tiles.shape)
    
    resorted = [tiles[np.where(cur_perm == ind)[0][0]].unsqueeze(0) for (ind,per) in enumerate(cur_perm)]
    recon_images = torch.zeros(1,3,h*3, w*3)
    for i in range(len(resorted)):
        col = i//3
        row = i%3
        recon_images[0][:,row*h : row*h + h, col*w:col*w+w ] = resorted[i]
        
    
    print("recon_images shpape : ",recon_images.shape)
    unnormalized = getPILImage(recon_images[0])

    
    
    return unnormalized