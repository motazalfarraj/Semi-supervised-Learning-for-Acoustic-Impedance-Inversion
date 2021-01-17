import argparse
import numpy as np
import torch
from os.path import isdir
import os
from core.models import inverse_model, forward_model
from torch.utils import data
from core.functions import *
from torch import nn, optim
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import wget



#Manual seeds for reproducibility
random_seed=30
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)
np.random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_data(args, test=False):
    #Loading data
    try:
        data_dic = np.load("data/data.npy",allow_pickle=True).item()

    except FileNotFoundError:
        print("Data file not found. Downloading the data..")
        url= "https://www.dropbox.com/s/bvqfud2dv731v59/data.npy?raw=1"
        wget.download(url,"")

    seismic_data = data_dic["seismic"]
    acoustic_impedance_data = data_dic["acoustic_impedance"]

    assert seismic_data.shape[1]==acoustic_impedance_data.shape[1] ,'Data dimensions are not consistent. Got {} channels for seismic data and {} for acoustic acoustic impedance dimensions'.format(seismic_data.shape[1],acoustic_impedance_data.shape[1])
    assert seismic_data.shape[0]==acoustic_impedance_data.shape[0] ,'Number of traces is not consistent. Got {} traces for seismic data and {} traces for acoustic acoustic impedance'.format(seismic_data.shape[0],acoustic_impedance_data.shape[0])


    seismic_mean = torch.tensor(np.mean(seismic_data,keepdims=True)).float()
    seismic_std = torch.tensor(np.std(seismic_data,keepdims=True)).float()

    acoustic_mean= torch.tensor(np.mean(acoustic_impedance_data, keepdims=True)).float()
    acoustic_std = torch.tensor(np.std(acoustic_impedance_data,keepdims=True)).float()


    seismic_data = torch.tensor(seismic_data).float()
    acoustic_impedance_data = torch.tensor(acoustic_impedance_data).float()

    if torch.cuda.is_available():
        seismic_data = seismic_data.cuda()
        acoustic_impedance_data = acoustic_impedance_data.cuda()

        seismic_mean = seismic_mean.cuda()
        seismic_std = seismic_std.cuda()

        acoustic_mean = acoustic_mean.cuda()
        acoustic_std = acoustic_std.cuda()

    seismic_normalization = Normalization(mean_val=seismic_mean,
                                          std_val=seismic_std)

    acoustic_normalization = Normalization(mean_val=acoustic_mean,
                                          std_val=acoustic_std)


    seismic_data = seismic_normalization.normalize(seismic_data)
    acoustic_impedance_data = acoustic_normalization.normalize(acoustic_impedance_data)


    if not test:
        num_samples = seismic_data.shape[0]
        indecies = np.arange(0,num_samples)
        train_indecies = indecies[(np.linspace(0,len(indecies)-1,args.num_train_wells)).astype(int)]

        train_data = data.Subset(data.TensorDataset(seismic_data,acoustic_impedance_data), train_indecies)
        train_loader = data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False)

        unlabeled_loader = data.DataLoader(data.TensorDataset(seismic_data), batch_size=args.batch_size, shuffle=True)
        return train_loader, unlabeled_loader, seismic_normalization, acoustic_normalization
    else:
        test_loader = data.DataLoader(data.TensorDataset(seismic_data,acoustic_impedance_data), batch_size=args.batch_size, shuffle=False, drop_last=False)
        return test_loader, seismic_normalization, acoustic_normalization

def get_models(args):

    if args.test_checkpoint is None:
        inverse_net = inverse_model(nonlinearity=args.nonlinearity)
        forward_net = forward_model(nonlinearity=args.nonlinearity)
        optimizer = optim.Adam(list(inverse_net.parameters())+list(forward_net.parameters()), amsgrad=True,lr=0.005)
    else:
        try:
            inverse_net = torch.load(args.test_checkpoint + "_inverse")
            forward_net = torch.load(args.test_checkpoint + "_forward")
            optimizer = torch.load(args.test_checkpoint + "_optimizer")

        except FileNotFoundError:
            print("No checkpoint found at '{}'- Please specify the model for testing".format(args.test_checkpoint))
            exit()

    if torch.cuda.is_available():
        inverse_net.cuda()
        forward_net.cuda()

    return inverse_net, forward_net, optimizer

def train(args):

    #writer = SummaryWriter()
    train_loader, unlabeled_loader, seismic_normalization, acoustic_normalization = get_data(args)
    inverse_net, forward_net, optimizer = get_models(args)
    inverse_net.train()
    criterion = nn.MSELoss()

    #make a direcroty to save models if it doesn't exist
    if not isdir("checkpoints"):
        os.mkdir("checkpoints")

    print("Training the model")
    best_loss = np.inf
    for epoch in tqdm(range(args.max_epoch)):
        train_loss = []
        train_property_corr = []
        train_property_r2 = []
        for x,y in train_loader:
            optimizer.zero_grad()
            y_pred = inverse_net(x)
            x_rec = forward_net(y)

            property_loss = criterion(y_pred,y)+criterion(x_rec, x)
            corr, r2 = metrics(y_pred.detach(),y.detach())
            train_property_corr.append(corr)
            train_property_r2.append(r2)

            if args.beta!=0:
                #loading unlabeled data
                try:
                    x_u = next(unlabeled)[0]
                except:
                    unlabeled = iter(unlabeled_loader)
                    x_u = next(unlabeled)[0]

                y_u_pred = inverse_net(x_u)
                x_u_rec = forward_net(y_u_pred)

                seismic_loss = criterion(x_u_rec,x_u)
            else:
                seismic_loss=0
            loss = args.alpha*property_loss + args.beta*seismic_loss
            loss.backward()
            optimizer.step()

            train_loss.append(loss.detach().clone())

    torch.save(inverse_net,"checkpoints/{}_inverse".format(args.session_name))
    torch.save(forward_net,"checkpoints/{}_forward".format(args.session_name))
    torch.save(optimizer,"checkpoints/{}_optimizer".format(args.session_name))

def test(args):
    #make a direcroty to save precited sections
    if not isdir("output_images"):
        os.mkdir("output_images")

    test_loader, seismic_normalization, acoustic_normalization = get_data(args, test=True)
    if args.test_checkpoint is None:
        args.test_checkpoint = "checkpoints/{}".format(args.session_name)
    inverse_net, forward_net, _ = get_models(args)
    criterion = nn.MSELoss(reduction="sum")
    predicted_impedance = []
    true_impedance = []
    test_property_corr = []
    test_property_r2 = []
    inverse_net.eval()
    print("\nTesting the model\n")

    with torch.no_grad():
        test_loss = []
        for x,y in test_loader:
            y_pred = inverse_net(x)
            property_loss = criterion(y_pred,y)/np.prod(y.shape)
            corr, r2 = metrics(y_pred.detach(),y.detach())
            test_property_corr.append(corr)
            test_property_r2.append(r2)

            x_rec = forward_net(y_pred)
            seismic_loss = criterion(x_rec, x)/np.prod(x.shape)
            loss = args.alpha*property_loss + args.beta*seismic_loss
            test_loss.append(loss.item())

            true_impedance.append(y)
            predicted_impedance.append(y_pred)


        display_results(test_loss, test_property_corr, test_property_r2, args, header="Test")

        predicted_impedance = torch.cat(predicted_impedance, dim=0)
        true_impedance = torch.cat(true_impedance, dim=0)

        predicted_impedance = acoustic_normalization.unnormalize(predicted_impedance)
        true_impedance = acoustic_normalization.unnormalize(true_impedance)

        if torch.cuda.is_available():
            predicted_impedance = predicted_impedance.cpu()
            true_impedance = true_impedance.cpu()

        predicted_impedance = predicted_impedance.numpy()
        true_impedance = true_impedance.numpy()

        #diplaying estimated section

        plt.subplot(3,1,1)
        plt.imshow(predicted_impedance[:,0].T, cmap='rainbow',aspect=0.5, vmin=true_impedance.min(), vmax=true_impedance.max())
        plt.axis('off')
        plt.title("Estimated Acoustic Impedance")

        plt.subplot(3,1,2)
        plt.imshow(true_impedance[:,0].T, cmap='rainbow',aspect=0.5,vmin=true_impedance.min(), vmax=true_impedance.max())
        plt.title("True Acoustic Impedance")
        plt.axis('off')

        plt.subplot(3,1,3)
        plt.imshow(abs(true_impedance[:,0].T-predicted_impedance[:,0].T), cmap='gray',aspect=0.5)
        plt.title("Absolute Difference")
        plt.axis('off')

        plt.tight_layout()
        plt.savefig("output_images/{}.png".format(args.test_checkpoint.split("/")[-1]))

        plt.show()



if __name__ == '__main__':
    ## Arguments and parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_train_wells', type=int, default=20, help="Number of AI traces from the model to be used for training")
    parser.add_argument('-max_epoch', type=int, default=1000, help="maximum number of training epochs")
    parser.add_argument('-batch_size', type=int, default=50,help="Batch size for training")
    parser.add_argument('-alpha', type=float, default=1, help="weight of property loss term")
    parser.add_argument('-beta', type=float, default=0.2, help="weight of seismic loss term")
    parser.add_argument('-test_checkpoint', type=str, action="store", default=None,help="path to model to test on. When this flag is used, no training is performed")
    parser.add_argument('-session_name', type=str, action="store", default=datetime.now().strftime('%b%d_%H%M%S'),help="name of the session to be ised in saving the model")
    parser.add_argument('-nonlinearity', action="store", type=str, default="tanh",help="Type of nonlinearity for the CNN [tanh, relu]", choices=["tanh","relu"])

    ## Do not change these values unless you use the code on a different data and edit the code accordingly
    parser.add_argument('-resolution_ratio', type=int, default=4, action="store",help="resolution mismtach between seismic and AI")
    args = parser.parse_args()

    if args.test_checkpoint is not None:
        test(args)
    else:
        train(args)
        test(args)
