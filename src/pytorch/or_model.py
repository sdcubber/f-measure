import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch.optim as optim
import models.evaluator as eval
import torchvision.models as models
import models.gfm as gfm
from sklearn.metrics import fbeta_score
from torch.autograd import Variable
import torch
import sys

def get_num_learnable_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return("NUMBER OF LEARNABLE PARAMS: " + str(params))

class LLORLoss(torch.nn.Module):
    def __init__(self, nclass, sm, l=0.01, eps=1e-10):
        super(LLORLoss, self).__init__()
        self.nclass = nclass
        self.sm = sm
        self.l = l
        self.eps = eps

    def forward(self,input,target,or_bias):
        # first transform our target matrix
        newTarget = self.transformTarget(target).cuda()

        # clamp our input in order to avoid 0's or negative values and take log
        newInput = input.clamp(min=self.eps).log()

        # store copy of or_bias matrix for log. barrier penalty term
        ob_d = or_bias.clone()

        LL1 = torch.mul(newInput,newTarget).sum()
        LL2 = (ob_d[:,1:self.sm-1]-or_bias[:,0:self.sm-2]).clamp(min=self.eps).log().sum()

        # multiply with new target and take sum
        return (LL1 + (self.l * LL2)) * -1

    def transformTarget(self,target):
        # create our new target tensor
        newTarget = torch.zeros(target.shape[0],self.nclass,self.sm)

        # run over batch
        for b in range(target.shape[0]):
            # calculate sy
            sy = len(target[b,:].nonzero())
            # now create sy indicator vector
            syv = torch.eye(self.sm)[sy-1,:]
            # calculate the final indicator matrix
            pm = torch.mm(target[b,:].view(-1, 1).cpu().data, syv.view(-1, 1).t())
            # and store
            newTarget[b,:] = pm

        return Variable(newTarget.view(-1,self.nclass*self.sm),requires_grad=False)

    def checkORBiasRows(self,or_bias):
        diff = or_bias[:,1:self.sm-1]>or_bias[:,0:self.sm-2]
        ind = []
        for i in range(diff.shape[0]):
            if not (diff[i,:]).all():
               ind.append(i)
        return ind

class BCORLoss(torch.nn.Module):
    def __init__(self, nclass, sm, l=1e-4, eps=1e-10):
        super(BCORLoss, self).__init__()
        self.nclass = nclass
        self.sm = sm
        self.l = l
        self.eps = eps

    def forward(self,input,target,or_bias):
        # first transform our target matrix
        newTarget = self.transformTarget(target).cuda()

        # clamp our input in order to avoid 0's or negative values and take log
        #newInput = input.clamp(min=self.eps).log()
        newInput = input.clamp(min=self.eps)

        # store copy of or_bias matrix for log. barrier penalty term
        ob_d = or_bias.clone()

        LL1 = F.binary_cross_entropy(newInput,newTarget)
        LL2 = (ob_d[:,1:self.sm-1]-or_bias[:,0:self.sm-2]).clamp(min=self.eps).log().sum()
        #print("LL1:" + str(LL1))
        #print("LL2:" + str(LL2 * -1 * self.l))
        #print("BIAS" + str(self.checkORBiasRows(or_bias)))

        # multiply with new target and take sum
        return (LL1 - (self.l * LL2))
        #return LL1
    def transformTarget(self,target):
        # create our new target tensor
        newTarget = torch.zeros(target.shape[0],self.nclass,self.sm)

        # run over batch
        for b in range(target.shape[0]):
            # calculate sy
            sy = len(target[b,:].nonzero())
            # now create sy indicator vector
            syv = torch.eye(self.sm)[sy-1,:]
            # calculate the final indicator matrix
            pm = torch.mm(target[b,:].view(-1, 1).cpu().data, syv.view(-1, 1).t())
            # and store
            newTarget[b,:] = pm

        return Variable(newTarget.view(-1,self.nclass*self.sm),requires_grad=False)

    def checkORBiasRows(self,or_bias):
        diff = or_bias[:,1:self.sm-1]>or_bias[:,0:self.sm-2]
        ind = []
        for i in range(diff.shape[0]):
            if not (diff[i,:]).all():
               ind.append(i)
        return ind

class ORNet(nn.Module):
    def __init__(self, in_features, nclass, sm):
        super(ORNet, self).__init__()
        # first store constants
        self.nclass = nclass # number of classes
        self.sm = sm # maximal hamming weight of label vector in dataset

        # initialize bias parameters for or ordinal regression model
        biasm = torch.randn(self.nclass, self.sm-1)
        self.or_bias = nn.Parameter(biasm.sort(1)[0])

        # this is the first layer in our OR network which takes in_features features
        self.linear = nn.Linear(in_features, nclass, bias=False)

    def forward(self, x):
        # calculate t(W_or,i)*Phi(x)
        x = self.linear(x)
        # now split among last dimension
        splitted = torch.split(x,1,1)
        out = []
        # run over each class
        for i,wp_i in enumerate(splitted):
            # repeat t(W_or,i)*Phi(x) sm+1 times
            splt_rep = splitted[i].repeat(1,self.sm-1)
            # subtract with biases and store in temporary tensor
            temp = F.sigmoid(self.or_bias[i,:]+splt_rep)
            # now we run over each couple and subtract consecutive values
            out_i = []
            out_i.append(temp[:,0].unsqueeze(1))
            #print(temp[:,self.sm-2].unsqueeze(1))
            for j in range(1,self.sm-1):
                # calculate difference
                diff = temp[:,j]-temp[:,j-1]
                out_i.append(diff.unsqueeze(1))
            out_i.append(temp[:,self.sm-2].unsqueeze(1))
            out.append(torch.cat(out_i,1))
        x = torch.cat(out,1)

        return x

    def checkOrderBias(self):
        for i in range(1,self.or_bias.shape[1]):
            if (self.or_bias[:,i]<=self.or_bias[:,i-1]).any():
               return self.or_bias[:,i-1:i+1]

class OR_MODEL(nn.Module):
    def __init__(self,trdl=None,vldl=None,bpreds=None,nclass=17,sm=9,gpu=False,patience=5,ds=None):
        super(OR_MODEL, self).__init__()

        # register number of classes, Sm, dataloader object, etc.
        self.nclass = nclass
        self.sm = sm
        self.trdl = trdl
        self.vldl = vldl
        self.bpreds = bpreds
        self.gpu = gpu
        self.patience = patience
        self.dataset = ds

        self.features = nn.Sequential(
            *list(models.vgg16_bn(pretrained=True).features.children())
        #    nn.MaxPool2d(kernel_size=(7,7))
        )
        self.midpart = nn.Sequential(
            nn.Dropout(p=0.25, inplace=True),
            nn.Linear(in_features=25088, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.25, inplace=True),
        )
        self.ornet = ORNet(128,self.nclass,sm)

    def forward(self,x):
        x = self.features(x)
        x = x.view(-1,25088)
        x = self.midpart(x)
        x = self.ornet(x)

        return x

    def checkORBiasRows(self):
        diff = self.ornet.or_bias[:,1:self.sm-1]>self.ornet.or_bias[:,0:self.sm-2]
        ind = []
        for i in range(diff.shape[0]):
            if not (diff[i,:]).all():
               ind.append(i)
        return ind

    def array_to_list(self,arr):
        ret = [None] * arr.shape[0]
        for i in range(arr.shape[0]):
            ret[i] = arr[i, :]
        return ret

    def train_model(self,ne=100,pbl=100,fwc=False,lr=0.01,ll=True,lamb=1e-4,verbose=True):
        # fix weights of convnet
        for p in self.features.parameters():
            p.requires_grad = fwc

        # print model
        if verbose:
            print(self)
        print(get_num_learnable_params(self))
        # loss function which basically consists of minimizing the negative LL or BCE
        if ll:
            criterion = LLORLoss(self.nclass, self.sm,l=lamb)
        else:
            criterion = BCORLoss(self.nclass,self.sm,l=lamb)

        if not fwc:
            optimizer = optim.Adam(self.ornet.parameters(), lr=lr)
        else:
            optimizer = optim.Adam(self.parameters(), lr=lr)

        val_loss_list = [sys.maxsize]
        for epoch in range(ne):  # loop over the dataset multiple times
            train_running_loss = 0.0
            train_part_running_loss = 0.0
            val_running_loss = 0.0
            val_f1 = 0.0

            # loop over training data
            counter_train = 0
            for i, data in enumerate(self.trdl, 0):

                # transform data to valid pytorch datatypes
                if self.gpu:
                    dtype = torch.cuda.FloatTensor
                else:
                    dtype = torch.FloatTensor

                inputs, labels = data
                inputs, labels = Variable(inputs.type(dtype)), Variable(labels.type(dtype), requires_grad=False)

                # set model to training mode
                self.train()

                # zero the parameter gradients
                optimizer.zero_grad()

                outputs = self(inputs)

                loss = criterion(outputs, labels, self.ornet.or_bias)
                loss.backward()
                optimizer.step()

                # print statistics
                train_part_running_loss += loss.data[0]
                train_running_loss += loss.data[0]
                counter_train += 1

                if i % pbl == pbl-1:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, train_part_running_loss / pbl))
                    train_part_running_loss = 0.0

            # loop over validation data
            counter_val = 0
            for i, data in enumerate(self.vldl, 0):

                # set model to evaluation mode and
                self.eval()

                # transform data to valid pytorch datatypes
                if self.gpu:
                    dtype = torch.cuda.FloatTensor
                else:
                    dtype = torch.FloatTensor

                inputs, labels = data
                inputs, labels = Variable(inputs.type(dtype)), Variable(labels.type(dtype))
                outputs = self(inputs)
                # get batchsize
                t = outputs.shape[0]

                loss = criterion(outputs, labels, self.ornet.or_bias)

                # print statistics
                val_running_loss += loss.data[0]
                val_f1 += self.calculate_f1gfm_batch(self(inputs), labels, self.bpreds[(i*32):(i*32)+t,:])
                counter_val += 1

            print('EPOCH %d: lossTr: %.3f    lossVal: %.3f  GFMF1Val: %.3f' %
                  (epoch + 1, train_running_loss/counter_train, val_running_loss/counter_val,
                   val_f1/counter_val))
            self.save_state("inter_models/"+self.dataset+"/OR_TRAIN_" + str(lr) + "_" + str(ll) + "_" + str(epoch+1))

            # check if early stopping applies
            valLoss = round(val_running_loss/counter_val,4)
            if val_loss_list[-1] <= valLoss:
                if len(val_loss_list) < self.patience-1:
                    val_loss_list.append(valLoss)
                else:
                    print("[info] early stopping applied after exceeding patience counter of " + str(self.patience))
                    break
            else:
                val_loss_list = [valLoss]

        print('FINISHED TRAINING')

    def calculate_f1gfm_batch(self,outputs,labels,bpreds):
        # transform labels and outputs
        labels_n = labels.cpu().data.numpy()
        outputs_n = outputs.cpu().data.numpy()
        p_o = outputs_n.reshape((-1, self.nclass, self.sm))
        p_o = np.concatenate((p_o, np.zeros((p_o.shape[0], self.nclass, self.nclass - self.sm))), axis=2)
        preds_br = np.repeat(bpreds.reshape(-1, self.nclass, 1), self.nclass, axis=2)
        p_or = p_o*preds_br
        gfm_class = gfm.GeneralFMaximizer(beta=1, n_labels=self.nclass)
        p_g, E_F = gfm_class.get_predictions(self.array_to_list(p_or))
        f1 = fbeta_score(labels_n, p_g, beta=1, average='samples')
        return f1

    def save_state(self,filename,verbose=False):
        if verbose:
            print("Saving state of model...")
        torch.save(self.state_dict(), filename+".pt")
        if verbose:
            print("Done!")

    def load_state(self,filename):
        print("Loading state of model...")
        self.load_state_dict(torch.load(filename))
        print("Done!")











