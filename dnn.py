#!/usr/bin/env python3

# MNIST NUMBER CATEGORIZER
import argparse
import sys

# for deep learning
import torch
import numpy as np

# This is a dataset class, datasets are to help us
# make a batch worth of data at a time. Its job is to
# hold all of the data and when we give it an index it
# should give us an x and a y, an input and expected
# output. The important methods we need to override are
# the __len__ function which tells it how many indices
# it has, and the __getitem__, which when passed an index
# should grab that data point and return it as a tuple.
class MNISTDataset(torch.utils.data.Dataset):
    # this is the constructor function for the class.
    # it takes in the x and ys
    def __init__(self, inputs, targets):
        """
        Data loader class for preparing minibatches
        """
        self.inputs = np.load(inputs).astype(np.float32)
        self.targets = np.load(targets).astype(np.float32)

    # the underscores mean its a built in function we are overriding
    def __len__(self):
        """
        Accessor for total number of samples
        """
        return len(self.inputs)

    # this is just helpful so we dont hardcode dataset sizes
    def D(self):
        """
        Return input feature dimensions
        """
        return self.inputs.shape

    # this function gets called when you enumerate a class
    # in python. It is handed an index, and the class can
    # return an item. Ours just looks in our x and y lists,
    # grabs the matching pair, and returns them.
    def __getitem__(self, index):
        """
        Generate a single data sample
        """
        # Load data and get label
        x = self.inputs[index]
        y = self.targets[index]

        return (x, y)

# This is the actual model. This torch.nn.Module is a very
# important class since it is the framework for a model.
# In the __init__ we need to call super to inherit a bunch of class
# junk, then this needs to hold our weights and biases.
# Here, we want to make a variable number of layers depending on
# the input args. We make a ModuleList() as linear_layers (line 85)
# and insert however many layers we need into it, and then
# send the outputs through the whole sequence to make a prediction.
class DeepNeuralNet(torch.nn.Module):
    def __init__(self, args, D):
        super(DeepNeuralNet, self).__init__()
        self.activationfunction = args.f1

        # extract layer units and length information
        layerlist = getattr(args, 'L')
        comma_delimited_list = layerlist.split(',')
        nunits_list = [i.split('x', 1)[0] for i in comma_delimited_list]
        nlayers_list = [i.split('x', 1)[1] for i in comma_delimited_list]
        nunits_list.insert(0, D)
        nunits_list.append(args.C)

        # convert lists to numbers
        nunits_list = list(map(int,nunits_list))
        nlayers_list = list(map(int,nlayers_list))

        # make module list of layers
        self.linear_layers = torch.nn.ModuleList()

        # assign width of each layer (inputs outputs)
        for i in range(len(nlayers_list)):
            num_layers = nlayers_list[i]
            for j in range(num_layers):
                # this is where we actually make the linear layers and append them
                # to our list. Each layer is a matrix of weights to learn. We define
                # its shape with the arguments to torch.nn.Linear. eg. torch.nn.linear(3, 4)
                # makes a 3 row by 4 col matrix. When we define these we need to make sure
                # you can multiply through hence this bit of trickiness here to deal
                # with layers of differing size.
                if (j == 0):
                    self.linear_layers.append(torch.nn.Linear(nunits_list[i],
                                                              nunits_list[i + 1]))
                else:
                    self.linear_layers.append(torch.nn.Linear(nunits_list[i + 1],
                                                              nunits_list[i + 1]))

        # print paramater dimensions
        # this is just a santiy check
        for name, param in self.named_parameters():
            print(name,param.data.shape)

    # this is the part that makes a prediction
    # it takes an x and should return a y prediction.
    # its important to realize this doesnt have to be a
    # single data point, we can input a whole matrix of
    # data points and it will run a guess on the whole
    # batch at once. This helps get a model that generalizes
    # well, since we are being trained with average error, not
    # the error of a single point that could be an outlier.
    # Forward function
    def forward(self, x):
        # i here is just the index that enumerate returns
        for i, l in enumerate(self.linear_layers):

            # so this is just passing x through each index in
            # the list of layers, beginning to end
            x = self.linear_layers[i](x)

            # selecting activation function
            # the activation function is a way that
            # we strengthen the boundaries of our decision.
            # Each of these in some way restricts units.
            # eg. the sigmoid squished all number to between
            # 0 and 1. Think of this like probabilities,
            # high and low probabilities only ever equal 0 or 1
            if (self.activationfunction == "relu"):
                func = torch.nn.ReLU()
            elif (self.activationfunction == "sigmoid"):
                func = torch.nn.Sigmoid()
            elif (self.activationfunction == "tanh"):
                func = torch.nn.Tanh()
            x = func(x)

        return x

# this is just to get all the nice command line arguments.
# check it out, it lets you define flags, usage and defaults.
# it also automatically generates a -h/--help function
# to display this info.
def parse_all_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("C",help="The number of categories to fit (int)",\
            type=int)
    parser.add_argument("train_x",help="The training set input data (npy)")
    parser.add_argument("train_y",help="The training set target data (npy)")
    parser.add_argument("dev_x",help="The development set input data (npy)")
    parser.add_argument("dev_y",help="The development set target data (npy)")

    parser.add_argument("-f1",type=str,\
            help="The hidden activation functon: \"relu\" or \"tanh\" or \"sigmoid\" (string)", \
            default="relu")
    parser.add_argument("-opt",type=str,\
            help="The optimizer: \"adadelta\", \"adagrad\", \"adam\", \"rmsprop\", \"sgd\" (string)", \
            default="adam")
    parser.add_argument("-L",type=str,\
            help="A comma delimited list of nunits by nlayers specifiers (string) [default: \"32x1\"]", \
            default="32x1")
    parser.add_argument("-lr",type=float,\
            help="The learning rate (float) [default: 0.1]",default=0.1)
    parser.add_argument("-mb",type=int,\
            help="The minibatch size (int) [default : 32]",default=32)
    parser.add_argument("-report_freq",type=int,\
            help="Dev performance is reported every report_freq updates (int)",\
            default=128)
    parser.add_argument("-epochs",type=int,\
            help="The number of training epochs (int) [default: 100]",\
            default=100)

    return parser.parse_args()

# training loop
def train(model, train_loader, dev_loader, args):

    """
    Train current all epochs in a loop
    """
    # define our loss function and optimizer
    # these are all pretty much premade, but there are many different
    # options, or you could define your own loss function and optimizer.
    # Optimizer is the bit that calculates the gradient
    criterion = torch.nn.CrossEntropyLoss()

    if (args.opt == 'adadelta'):
        optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
    elif (args.opt == 'adagrad'):
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
    elif (args.opt == 'adam'):
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif (args.opt == 'rmsprop'):
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=0.9)
    elif (args.opt == 'sgd'):
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # epochs is how we control how long the model trains for.
    # one full training set training and a dev set test at the
    # end (usually) comprises one epoch. Basically means we made
    # it through all our data once.
    for epoch in range(args.epochs):

        # we need to make a bit more than just the loss
        # since we are also trying to calculate percentage
        # we got right. The loss function is just an arbitrary
        # number so it doesn't actually tell us much about how
        # good our model is, just whether its getting better
        # or worse.
        loss = 0
        correct = 0
        firsttime = True
        training_accuracy = 0

        # this pulls out however many mininbatches we set up
        # our train_loader to have. For each batch it gets a
        # minibatch x (mb_x) and a minibatch y (mb_y)
        for update, (mb_x, mb_y) in enumerate(train_loader):

            # EVALUATE MODEL
            # make a prediction based on this x
            mb_y_pred = model(mb_x)

            # more numpy shenanigans to print stuff, skip to 236
            detached_pred = mb_y_pred.detach().clone()
            pred_y_npy = detached_pred.detach().numpy()
            categories = np.argmax(pred_y_npy, axis=1)

            # here we see how different our guess was from the truth
            loss = criterion(mb_y_pred, mb_y.long())

            # take gradient step
            # this is where we change all the weights 
            # to try and lower the error
            optimizer.zero_grad() # reset the gradient values to 0
            loss.backward()       # compute the gradient values
            optimizer.step()      # apply gradients

            # calculate accuracy with more shenanigans,
            # at this point the actual training is done.

            detached_pred = mb_y_pred.detach().clone()
            pred_y_npy = detached_pred.detach().numpy()
            train_y_npy = mb_y.numpy()

            categories = np.argmax(pred_y_npy, axis=1)

            dif = np.subtract(categories, train_y_npy)
            correct = np.count_nonzero(dif == 0)
            training_accuracy = correct/args.mb

            # report freq is so we only print stuff sometimes,
            # printing is super slow.
            # dev on each mb and average results
            if (update % args.report_freq == 0):
                mb_accuracy = 0
                dev_correct = 0
                firstdev = True
                for dev_update, (mb_x, mb_y) in enumerate(dev_loader):


                    mb_y_pred = model(mb_x)
                    #print(mb_y_pred)
                    # calculate accuracy percentage
                    # guesses correct / total guess
                    mb_y_pred = mb_y_pred.detach().clone()
                    mb_y_pred = mb_y_pred.cpu().numpy()
                    mb_y_npy = mb_y.detach().cpu().numpy()

                    categories = np.argmax(mb_y_pred, axis=1)

                    dif = np.subtract(categories, mb_y_npy)
                    dev_correct += np.count_nonzero(dif == 0)

                dev_accuracy = dev_correct/(args.mb * dev_update)

                print("Dev Accuracy =", "{:.3f}".format(dev_accuracy),
                      "\tTraining Accuracy =","{:.3f}".format(training_accuracy),
                      "\tLoss = ""{:.3f}".format(loss))


# MAIN #======================================================================80
def main(argv):

    # parse arguments
    args = parse_all_args()


    # load data
    # we are using the MNIST data set, its a bunch of hand written
    # numbers with the corresponding number as the y value.
    # We can put these into dataloaders which are nice prebuilt 
    # classes who pull points out of the dataset for us. They are
    # very useful because they can pull out a batch worth of data
    # at a time and stick all the points together into one vector
    # for us. By setting shuffle to true the dataset will also randomize
    # its batches each iteration.
    training_set = MNISTDataset(args.train_x, args.train_y)
    dev_set = MNISTDataset(args.dev_x, args.dev_y)
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.mb, shuffle=True, drop_last=False)
    dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=args.mb, shuffle=False, drop_last=False)


    print(training_set.D())
    # the model needs the C num of classes and the input features of dev_x
    model = DeepNeuralNet(args, dev_set.D()[1])
    # train it up
    train(model, train_loader, dev_loader, args)

if __name__ == "__main__":
    main(sys.argv)
