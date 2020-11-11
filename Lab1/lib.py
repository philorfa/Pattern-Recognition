# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 15:09:18 2020

@author: Δημήτρης
"""
# most of the following functions have already been implemented in the .ipynb file
# check there for comments and explenations

from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn



f = np.loadtxt("pr_lab1_2020-21_data/train.txt")
X_train = f[:, 1:]
y_train = f[:, 0]

f = np.loadtxt("pr_lab1_2020-21_data/test.txt")
X_test = f[:, 1:]
y_test = f[:, 0]


def show_sample(X, index):
    '''Takes a dataset (e.g. X_train) and imshows the digit at the corresponding index
	
    	Args:
		X (np.ndarray): Digits data (nsamples x nfeatures)
		index (int): index of digit to show
	'''
    plt.imshow(X[index].reshape(16,16), cmap='gray')
    plt.show()


def plot_digits_samples(X, y):
    '''
        Takes a dataset and selects one example from each label and plots it in subplots
    Args:
        np.ndarray): Digits data (nsamples x nfeatures)
    y (np.ndarray): Labels for dataset (nsamples)
    '''
    np.random.seed(1312)
    for i in range(9,-1,-1):
        subset = X[y == i]
        digit = subset[np.random.choice(subset.shape[0], 1), :]
        plt.subplot(5, 2, i+1)
        plt.imshow(digit.reshape(16, 16), cmap='gray')
        plt.show()


def digit_mean_at_pixel(X, y, digit, pixel=(10, 10)):
    '''
    Calculates the mean for all instances of a specific digit at a pixel location
    Args:
        np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select
		pixels (tuple of ints): The pixels we need to select. 
	Returns:
		(float): The mean value of the digits for the specified pixels
	'''
    subset = X[y == digit]
    return np.mean([subset[i, ].reshape(16, 16)[pixel[0], pixel[1]] for i in range(subset.shape[0])])


def digit_variance_at_pixel(X, y, digit, pixel=(10, 10)):
    '''Calculates the variance for all instances of a specific digit at a pixel location
    Args:
		X (np.ndarray): Digits data (nsamples x nfeatures)
		y (np.ndarray): Labels for dataset (nsamples)
		digit (int): The digit we need to select
		pixels (tuple of ints): The pixels we need to select
	Returns:
        (float): The variance value of the digits for the specified pixels
    '''
    subset = X[y == digit]
    return np.var([subset[i, ].reshape(16, 16)[pixel[0], pixel[1]] for i in range(subset.shape[0])])


def digit_mean(X, y, digit):
	'''Calculates the mean for all instances of a specific digit
	Args:
		X (np.ndarray): Digits data (nsamples x nfeatures)
		y (np.ndarray): Labels for dataset (nsamples)
		digit (int): The digit we need to select
	Returns:
		(np.ndarray): The mean value of the digits for every pixel
	'''
	return np.mean(X[y == digit,], axis = 0)


def digit_variance(X, y, digit):
    '''Calculates the variance for all instances of a specific digit
    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select
    Returns:
        (np.ndarray): The variance value of the digits for every pixel
    '''
    return np.var(X[y == 0,], axis = 0)

    
def euclidean_distance_classifier(X, X_mean):
    '''Classifiece based on the euclidean distance between samples in X and template vectors in X_mean
    Args: 
        X (np.ndarray): Digits data (nsamples x nfeatures)
        X_mean (np.ndarray): Digits data (n_classes x nfeatures)
    Returns:
        (np.ndarray) predictions (nsamples)
    '''
    return np.array([np.sqrt(np.sum(np.square(X - m), axis = 1*(X.ndim != 1) )) for m in X_mean])



class EuclideanDistanceClassifier(BaseEstimator, ClassifierMixin):  
    """Classify samples based on the distance from the mean feature value"""

    def __init__(self):
        self.X_mean_ = None


    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.
        
        Calculates self.X_mean_ based on the mean 
        feature values in X for each class.
        
        self.X_mean_ becomes a numpy.ndarray of shape 
        (n_classes, n_features)
        
        fit always returns self.
        """
        
        self.X_mean_ = np.zeros((len(np.unique(y)), X.shape[1])) #initialize self.X_mean_ with the appropriate shape
        for i in range(len(np.unique(y))): # set an appropriate range to go through all labels
            self.X_mean_[i, ]  = np.mean(X[y == i, ], axis = 0)
        
        return self

    def predict(self, X):
        """
        Make predictions for X based on the
        euclidean distance from self.X_mean_
        """
        dis = euclidean_distance_classifier(X, self.X_mean_)
        pred = (dis == np.min(dis, axis = 0)).T.dot(np.array(range(dis.shape[0])))
        return pred.astype(np.float64)
    
    def score(self, X, y):
        """
        Return accuracy score on the predictions
        for X based on ground truth y
        """
        return np.sum(self.predict(X) == y)/len(y)


def evaluate_classifier(clf, X, y, folds=5):
    """Returns the 5-fold accuracy for classifier clf on X and y
    Args:
		clf (sklearn.base.BaseEstimator): classifier
		X (np.ndarray): Digits data (nsamples x nfeatures)
		y (np.ndarray): Labels for dataset (nsamples)
	Returns:
		(float): The 5-fold classification score (accuracy)
    """
    #Using sklearn we calculate the k.fold cv - accurracy 
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    
    scores = cross_val_score(clf, X, y, 
                         cv=KFold(n_splits=folds, shuffle = True, random_state=1312), 
                         scoring="accuracy")
    return np.mean(scores)


def evaluate_euclidean_classifier(X, y, folds=5):
    """ Create a euclidean classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    clf = EuclideanDistanceClassifier()
    return evaluate_classifier(clf, X, y, folds)


def calculate_priors(X, y):
    """Return the a-priori probabilities for every class
    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
    Returns:
        (np.ndarray): (n_classes) Prior probabilities for every class
    """
    _, counts = np.unique(y, return_counts=True)
    return counts/len(y)


class CustomNBClassifier(BaseEstimator, ClassifierMixin):
    """Custom implementation Naive Bayes classifier"""

    def __init__(self, use_unit_variance=False):
        self.X_mean_ = None
        self.use_unit_variance = use_unit_variance


    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.
        Calculates self.X_mean_ based on the mean
        feature values in X for each class.
        self.X_mean_ becomes a numpy.ndarray of shape
        (n_classes, n_features)
        fit always returns self.
        """
        self.X_mean_ = np.array([digit_mean(X, y, i) for i in range(len(np.unique(y)))])
        self._priors = calculate_priors(X, y)
        self.sigma = np.ones(self.X_mean_.shape[1])
        if not self.use_unit_variance:
            self._variance = np.array([digit_variance(X, y, i) for i in range(len(np.unique(y)))])
            self._variance = np.where(self._variance== 0.0, 1e-05, self._variance) #correct 0.0 variances   
            self.sigma = np.sqrt(self._variance)
        return self    
    
    
    def predict(self, X):
        """
        Make predictions for X based on the
        euclidean distance from self.X_mean_
        """
        def posterior(self, X):
            jll = np.zeros((X.shape[0], self.X_mean_.shape[0]))
            for i in range(X.shape[0]): # all samples
                for j in range(self.X_mean_.shape[0]): #all 0-9 digits
                    logprior = np.log(self._priors[j])
                    lognormal = - 0.5 * np.sum(np.log(2. * np.pi * self.sigma[j]))
                    lognormal -= 0.5 * np.sum(((X[i] - self.X_mean_[j]) ** 2) /(self.sigma[j]))
                    jll[i, j] = logprior + lognormal
    
    
            return jll
    
        return np.argmax(posterior(self, X), axis=1).astype(np.float64)


    def score(self, X, y):
        """
        Return accuracy score on the predictions
        for X based on ground truth y
        """
        return np.sum(self.predict(X) == y)/len(y)


def evaluate_custom_nb_classifier(X, y, folds=5):
    """ Create a custom naive bayes classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    clf = CustomNBClassifier()
    return evaluate_classifier(clf, X, y, folds)


def evaluate_sklearn_nb_classifier(X, y, folds=5):
    """ Create an sklearn naive bayes classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    return evaluate_classifier(clf, X, y, folds)


def evaluate_knn_classifier(X, y, folds=5):
    """ Create a knn and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=1)
    return evaluate_classifier(clf, X, y, folds)


def evaluate_linear_svm_classifier(X, y, folds=5):
    """ Create an svm with linear kernel and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    from sklearn.svm import SVC
    clf = SVC(kernel='linear')
    return evaluate_classifier(clf, X, y, folds)


def evaluate_rbf_svm_classifier(X, y, folds=5):
    """ Create an svm with rbf kernel and evaluate it using cross-validation
    Calls evaluate_classifier
    """    
    from sklearn.svm import SVC
    clf = SVC(kernel='rbf')
    return evaluate_classifier(clf, X, y, folds)


def evaluate_bagging_classifier(X, y, folds=5):
    """ Create a bagging ensemble classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    from sklearn.ensemble import BaggingClassifier
    from sklearn.svm import SVC

    clf = BaggingClassifier(base_estimator=SVC(), n_estimators=10, random_state=1312)
    return evaluate_classifier(clf, X, y, folds)



def evaluate_voting_classifier(X, y, folds=5):
    """ Create a voting ensemble classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.ensemble import VotingClassifier

    
    clf1 = KNeighborsClassifier(n_neighbors=1)
    clf2 = GaussianNB()
    clf3 = SVC(kernel='linear', probability=True)
    eclf = VotingClassifier(estimators=[('kn', clf1), ('gnb', clf2), ('svc', clf3)], voting=np.random.choice(('hard', 'soft')))
    print(eclf.voting.capitalize() + ' vote used for the Voting Classifier')
    return evaluate_classifier(eclf, X, y, folds)      


import torch
from torch.utils.data import DataLoader
from torch import optim
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold

class PytorchNNModel(BaseEstimator, ClassifierMixin, nn.Module):  
    """Classify samples based on the distance from the mean feature value"""

    def __init__(self,epochs,input_size,hidden_size,output_size):
        super(PytorchNNModel, self).__init__()
        self.epochs=epochs
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        
        
        self.layer1 = nn.Linear(self.input_size, self.hidden_size[0])
        self.layer2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.layer3 = nn.Linear(self.hidden_size[1], self.hidden_size[2])
        self.layer4 = nn.Linear(self.hidden_size[2], self.output_size)
        self.bn1 = nn.BatchNorm1d(self.hidden_size[0])
        self.bn2 = nn.BatchNorm1d(self.hidden_size[1])
        self.bn3 = nn.BatchNorm1d(self.hidden_size[2])


        self.leaky = nn.LeakyReLU()
        self.Dropout = nn.Dropout(p=0.2)
        self.LogSoftmax = nn.LogSoftmax(dim=1)
    
        self.criterion = nn.CrossEntropyLoss()

        

    def forward(self, x):
        out = self.leaky(self.bn1(self.layer1(x)))
        out = self.Dropout(out)
        out = self.leaky(self.bn2(self.layer2(out)))
        out = self.Dropout(out)
        out = self.leaky(self.bn3(self.layer3(out)))
        out = self.LogSoftmax(self.layer4(out))
        return out
    
    def fit(self, X,y,verbose=0):
            Χ = (X-127.5)/127.5 # προεραιτική τροποποίηση
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            train_data = []
            validation_data = []
            for i in range(len(X_train)):
                train_data.append([X_train[i], y_train[i]])
            for i in range(len(X_val)):
                validation_data.append([X_val[i], y_val[i]])
            train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
            val_loader = DataLoader(validation_data, batch_size=64, shuffle=False)
            
            self.optimizer = optim.SGD(self.parameters(), lr=0.01)
            self.train()
            
            self.epoch_loss = []
            self.epoch_acc = []
            self.epoch_valacc = []
            for epoch in range(self.epochs):
                running_average_loss = 0
                acc = 0
                samples = 0
                acc_val = 0
                samples_val = 0
                for i, data in enumerate(train_loader): # loop thorugh batches
                    X_batch, y_batch = data # get the features and labels
                    self.optimizer.zero_grad() # ALWAYS USE THIS!! 
                    out = self(X_batch.float())# forward pass
                    val, y_pred = out.max(1)
                    loss = self.criterion(out, y_batch.type(torch.LongTensor)) # compute per batch loss 
                    loss.backward() # compurte gradients based on the loss function
                    self.optimizer.step() # update weights 
                    running_average_loss += loss.detach().item()
                    acc += (y_batch == y_pred).sum().detach().item()
                    samples+=len(X_batch)
                for j,val in enumerate(val_loader):
                    X_batch, y_batch = data
                    out = self(X_batch.float())
                    val, y_pred = out.max(1)
                    acc_val += (y_batch == y_pred).sum().detach().item()
                    samples_val+=len(X_batch)
                self.epoch_valacc.append(float(acc_val/samples_val))
                self.epoch_loss.append(float(running_average_loss) / (i + 1))
                self.epoch_acc.append(float(acc/samples))
                if (verbose==1):
                    print("Epoch: {}  \t Loss {} \t Accuracy {} \t Val_Accuracy {} ".format(epoch+1, float(running_average_loss) / (i + 1), float(acc/samples),float(acc_val/samples_val)))
                
    def predict(self,X):
            if (torch.is_tensor(X)==False): X=torch.from_numpy(X)
            out = self(X.float()) # get net's predictions
            val, y_pred = out.max(1) # argmax since output is a prob distribution
            return y_pred.detach().numpy()
    
    def score(self,X,y):
        Χ = (X-127.5)/127.5
        score_data = []
        for i in range(len(X)):
            score_data.append([X[i], y[i]])
        score_loader = DataLoader(score_data, batch_size=64, shuffle=False)
        self.eval()
        acc = 0
        n_samples = 0
        with torch.no_grad(): # no gradients required!! eval mode, speeds up computation
            for i, data in enumerate(score_loader):
                X_batch, y_batch = data # test data and labels
                acc += (y_batch == torch.from_numpy(self.predict(X_batch))).sum().detach().item() # get accuracy
                n_samples += len(X_batch)
        return (acc / n_samples)

    
    
def evaluate_nn_classifier(X, y, folds=5):
    try:
            i=0
            kf = KFold(n_splits=folds)  
            acc=[]
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                model=PytorchNNModel(12,256,[128,64,32],10)
                print("\n")
                print(f"--------------------- FOLD {i}/{folds-1} ---------------------\n ")
                model.fit(X_train,y_train)
                acc.append(model.score(X_test,y_test))
                print(acc[i])
                i+=1
            acc=np.array(acc)
            return np.mean(acc)
    except:   
            raise NotImplementedError




