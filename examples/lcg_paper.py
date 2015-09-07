#-*- coding:utf-8 -*-
from __future__ import absolute_import
import keras,os
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop, Adadelta, Adagrad
from keras.utils import np_utils
import numpy as np
from sklearn.metrics import f1_score
import copy,sys
import cPickle
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.callbacks import ModelCheckpoint
import h5py

def load_data(trainset='./lcg/DL_train.csv',validset='./lcg/DL_valid.csv',testset='./lcg/DL_test.csv'):
    # train valid test all need to be with labels

    data=np.loadtxt(validset, delimiter=',', dtype=float, skiprows=1)
    valid_set_x, valid_set_y =(data[:,:-1],data[:,-1])

    data=np.loadtxt(testset, delimiter=',', dtype=float, skiprows=1)
    test_set_x, test_set_y =(data[:,:-1],data[:,-1])

    data=np.loadtxt(trainset, delimiter=',', dtype=float, skiprows=1)
    train_set_x, train_set_y=(data[:,:-1],data[:,-1])

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]

    return rval

def lcg_DL(nb_epoch,hiddenLayers,opMethod,directry):

    batch_size = 16
    nb_classes = 2
    sys.setrecursionlimit(100000)

    np.random.seed(1337) # for reproducibility

    # the data, shuffled and split between tran and test sets
    # keras不可以用Pandas的dataframe
    (X_train, Y_train),(X_valid, Y_valid) ,(X_test, Y_test) = load_data()

    # after readin data, change directory to save the result
    try:
        os.chdir(directry)
    except:
        os.makedirs(directry)
        os.chdir(directry)

    # redirect stdout and stderr
    sys.stdout = open('LCG_keras.txt', 'wb')
    sys.stderr = sys.stdout

    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    X_valid = X_valid.astype("float32")
    print >> sys.stdout, (X_train.shape[0], 'train samples')
    print >> sys.stdout, (X_valid.shape[0], 'valid samples')
    print >> sys.stdout, (X_test.shape[0], 'test samples')

    # 不做这步就不能categorical只能是binary
    # convert class vectors to binary class matrices
    Ytrain = np_utils.to_categorical(Y_train, nb_classes)
    Ytest = np_utils.to_categorical(Y_test, nb_classes)
    Yvalid = np_utils.to_categorical(Y_valid, nb_classes)


    # build the model
    model = Sequential()
    model.add(Dense(X_valid.shape[1], hiddenLayers[0],init='glorot_uniform')) # matrix + bias
    model.add(PReLU((hiddenLayers[0],)))
    model.add(BatchNormalization((hiddenLayers[0],)))
    model.add(Dropout(0.5))

    for i in range(1,len(hiddenLayers)):

        model.add(Dense(hiddenLayers[i-1], hiddenLayers[i],init='glorot_uniform'))
        model.add(PReLU((hiddenLayers[i],)))
        model.add(BatchNormalization((hiddenLayers[i],)))
        model.add(Dropout(0.5))

    model.add(Dense(hiddenLayers[-1], nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=opMethod) #rmsprop

    class LossHistory(keras.callbacks.Callback):
        """

        callback 可以在每次train, epoch, batch 完都塞入自己定义的函数,比方说保存weights
        load_weights, save_weights, set_weights, get_weights.
        """
        best_acc = 0

        def on_train_begin(self, logs={}):
            self.losses = []
            self.accs = []

        def on_epoch_end(self, epoch, logs={}):
            self.losses.append(logs.get('val_loss'))
            self.accs.append(logs.get('val_acc'))

            if LossHistory.best_acc < self.accs[-1]:
                model.save_weights("weights.hdf5",overwrite=True)
                # 需要用deepcopy，不然更新无效
                LossHistory.best_acc = self.accs[-1]
                print >>sys.stdout, "***********************************",LossHistory.best_acc,'\n'

    # callbacks for saving loss, accuracy history
    # history.losses可以返回多个指标
    history = LossHistory()

    model.fit(X_train, Ytrain, batch_size=batch_size, nb_epoch=nb_epoch, verbose=2, validation_data=(X_valid, Yvalid), callbacks=[history], show_accuracy=True)

    model.load_weights("weights.hdf5")

    # generate output
    print >> sys.stdout, "\nGenerating output..."

    # _predict用来输出每一层的结果，predict_proba可以用来输出概率, _predict用来输出概率的时候要注意输入文件不可以太大，否则就要分batch
    # output the probability and class
    valid_prob, valid_label = model.predict_proba(X_valid,batch_size=batch_size,verbose=0), model.predict_classes(X_valid,batch_size=batch_size,verbose=0)
    valid_output = np.hstack((valid_prob, valid_label.reshape(valid_label.shape[0], 1)))
    test_prob, test_label = model.predict_proba(X_test,batch_size=batch_size,verbose=0), model.predict_classes(X_test,batch_size=batch_size,verbose=0)
    test_output = np.hstack((test_prob, test_label.reshape(test_label.shape[0], 1)))

    fmt = ",".join(["%.6f"]*2 + ["%d"])
    np.savetxt("valid_ann_result.csv", valid_output, fmt=fmt, header="Prob0, Prob1, Label", comments='')
    np.savetxt("test_ann_result.csv", test_output, fmt=fmt, header="Prob0, Prob1, Label", comments='')

    # loss and acc log on valid data
    valid_log =  np.hstack((np.asarray(history.losses).reshape(len(history.losses),1),np.asarray(history.accs).reshape(len(history.accs),1)))
    np.savetxt("logOnValid.csv", valid_log, fmt = ",".join(["%.6f"]*2), header="Loss, Accuracy", comments='')

    # output the statistics
    print >>sys.stdout, "\nloss on Valid: %f, on Test:%f, \naccuracy on Valid: %f, on Test:%f, \nF1 socre on Valid:%f, on Test:%f" \
          % (model.evaluate(X_valid,Yvalid,batch_size=batch_size,show_accuracy=True,verbose=2)[0],
             model.evaluate(X_test,Ytest,batch_size=batch_size,show_accuracy=True,verbose=2)[0],
             model.evaluate(X_valid,Yvalid,batch_size=batch_size,show_accuracy=True,verbose=2)[1],
             model.evaluate(X_test,Ytest,batch_size=batch_size,show_accuracy=True,verbose=2)[1],
             f1_score(Y_valid,valid_label), f1_score(Y_test,test_label))

    #cPickle.dump(model,open("./LCGPAPER_nopretrain.pkl","wb"))

if __name__=="__main__":

    lcg_DL(nb_epoch=100,
           hiddenLayers=(128,1024,1024,128),
           opMethod = SGD(),
           directry="NormalSGD")