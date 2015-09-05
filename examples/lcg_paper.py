#-*- coding:utf-8 -*-
from __future__ import absolute_import
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.utils import np_utils
import numpy as np
from sklearn.metrics import f1_score
import copy,sys
import cPickle

def load_data(trainset='DP_train_balance.csv',validset='DL_valid.csv',testset='DL_test_withlabel.csv'):


    data=np.loadtxt(validset, delimiter=',', dtype=float, skiprows=1)
    valid_set_x, valid_set_y =(data[:,:-1],data[:,-1])

    data=np.loadtxt(testset, delimiter=',', dtype=float, skiprows=1)
    test_set_x, test_set_y =(data[:,:-1],data[:,-1])

    data=np.loadtxt(trainset, delimiter=',', dtype=float, skiprows=1)
    train_set_x, train_set_y=(data[:,:-1],data[:,-1])

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]

    return rval

def lcg_DL(nb_epoch):

    batch_size = 10
    nb_classes = 2

    np.random.seed(1337) # for reproducibility

    # the data, shuffled and split between tran and test sets
    # keras不可以用Pandas的dataframe
    (X_train, Y_train),(X_valid, Y_valid) ,(X_test, Y_test) = load_data()


    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    X_valid = X_valid.astype("float32")
    print >> sys.stdout, (X_train.shape[0], 'train samples')
    print >> sys.stdout, (X_valid.shape[0], 'valid samples')
    print >> sys.stdout, (X_test.shape[0], 'test samples')

    # 不做这步就不能categorical只能是binary
    Ytrain = np_utils.to_categorical(Y_train, nb_classes)
    Ytest = np_utils.to_categorical(Y_test, nb_classes)
    Yvalid = np_utils.to_categorical(Y_valid, nb_classes)
    # convert class vectors to binary class matrices


    model = Sequential()
    model.add(Dense(30, 128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, 1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, 1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, 128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, 2))
    model.add(Activation('softmax'))

    #rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=Adam())

    best_acc = 0
    for i in range(nb_epoch):

        print >>sys.stdout, ("Epoch:%d ..." % i)
        performance = model.fit(X_train, Ytrain, batch_size=batch_size, nb_epoch=1, show_accuracy=True, verbose=1,
                                validation_data=(X_valid,Yvalid))

        print("loss on training set is %.2f, accuracy is %.2f %%" % (performance.loss[0]*100, performance.accuracy[0]*100))
        # early stopping
        if performance.validation_accuracy > best_acc:
            best_acc = performance.validation_accuracy
            best_model = copy.deepcopy(model)
            print("up-till-now best accuracy on validation set is %.2f %%" % (best_acc[0]*100))


    model = copy.deepcopy(best_model)
    valid_prob, valid_label = model.predict_proba(X_valid), model.predict_classes(X_valid)
    valid_output = np.hstack((valid_prob, valid_label.reshape(valid_label.shape[0], 1)))
    test_prob, test_label = model.predict_proba(X_test), model.predict_classes(X_test)
    test_output = np.hstack((test_prob, test_label.reshape(test_label.shape[0], 1)))


    fmt = ",".join(["%.6f"]*2 + ["%d"])
    np.savetxt("valid_ann_result.csv", valid_output, fmt=fmt, header="Prob0, Prob1, Label", comments='')
    np.savetxt("test_ann_result.csv", test_output, fmt=fmt, header="Prob0, Prob1, Label", comments='')

    print "accuracy on Valid: %f, on Test:%f, F1 socre on Valid:%f, on Test:%f" \
          % (model.test(X_valid,Yvalid,accuracy=True)[1],
             model.test(X_test,Ytest,accuracy=True)[1],
            f1_score(Y_valid,valid_label), f1_score(Y_test,test_label))

    #sys.setrecursionlimit(2000) # if not increase this, the model is too complicated to save
    #cPickle.dump(best_model,open("./LCGPAPER_nopretrain.pkl","wb"))

if __name__=="__main__":
    lcg_DL(1)