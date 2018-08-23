from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras.optimizers import SGD, Adam
from keras.losses import squared_hinge
import os
import keras
import argparse
import keras.backend as K
import numpy as np
from models.model_factory import build_model
from utils.config_utils import Config
from utils.load_data import load_dataset
from keras.preprocessing.image import ImageDataGenerator
from keras import losses
import datetime
import logging


logging.basicConfig(filename='test_result.log',level=logging.DEBUG)


# parse arguments
#parser = argparse.ArgumentParser(description='Model training')
#parser.add_argument('-c', '--config_path', type=str,
#                default=None, help='Configuration file')
#parser.add_argument('-o' ,'--override',action='store',nargs='*',default=[])
#
#arguments = parser.parse_args()
override_dir = {}
#arguments.override=
#for s in arguments.override:
#    s_s = s.split("=")
#    k = s_s[0].strip()
#    v = "=".join(s_s[1:]).strip()
#    override_dir[k]=v
#arguments.override = override_dir
#override_dir['lr']=0.01##########
#override_dir['wbits']=4
#override_dir['abits']=4jso
#override_dir['network_type']='full-bnn'
#override_dir['finetune']=True
override_dir['network_type']='full-qnn'

#config_path
for ii in range(4,6):
    cfg = "config_CIFAR-10"
    cfg=str(cfg+"_"+str(ii))
    cf = Config(cfg, cmd_args = override_dir)
    
    
    # if necessary, only use the CPU for debugging
    #if cf.cpu:
    #    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    
    # ## Construct the network
    print('Construct the Network\n')
    
    # In[4]:
    model = build_model(cf)
    
    
    
    print('setting up the network and creating callbacks\n')
    
    early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=10, mode='min', verbose=1)
    checkpoint = ModelCheckpoint(cf.out_wght_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)
    tensorboard = TensorBoard(log_dir='./logs/' + str(cf.tensorboard_name), histogram_freq=0, write_graph=True, write_images=False)
    
    print('loading data\n')
    
    train_data, val_data, test_data = load_dataset(cf.dataset)
    train_data['y']=np.squeeze(train_data['y'])
    test_data['y']=np.squeeze(test_data['y'])
    val_data['y']=np.squeeze(val_data['y'])
    
    # learning rate schedule
    def scheduler(epoch):
        if epoch in cf.decay_at_epoch:
            index = cf.decay_at_epoch.index(epoch)
            factor = cf.factor_at_epoch[index]
            lr = K.get_value(model.optimizer.lr)
            IT = train_data['X'].shape[0]/cf.batch_size
            current_lr = lr * (1./(1.+cf.decay*epoch*IT))
            K.set_value(model.optimizer.lr,current_lr*factor)
            print('\nEpoch {} updates LR: LR = LR * {} = {}\n'.format(epoch+1,factor, K.get_value(model.optimizer.lr)))
        return K.get_value(model.optimizer.lr)
        
    lr_decay = LearningRateScheduler(scheduler)
    
    
    #sgd = SGD(lr=cf.lr, decay=cf.decay, momentum=0.9, nesterov=True)
    adam= Adam(lr=cf.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=cf.decay)
    
    print('compiling the network\n')
    
    #model.compile(loss=squared_hinge, optimizer=adam, metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy',optimizer='RMSprop',metrics=['accuracy'])
    if cf.finetune:
        print('Load previous weights\n')
        model.load_weights(cf.out_wght_path)
    else:
        print('No weights preloaded, training from scratch\n')
    
    
    '''
    #data augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        )
    datagen.fit(train_data['X'])
    #opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
    #model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
    model.fit_generator(datagen.flow(train_data['X'], train_data['X'], batch_size=cf.batch_size),verbose=1,
                        validation_data=(val_data['X'],val_data['y']),callbacks=[checkpoint, tensorboard,lr_decay])
    '''
    print('(re)training the network\n')
    
    
    model.fit(train_data['X'],train_data['y'],
                batch_size = cf.batch_size,
                epochs = cf.epochs,
                verbose = cf.progress_logging,
                callbacks = [checkpoint, tensorboard,lr_decay],
                validation_data = (val_data['X'],val_data['y']),
                shuffle=True)
    
    
    #evaluate model
    
    score,acc=model.evaluate(test_data['X'],test_data['y'])
    #log the results
    logging.debug('time is '+ str(datetime.datetime.now()))
    logging.debug('wbit {} wbit2 {} -> accuracy =  {}\n'.format(cf.wbits,cf.wbits2,acc))
    print('Test score:', score)
    print('Test accuracy:', acc)
    print('Done\n')
    


