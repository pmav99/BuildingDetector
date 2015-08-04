import numpy as np
import os, sys
from scipy import misc 

CAFFE = 'scail/data/group/atlas/packages/caffe'
#CAFFE = 'caffe'
# hack sys.path so we can import caffe
caffe_python_path = os.path.join(CAFFE, 'python')
sys.path.insert(0, caffe_python_path)
import caffe
caffe.set_device(0)

def init_net(model_file, weights_file):
    return caffe.Net(model_file, weights_file, caffe.TEST)

def get_mean_arr(mean_file):
    #get mean file 
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(mean_file, 'rb').read()
    blob.ParseFromString(data)
    arr = np.array(caffe.io.blobproto_to_array(blob))
    
    return arr[0]


def get_prob(net, mean_arr, batch):
    '''
    param net: network object returned from init_net
    param batch: list-like of images in np array form, RGB
    return : list-like of probability distribution arrays for each class
    '''
    #preprocess
    net_data_size = net.blobs['data'].data.shape[2]

    for i in xrange(len(batch)):
        batch[i] = batch[i][:,:,[2,1,0]]
        if batch[i].shape[0] != net_data_size:
            batch[i] = misc.imresize(batch[i], (net_data_size, net_data_size))
        print batch[i].shape
        batch[i] = batch[i][:,:,[2,1,0]]
        batch[i] = np.transpose(batch[i], [2, 0, 1])
        batch[i] -= mean_arr #subtract mean

    net.blobs['data'].data[...] = np.asarray(batch)
    net.forward()
    prob = net.blobs['prob'].data
    return prob
