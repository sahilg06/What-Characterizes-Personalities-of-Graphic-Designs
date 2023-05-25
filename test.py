'''
What Characterizes Personalities of Graphic Designs? SIGGRAPH 2018
Test
'''


from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import h5py
import random
from keras.models import Model
from keras.layers import Input, Lambda
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adadelta
from keras import backend as K
import networks
import utils
from options import BaseOptions
from PIL import Image
np.random.seed(1337) #for reproducibility

opt=BaseOptions().parse()
if K.image_dim_ordering() == 'th':
    original_img_size = (opt.img_chns, opt.img_rows, opt.img_cols)
else:
    original_img_size = (opt.img_rows, opt.img_cols, opt.img_chns)


print('---------- Networks initialized ----------')
design_feature_network = networks.create_design_feature_network(opt.img_chns,opt.feature_output_dim,opt.img_rows,opt.img_cols,opt.feature_dropout_rate, opt.feature_w_regularizer,opt.batch_norm_flag)
semantic_embedding_network = networks.create_semantic_embedding_network(opt.number_of_personalities,opt.word_intermediate_dim,0, opt.word_w_regularizer)
semantic_scoring_network = networks.create_semantic_scoring_network(opt.feature_output_dim+opt.word_output_dim,opt.scoring_intermediate_dim,opt.scoring_dropout_rate, opt.scoring_w_regularizer)
merge_network = networks.create_base_network_merge(design_feature_network, semantic_embedding_network)
print('--------- Design feature network ---------')
design_feature_network.summary()
print('------- Semantic embedding network -------')
semantic_embedding_network.summary()
print('-------- Semantic scoring network --------')
semantic_scoring_network.summary()



input_a = Input(batch_shape=(None,) + original_img_size)
input_b = Input(batch_shape=(None,) + original_img_size)
input_personality = Input(shape=(opt.number_of_personalities,))
merged_a=merge_network([input_a,input_personality])
merged_b=merge_network([input_b,input_personality])
score_a = semantic_scoring_network(merged_a)
score_b = semantic_scoring_network(merged_b)


distance = Lambda(utils.distance, output_shape=utils.eucl_dist_output_shape)([score_a, score_b])
deep_ranking_model = Model(input=[input_a, input_b, input_personality], output=distance)
# adadelta=Adadelta()
deep_ranking_model.load_weights('personality_scoring_network_weights.h5')
personality_scoring_network = Model(input=[input_a, input_personality], output=score_a)

print('---------- Start testing ----------')
personality_to_index={'artistic':0,
                      'creative':1,
                      'cute':2,
                      'dynamic':3,
                      'elegant':4,
                      'energetic':5,
                      'fashion':6,
                      'fresh':7,
                      'futuristic':8,
                      'minimalist':9,
                      'modern':10,
                      'mysterious':11,
                      'playful':12,
                      'romantic':13,
                      'vintage':14}
test_img_filename='./imgs/romantic_test.jpg'
test_personality='romantic'

#load the image
img=Image.open(test_img_filename)
img.convert('RGB')
img = img.resize((opt.img_cols, opt.img_rows))
img_array = ((np.asarray(img, dtype='float32'))-128)*1.0/128
img_array = img_array.reshape((1,opt.img_rows,opt.img_cols,opt.img_chns))

if K.image_dim_ordering() == 'th':
    img_array = img_array.transpose(0, 3, 1, 2)


personality_one_hot=np.zeros(opt.number_of_personalities)
personality_one_hot[personality_to_index[test_personality]]=1
personality_one_hot=personality_one_hot.reshape(1,opt.number_of_personalities)

predicted_score=personality_scoring_network.predict([img_array,personality_one_hot])
print('Image:'+test_img_filename+', Personality:'+test_personality+', Score:'+str(predicted_score[0]))
