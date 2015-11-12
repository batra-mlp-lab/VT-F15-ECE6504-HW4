# uses the vqa helper tools from
# https://github.com/VT-vision-lab/VQA/blob/master/PythonHelperTools/vqaTools/
# to create VQA text file in the format
# image_id - question - best_answer
# images correspond to ms_coco_train2014a

import sys
import unicodedata as uni
import string
import numpy as np
# CAFFE
caffe_root = '/home/ashwin/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
# path to folder containing vqaTools.py
vqaTools_root = '/home/ashwin/vqa/VQA/PythonHelperTools/'
sys.path.insert(0, vqaTools_root)
# import VQA library
from vqaTools.vqa import VQA

# DEFS 
# path to annotations
annFile = '/srv/share/vqa/release_data/mscoco/vqa/mscoco_train2014_annotations.json' # INSERT appropriate path
# path to questions
quesFile = '/srv/share/vqa/release_data/mscoco/vqa/OpenEnded_mscoco_train2014_questions.json' # insert appropriate path
dataSubType = 'train2014'
qtype = ['what color','what is on the','what sport is']
# path to images 
data_dir = '/srv/share/data/mscoco/coco/images/train2014/'
model = '/home/ashwin/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
prototxt = '/home/ashwin/caffe/models/bvlc_reference_caffenet/deploy.prototxt'
# load QAs
vqa = VQA(annFile, quesFile) 
# add question type

annIds = []
anns = []
ids = []
for qitem in qtype:
  annIds = vqa.getQuesIds(quesTypes= qtype)
  anns.extend(vqa.loadQA(annIds))
  ids.extend(vqa.getImgIds(quesTypes = qtype))

UIDs = list(np.unique(np.array(ids)))

# extract fc7 features
caffe.set_mode_gpu()
caffe.set_device(1)
net = caffe.Net(prototxt,model,caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
net.blobs['data'].reshape(1,3,227,227)

fc7_feat = []
for idx,imgId in enumerate(UIDs):
  img_file_name = data_dir + 'COCO_' + dataSubType + '_'+ str(imgId).zfill(12) + '.jpg'
  net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(img_file_name))
  out_fc7 = np.array(net.forward(['fc7'])['fc7'])
  fc7_feat.append(out_fc7)
  print 'done with image:', idx
fc7_feat = np.array(fc7_feat, dtype = 'float32').squeeze(1)
np.save('fc7_3_cat', fc7_feat)

np.savetxt('imgId.csv',UIDs)

print 'number of QAs loaded: ', len(anns)
qalines = []

# this is a list of list containing the lines in the target file described above

# to strip punctiation
punk = ''.join(string.punctuation.split('?'))
punk = ''.join(punk.split("'"))
for i in range(len(anns)):
    temp = str(anns[i]['image_id']) + ' ' + vqa.qqa[anns[i]['question_id']]['question'] + ' ' + anns[i]['multiple_choice_answer'] + '\n'
    temp = uni.normalize('NFKD',temp).encode('ascii','ignore')
    temp = ''.join(ch for ch in temp if ch not in punk)
    temp = ''.join(temp.split("'s"))
    qalines.append(temp.lower())
    print 'done with QA: ', i

with open('testvqa_3_cat.txt', 'w') as f:
    f.writelines(qalines)








