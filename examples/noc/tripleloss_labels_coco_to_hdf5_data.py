#!/usr/bin/env python
# image fc7 labels to hdf

import numpy as np
import os
import random
random.seed(3)
import sys

from hdf5_npsequence_generator import SequenceGenerator, HDF5SequenceWriter

# UNK_IDENTIFIER is the word used to identify unknown words
UNK_IDENTIFIER = 'unk'

# start every sentence in a new array, pad if <max
MAX_FRAMES = 1
MAX_WORDS = 20

"""Filenames has file with image/frame paths for vidids
   and sentences with video ids"""
class fc7SequenceGenerator(SequenceGenerator):
  def __init__(self, filenames, vocab_filename, batch_num_streams=1, max_frames=MAX_FRAMES,
               align=True, shuffle=True, pad=True, truncate=True):
    self.max_frames = max_frames
    self.vid_labels = {}
    self.lines = []
    num_empty_lines = 0
    self.vid_poolfeats = {} # listofdict [{}]
    self.image_ids = {}

    for imgid_file, poolfeatfile, sentfile in filenames:
      print 'Reading features from file: %s' % poolfeatfile
      with open(imgid_file, 'r') as infd:
        image_ids = infd.read().splitlines()
      self.image_ids = dict.fromkeys(image_ids, 1)
      with open(poolfeatfile, 'rb') as poolfd:
        # each line has the fc7 mean of 1 video
        for line in poolfd:
          line = line.strip()
          id_feat = line.split(',')
          img_id = id_feat[0]
          line = ','.join(id_feat[1:])
          if img_id in self.image_ids:
            if img_id not in self.vid_poolfeats:
              self.vid_poolfeats[img_id]=[]
            self.vid_poolfeats[img_id].append(line)
      # reset max_words based on maximum frames in the video
      if os.path.exists(sentfile):
        print 'Reading labels in: %s' % sentfile
        with open(sentfile, 'r') as sentfd:
          for line in sentfd:
            line = line.strip()
            id_sent = line.split('\t')
            if len(id_sent)<2:
              num_empty_lines += 1
              continue
            #labels = id_sent[1].split(',')
            if id_sent[0] in self.image_ids:
              self.vid_labels[id_sent[0]] = id_sent[1]
              self.lines.append((id_sent[0], id_sent[1]))
      if num_empty_lines > 0:
        print 'Warning: ignoring %d empty lines.' % num_empty_lines
    self.line_index = 0
    self.num_resets = 0
    self.num_truncates = 0
    self.num_pads = 0
    self.num_outs = 0
    self.frame_list = []
    self.vocabulary = {}
    self.vocabulary_inverted = []
    # initialize vocabulary
    self.init_vocabulary(vocab_filename)
    SequenceGenerator.__init__(self)
    self.batch_num_streams = batch_num_streams  # needed in hdf5 to seq
    # make the number of image/sentence pairs a multiple of the buffer size
    # so each timestep of each batch is useful and we can align the images
    if shuffle:
      random.shuffle(self.lines)
    self.pad = pad
    self.truncate = truncate

  def streams_exhausted(self):
    return self.num_resets > 0

  def init_vocabulary(self, vocab_filename):
    print "Initializing the vocabulary."
    if os.path.isfile(vocab_filename):
      with open(vocab_filename, 'rb') as vocab_file:
        self.init_vocab_from_file(vocab_file)
    else:
      print "Error: No vocab file!"

  def init_vocab_from_file(self, vocab_filedes):
    # initialize the vocabulary with the UNK word
    self.vocabulary = {UNK_IDENTIFIER: 0}
    self.vocabulary_inverted = [UNK_IDENTIFIER]
    num_words_dataset = 0
    for line in vocab_filedes.readlines():
      split_line = line.split()
      word = split_line[0]
      print word
      #if unicode(word) == UNK_IDENTIFIER:
      if word == UNK_IDENTIFIER:
        continue
      else:
        assert word not in self.vocabulary
      num_words_dataset += 1
      self.vocabulary[word] = len(self.vocabulary_inverted)
      self.vocabulary_inverted.append(word)
    num_words_vocab = len(self.vocabulary.keys())
    print ('Initialized vocabulary from file with %d unique words ' +
           '(from %d total words in dataset).') % \
          (num_words_vocab, num_words_dataset)
    assert len(self.vocabulary_inverted) == num_words_vocab

  def next_line(self):
    num_lines = float(len(self.lines))
    if self.line_index == 1 or self.line_index == num_lines or self.line_index % 10000 == 0:
      print 'Processed %d/%d (%f%%) lines' % (self.line_index, num_lines,
                                              100 * self.line_index / num_lines)
    self.line_index += 1
    if self.line_index == num_lines:
      self.line_index = 0
      self.num_resets += 1

  def get_pad_value(self, stream_name):
    return 0

  """label_list: line with "," separated list of positive labels
     return: list of 0/1s dim:#lexical labels, 1 for +ve"""
  def labels_to_values(self, label_list):
    pos_labels = label_list.split(',')
    label_arr = np.zeros((1, len(self.vocabulary)+1))
    label_indices = [self.vocabulary[label]+1 for label in pos_labels]
    label_arr[0, label_indices] = 1
    return label_arr

  def float_line_to_stream(self, line):
    return map(float, line.split(','))

  # we have pooled fc7 features already in the file
  def get_streams(self):
    vidid, line = self.lines[self.line_index]
    assert vidid in self.vid_poolfeats
    text_mean_fc7 = self.vid_poolfeats[vidid][0] # list inside dict
    mean_fc7 = self.float_line_to_stream(text_mean_fc7)
    self.vid_labels[vidid] = self.labels_to_values(line)
    labels = self.vid_labels[vidid]

    self.num_outs += 1
    out = {}
    out['img_only_fc7'] = np.array(mean_fc7).reshape(1, len(mean_fc7))
    out['img_only_labels'] = labels
    self.next_line()
    return out


VIDEO_STREAMS = 1
BUFFER_SIZE = 32 # TEXT streams
BATCH_STREAM_LENGTH = 1000 # (21 * 50)
SETTING = 'data/coco2014'
# OUTPUT_BASIS_DIR = '{0}/hdf5/buffer_{1}_only8obj_label72k_{2}'.format(SETTING,
OUTPUT_BASIS_DIR = '{0}/hdf5/buffer_{1}_only8newobj_label72k_{2}'.format(SETTING,
VIDEO_STREAMS, MAX_FRAMES)
VOCAB = './surf_intersect_glove.txt'
OUTPUT_BASIS_DIR_PATTERN = '%s/%%s_batches' % OUTPUT_BASIS_DIR
POOLFEAT_FILE_PATTERN = 'data/coco2014/coco2014_{0}_vgg_fc7.txt'
LABEL_FILE_PATTERN = 'data/coco2014/sents/labels_glove72k_train2014.txt'
# IMAGEID_FILE_PATTERN='data/coco2014/coco_only8objs_image_list_train2014.txt'
IMAGEID_FILE_PATTERN='data/coco2014/cvpr17_rm8newobjs/coco_only8newobjs_image_list_train2014.txt'

def preprocess_dataset(split_name, data_split_name, batch_stream_length, aligned=False):
  filenames = [(IMAGEID_FILE_PATTERN,
                POOLFEAT_FILE_PATTERN.format(data_split_name),
                LABEL_FILE_PATTERN)]
  vocab_filename = VOCAB
  output_basis_path = OUTPUT_BASIS_DIR_PATTERN % split_name
  aligned = True
  fsg = fc7SequenceGenerator(filenames, vocab_filename, VIDEO_STREAMS,
         max_frames=MAX_FRAMES, align=aligned, shuffle=True, pad=aligned,
         truncate=aligned)
  fsg.batch_stream_length = batch_stream_length
  writer = HDF5SequenceWriter(fsg, output_dir=output_basis_path)
  writer.write_to_exhaustion()
  writer.write_filelists()
  if not os.path.isfile(vocab_filename):
    print "Vocabulary not found"

def process_splits():
  DATASETS = [ # split_name, data_split_name, aligned
      # ('valid', 'mytest', True),
      ('train', 'trainvallstm', False),
      # ('test', 'test', False),
  ]
  for split_name, data_split_name, aligned in DATASETS:
    preprocess_dataset(split_name, data_split_name, BATCH_STREAM_LENGTH,aligned)

if __name__ == "__main__":
  process_splits()
