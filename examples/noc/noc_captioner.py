DEVICE_ID = -1

from collections import OrderedDict
import argparse
import cPickle as pickle
import h5py
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
import pdb

sys.path.append('../../python/')
import caffe

from extract_vgg_features import *

# UNK_IDENTIFIER is the word used to identify unknown words
UNK_IDENTIFIER = 'unk'


def init_vocab_from_file(vocab_filename):
  print "Initializing vocabulary from %s ..." % vocab_filename
  vocabulary = {}
  vocabulary_inverted = []
  if os.path.isfile(vocab_filename):
    with open(vocab_filename, 'rb') as vocab_filedes:
      # initialize the vocabulary with the UNK word
      vocabulary = {UNK_IDENTIFIER: 0}
      vocabulary_inverted = [UNK_IDENTIFIER]
      num_words_dataset = 0
      for line in vocab_filedes.readlines():
        split_line = line.split()
        word = split_line[0]
        print word
        #if unicode(word) == UNK_IDENTIFIER:
        if word == UNK_IDENTIFIER:
          continue
        else:
          assert word not in vocabulary
        num_words_dataset += 1
        vocabulary[word] = len(vocabulary_inverted)
        vocabulary_inverted.append(word)
      num_words_vocab = len(vocabulary.keys())
      print ('Initialized vocabulary from file with %d unique words ' +
             '(from %d total words in dataset).') % \
            (num_words_vocab, num_words_dataset)
      assert len(vocabulary_inverted) == num_words_vocab
  else:
    print('Vocabulary file %s does not exist' % vocab_filename)
  return vocabulary, vocabulary_inverted

def vocab_inds_to_sentence(vocab, inds):
  sentence = ' '.join([vocab[i] for i in inds])
  # Capitalize first character.
  sentence = sentence[0].upper() + sentence[1:]
  # Replace <EOS> with '.', or append '...'.
  if sentence.endswith(' ' + vocab[0]):
    sentence = sentence[:-(len(vocab[0]) + 1)] + '.'
  else:
    sentence += '...'
  return sentence
  
def video_to_descriptor(video_id, fsg):
  video_features = []
  assert video_id in fsg.vid_poolfeats
  text_mean_fc7 = fsg.vid_poolfeats[video_id][0]
  mean_fc7 = fsg.float_line_to_stream(text_mean_fc7)
  pool_feature = np.array(mean_fc7).reshape(1, 1, len(mean_fc7))
  return pool_feature

def predict_single_word(net, mean_pool_fc7, previous_word, output='probs'):
  cont_input = 0 if previous_word==0 else 1
  cont = np.array([cont_input])
  data_en = np.array([previous_word])
  image_features = np.zeros_like(net.blobs['mean_fc7'].data)
  image_features[:] = mean_pool_fc7
  net.forward(mean_fc7=image_features, cont_sentence=cont, input_sentence=data_en)
  output_preds = net.blobs[output].data.reshape(-1)
  return output_preds

def predict_single_word_from_all_previous(net, mean_pool_fc7, previous_words):
  probs = predict_single_word(net, mean_pool_fc7, 0)
  for index, word in enumerate(previous_words):
    probs = predict_single_word(net, mean_pool_fc7, word)
  return probs

# Strategy must be either 'beam' or 'sample'.
# If 'beam', do a max likelihood beam search with beam size num_samples.
# Otherwise, sample with temperature temp.
def predict_image_caption(net, mean_pool_fc7, vocab_list, strategy={'type': 'beam'}):
  assert 'type' in strategy
  assert strategy['type'] in ('beam', 'sample')
  if strategy['type'] == 'beam':
    return predict_image_caption_beam_search(net, mean_pool_fc7, vocab_list, strategy)
  num_samples = strategy['num'] if 'num' in strategy else 1
  samples = []
  sample_probs = []
  for _ in range(num_samples):
    sample, sample_prob = sample_image_caption(net, mean_pool_fc7, strategy)
    samples.append(sample)
    sample_probs.append(sample_prob)
  return samples, sample_probs

def softmax(softmax_inputs, temp):
  exp_inputs = np.exp(temp * softmax_inputs)
  exp_inputs_sum = exp_inputs.sum()
  if math.isnan(exp_inputs_sum):
    return exp_inputs * float('nan')
  elif math.isinf(exp_inputs_sum):
    assert exp_inputs_sum > 0  # should not be -inf
    return np.zeros_like(exp_inputs)
  eps_sum = 1e-8
  return exp_inputs / max(exp_inputs_sum, eps_sum)

def random_choice_from_probs(softmax_inputs, temp=1.0, already_softmaxed=False):
  if already_softmaxed:
    probs = softmax_inputs
    assert temp == 1.0
  else:
    probs = softmax(softmax_inputs, temp)
  r = random.random()
  cum_sum = 0.
  for i, p in enumerate(probs):
    cum_sum += p
    if cum_sum >= r: return i
  return 1  # return UNK?

def sample_image_caption(net, image, strategy, net_output='predict-multimodal', max_length=20):
  sentence = []
  probs = []
  eps_prob = 1e-8
  temp = strategy['temp'] if 'temp' in strategy else 1.0
  if max_length < 0: max_length = float('inf')
  while len(sentence) < max_length and (not sentence or sentence[-1] != 0):
    previous_word = sentence[-1] if sentence else 0
    softmax_inputs = \
        predict_single_word(net, image, previous_word, output=net_output)
    word = random_choice_from_probs(softmax_inputs, temp)
    sentence.append(word)
    probs.append(softmax(softmax_inputs, 1.0)[word])
  return sentence, probs

def predict_image_caption_beam_search(net, mean_pool_fc7, vocab_list, strategy, max_length=50):
  beam_size = strategy['beam_size'] if 'beam_size' in strategy else 1
  assert beam_size >= 1
  beams = [[]]
  beams_complete = 0
  beam_probs = [[]]
  beam_log_probs = [0.]
  current_input_word = 0  # first input is EOS
  while beams_complete < len(beams):
    expansions = []
    for beam_index, beam_log_prob, beam in \
        zip(range(len(beams)), beam_log_probs, beams):
      if beam:
        previous_word = beam[-1]
        if len(beam) >= max_length or previous_word == 0:
          exp = {'prefix_beam_index': beam_index, 'extension': [],
                 'prob_extension': [], 'log_prob': beam_log_prob}
          expansions.append(exp)
          # Don't expand this beam; it was already ended with an EOS,
          # or is the max length.
          continue
      else:
        previous_word = 0  # EOS is first word
      if beam_size == 1:
        probs = predict_single_word(net, mean_pool_fc7, previous_word)
      else:
        probs = predict_single_word_from_all_previous(net, mean_pool_fc7, beam)
      assert len(probs.shape) == 1
      assert probs.shape[0] == len(vocab_list)
      expansion_inds = probs.argsort()[-beam_size:]
      for ind in expansion_inds:
        prob = probs[ind]
        extended_beam_log_prob = beam_log_prob + math.log(prob)
        exp = {'prefix_beam_index': beam_index, 'extension': [ind],
               'prob_extension': [prob], 'log_prob': extended_beam_log_prob}
        expansions.append(exp)
    # Sort expansions in decreasing order of probabilitf.
    expansions.sort(key=lambda expansion: -1 * expansion['log_prob'])
    expansions = expansions[:beam_size]
    new_beams = \
        [beams[e['prefix_beam_index']] + e['extension'] for e in expansions]
    new_beam_probs = \
        [beam_probs[e['prefix_beam_index']] + e['prob_extension'] for e in expansions]
    beam_log_probs = [e['log_prob'] for e in expansions]
    beams_complete = 0
    for beam in new_beams:
      if beam[-1] == 0 or len(beam) >= max_length: beams_complete += 1
    beams, beam_probs = new_beams, new_beam_probs
  return beams, beam_probs

def run_pred_iter(net, mean_pool_fc7, vocab_list, strategies=[{'type': 'beam'}]):
  outputs = []
  for strategy in strategies:
    captions, probs = predict_image_caption(net, mean_pool_fc7, vocab_list, strategy=strategy)
    for caption, prob in zip(captions, probs):
      output = {}
      output['caption'] = caption
      output['prob'] = prob
      output['gt'] = False
      output['source'] = strategy
      outputs.append(output)
  return outputs

def score_caption(net, image, caption, is_gt=True, caption_source='gt'):
  output = {}
  output['caption'] = caption
  output['gt'] = is_gt
  output['source'] = caption_source
  output['prob'] = []
  probs = predict_single_word(net, image, 0)
  for word in caption:
    output['prob'].append(probs[word])
    probs = predict_single_word(net, image, word)
  return output

def next_video_gt_pair(tsg):
  # modify to return a list of frames and a stream for the hdf5 outputs
  streams = tsg.get_streams()
  video_id = tsg.lines[tsg.line_index-1][0]
  gt = streams['target_sentence']
  return video_id, gt

# keep all frames for the video (including padding frame)
def all_video_gt_pairs(fsg):
  data = OrderedDict()
  if len(fsg.lines) > 0:
    prev_video_id = None
    while True:
      video_id, gt = next_video_gt_pair(fsg)
      if video_id in data:
        if video_id != prev_video_id:
          break
        data[video_id].append(gt)
      else:
        data[video_id] = [gt]
      prev_video_id = video_id
    print 'Found %d videos with %d captions' % (len(data.keys()), len(data.values()))
  else:
    data = OrderedDict(((key, []) for key in fsg.vid_poolfeats.keys()))
  return data

def gen_stats(prob, normalizer=None):
  stats = {}
  stats['length'] = len(prob)
  stats['log_p'] = 0.0
  eps = 1e-12
  for p in prob:
    assert 0.0 <= p <= 1.0
    stats['log_p'] += math.log(max(eps, p))
  stats['log_p_word'] = stats['log_p'] / stats['length']
  try:
    stats['perplex'] = math.exp(-stats['log_p'])
  except OverflowError:
    stats['perplex'] = float('inf')
  try:
    stats['perplex_word'] = math.exp(-stats['log_p_word'])
  except OverflowError:
    stats['perplex_word'] = float('inf')
  if normalizer is not None:
    norm_stats = gen_stats(normalizer)
    stats['normed_perplex'] = \
        stats['perplex'] / norm_stats['perplex']
    stats['normed_perplex_word'] = \
        stats['perplex_word'] / norm_stats['perplex_word']
  return stats

def run_pred_iters(pred_net, image_list, feature_extractor,
                   strategies=[{'type': 'beam'}], display_vocab=None):
  outputs = OrderedDict()
  num_pairs = 0
  descriptor_video_id = ''
  print "CNN ..."
  features = feature_extractor.compute_features(image_list)
  for index, video_id in enumerate(image_list):
    assert video_id not in outputs
    num_pairs += 1
    if descriptor_video_id != video_id:
      image_fc7 = features[index]
      desciptor_video_id = video_id
    outputs[video_id] = \
        run_pred_iter(pred_net, image_fc7, display_vocab, strategies=strategies)
    if display_vocab is not None:
      for output in outputs[video_id]:
        caption, prob, gt, source = \
            output['caption'], output['prob'], output['gt'], output['source']
        caption_string = vocab_inds_to_sentence(display_vocab, caption)
        if gt:
          tag = 'Actual'
        else:
          tag = 'Generated'
        if not 'stats' in output:
          stats = gen_stats(prob)
          output['stats'] = stats
        stats = output['stats']
        print '%s caption (length %d, log_p = %f, log_p_word = %f):\n%s' % \
            (tag, stats['length'], stats['log_p'], stats['log_p_word'], caption_string)
  return outputs

def to_html_row(columns, header=False):
  out= '<tr>'
  for column in columns:
    if header: out += '<th>'
    else: out += '<td>'
    try:
      if int(column) < 1e8 and int(column) == float(column):
        out += '%d' % column
      else:
        out += '%0.04f' % column
    except:
      out += '%s' % column
    if header: out += '</th>'
    else: out += '</td>'
  out += '</tr>'
  return out

def to_html_output(outputs, vocab):
  out = ''
  for video_id, captions in outputs.iteritems():
    for c in captions:
      if not 'stats' in c:
        c['stats'] = gen_stats(c['prob'])
    # Sort captions by log probability.
    if 'normed_perplex' in captions[0]['stats']:
      captions.sort(key=lambda c: c['stats']['normed_perplex'])
    else:
      captions.sort(key=lambda c: -c['stats']['log_p_word'])
    out += '<img src="%s"><br>\n' % video_id
    out += '<table border="1">\n'
    column_names = ('Source', '#Words', 'Perplexity/Word', 'Caption')
    out += '%s\n' % to_html_row(column_names, header=True)
    for c in captions:
      caption, gt, source, stats = \
          c['caption'], c['gt'], c['source'], c['stats']
      caption_string = vocab_inds_to_sentence(vocab, caption)
      if gt:
        source = 'ground truth'
        if 'correct' in c:
          caption_string = '<font color="%s">%s</font>' % \
              ('green' if c['correct'] else 'red', caption_string)
        else:
          caption_string = '<em>%s</em>' % caption_string
      else:
        if source['type'] == 'beam':
          source = 'beam (size %d)' % source['beam_size']
        elif source['type'] == 'sample':
          source = 'sample (temp %f)' % source['temp']
        else:
          raise Exception('Unknown type: %s' % source['type'])
        caption_string = '<strong>%s</strong>' % caption_string
      columns = (source, stats['length'] - 1,
                 stats['perplex_word'], caption_string)
      out += '%s\n' % to_html_row(columns)
    out += '</table>\n'
    out += '<br>\n\n' 
    out += '<br>' * 2
  out.replace('<unk>', 'UNK')  # sanitize...
  return out

def to_text_output(outputs, vocab):
  out_types = {}
  caps = outputs[outputs.keys()[0]]
  for c in caps:
    caption, gt, source = \
        c['caption'], c['gt'], c['source']
    if source['type'] == 'beam':
      source_meta = 'beam_size_%d' % source['beam_size']
    elif source['type'] == 'sample':
      source_meta = 'sample_temp_%.2f' % source['temp']
    else:
      raise Exception('Unknown type: %s' % source['type'])
    if source_meta not in out_types:
      out_types[source_meta] = []
  num_videos = 0
  out = ''
  for video_id, captions in outputs.iteritems():
    num_videos += 1
    for c in captions:
      if not 'stats' in c:
        c['stats'] = gen_stats(c['prob'])
    # Sort captions by log probability.
    if 'normed_perplex' in captions[0]['stats']:
      captions.sort(key=lambda c: c['stats']['normed_perplex'])
    else:
      captions.sort(key=lambda c: -c['stats']['log_p_word'])
    for c in captions:
      caption, gt, source, stats = \
          c['caption'], c['gt'], c['source'], c['stats']
      caption_string = vocab_inds_to_sentence(vocab, caption)
      if source['type'] == 'beam':
        source_meta = 'beam_size_%d' % source['beam_size']
      elif source['type'] == 'sample':
        source_meta = 'sample_temp_%.2f' % source['temp']
      else:
        raise Exception('Unknown type: %s' % source['type'])
      # out = '%s\t%s\t%s\n' % (source_meta, video_id, caption_string)
      out = '%s\t%s\tlog_p=%f, log_p_word=%f\t%s\n' % (source_meta, video_id,
        c['stats']['log_p'], c['stats']['log_p_word'], caption_string)
      # if len(out_types[source_meta]) < num_videos:
      out_types[source_meta].append(out)
  return out_types

def compute_descriptors(net, image_list, output_name='fc7'):
  batch = np.zeros_like(net.blobs['data'].data)
  batch_shape = batch.shape
  batch_size = batch_shape[0]
  descriptors_shape = (len(image_list), ) + net.blobs[output_name].data.shape[1:]
  descriptors = np.zeros(descriptors_shape)
  for batch_start_index in range(0, len(image_list), batch_size):
    batch_list = image_list[batch_start_index:(batch_start_index + batch_size)]
    for batch_index, image_path in enumerate(batch_list):
      batch[batch_index:(batch_index + 1)] = preprocess_image(net, image_path)
    print 'Computing descriptors for images %d-%d of %d' % \
        (batch_start_index, batch_start_index + batch_size - 1, len(image_list))
    net.forward(data=batch)
    print 'Done'
    descriptors[batch_start_index:(batch_start_index + batch_size)] = \
        net.blobs[output_name].data
  return descriptors

def sample_captions(net, image_features,
    prob_output_name='probs', output_name='samples', caption_source='sample'):
  cont_input = np.zeros_like(net.blobs['cont_sentence'].data)
  word_input = np.zeros_like(net.blobs['input_sentence'].data)
  batch_size = image_features.shape[0]
  outputs = []
  output_captions = [[] for b in range(batch_size)]
  output_probs = [[] for b in range(batch_size)]
  caption_index = 0
  num_done = 0
  while num_done < batch_size:
    if caption_index == 0:
      cont_input[:] = 0
    elif caption_index == 1:
      cont_input[:] = 1
    if caption_index == 0:
      word_input[:] = 0
    else:
      for index in range(batch_size):
        word_input[index] = \
            output_captions[index][caption_index - 1] if \
            caption_index <= len(output_captions[index]) else 0
    net.forward(image_features=image_features,
        cont_sentence=cont_input, input_sentence=word_input)
    net_output_samples = net.blobs[output_name].data
    net_output_probs = net.blobs[prob_output_name].data
    for index in range(batch_size):
      # If the caption is empty, or non-empty but the last word isn't EOS,
      # predict another word.
      if not output_captions[index] or output_captions[index][-1] != 0:
        next_word_sample = net_output_samples[index]
        assert next_word_sample == int(next_word_sample)
        next_word_sample = int(next_word_sample)
        output_captions[index].append(next_word_sample)
        output_probs[index].append(net_output_probs[index, next_word_sample])
        if next_word_sample == 0: num_done += 1
    print '%d/%d done after word %d' % (num_done, batch_size, caption_index)
    caption_index += 1
  for prob, caption in zip(output_probs, output_captions):
    output = {}
    output['caption'] = caption
    output['prob'] = prob
    output['gt'] = False
    output['source'] = caption_source
    outputs.append(output)
  return outputs

def load_weights_from_h5(net_object, h5_weights_file):
  h5_weights = h5py.File(h5_weights_file)
  for layer in net_object.params.keys():
    assert layer in h5_weights['data'].keys()
    num_axes = np.shape(net_object.params[layer])[0]
    wgt_axes = h5_weights['data'][layer].keys()
    assert num_axes == len(wgt_axes)
    for axis in range(num_axes):
      net_object.params[layer][axis].data[:] = h5_weights['data'][layer][wgt_axes[axis]]

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-m", "--modelname", type=str, required=True,
                      help='Path to NOC model (Imagenet/CoCo).')
  parser.add_argument("-v", "--vggmodel", type=str, required=True,
                      help='Path to vgg 16 model file.')
  parser.add_argument("-i", "--imagelist", type=str, required=True,
                      help='File with a list of images (full path to images).')
  parser.add_argument("-o", "--htmlout", action='store_true', help='output images and captions as html')
  args = parser.parse_args()

  DIR = '/scratch/cluster/vsub/NOC_models'
  VOCAB_FILE = './surf_intersect_glove.txt'
  LSTM_NET_FILE = './deploy.3loss_coco_fc7_voc72klabel.shared_glove.prototxt'
  VGG_NET_FILE = 'vgg_orig_16layer.deploy.prototxt'
  RESULTS_DIR = './results'
  MODEL_FILE = args.modelname
  NET_TAG = os.path.basename(args.modelname)

  if DEVICE_ID >= 0:
    caffe.set_mode_gpu()
    caffe.set_device(DEVICE_ID)
  else:
    caffe.set_mode_cpu()
  print "Setting up CNN..."
  feature_extractor = FeatureExtractor(args.vggmodel, VGG_NET_FILE, DEVICE_ID)
  print "Setting up LSTM NET"
  lstm_net = caffe.Net(LSTM_NET_FILE, MODEL_FILE, caffe.TEST)
  if MODEL_FILE.endswith('.h5'):
    load_weights_from_h5(lstm_net, MODEL_FILE)
  print "Done"
  nets = [lstm_net]



  STRATEGIES = [
    {'type': 'beam', 'beam_size': 1},
#    {'type': 'sample', 'temp': 2, 'num': 25},  # CoCo held-out reported in paper.
  ]
  NUM_OUT_PER_CHUNK = 30
  START_CHUNK = 0

  vocabulary, vocabulary_inverted = init_vocab_from_file(VOCAB_FILE)
  image_list = []
  assert os.path.exists(args.imagelist)
  with open(args.imagelist, 'r') as infd:
    image_list = infd.read().splitlines()

  print 'Captioning %d images...' % len(image_list)
  NUM_CHUNKS = (len(image_list)/NUM_OUT_PER_CHUNK) + 1 # num videos in batches of 30
  eos_string = '<EOS>'
  # add english inverted vocab 
  vocab_list = [eos_string] + vocabulary_inverted
  offset = 0
  data_split_name = 'output'
  for c in range(START_CHUNK, NUM_CHUNKS):
    chunk_start = c * NUM_OUT_PER_CHUNK
    chunk_end = (c + 1) * NUM_OUT_PER_CHUNK
    chunk = image_list[chunk_start:chunk_end]
    html_out_filename = '%s/%s.%s.%d_to_%d.html' % \
        (RESULTS_DIR, data_split_name, NET_TAG, chunk_start, chunk_end)
    text_out_filename = '%s/%s.%s_' % \
        (RESULTS_DIR, data_split_name, NET_TAG)
    if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)
    if os.path.exists(text_out_filename):
      print '(%d-%d) Appending to file: %s' % (chunk_start, chunk_end, text_out_filename)
    else:
      print 'Text output will be written to:', text_out_filename
    outputs = run_pred_iters(lstm_net, chunk, feature_extractor,
                             strategies=STRATEGIES, display_vocab=vocab_list)
    if args.htmlout:
      html_out = to_html_output(outputs, vocab_list)
      html_out_file = open(html_out_filename, 'w')
      html_out_file.write(html_out)
      html_out_file.close()
    text_out_types = to_text_output(outputs, vocab_list)
    for strat_type in text_out_types:
      text_out_fname = text_out_filename + strat_type + '.txt'
      text_out_file = open(text_out_fname, 'a')
      text_out_file.write(''.join(text_out_types[strat_type]))
      text_out_file.close()
    offset += NUM_OUT_PER_CHUNK
    print 'Wrote HTML output to:', html_out_filename

if __name__ == "__main__":
 main()

