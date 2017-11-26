## Captioning Images with Diverse Objects ##

This is repository contains pre-trained models and code accompanying the paper
[Captioning Images with Diverse Objects](https://arxiv.org/abs/1606.07770).

### Novel Object Captioner (NOC)

![Novel Object
Captioner](http://bair.berkeley.edu/blog/assets/novel_image_captioning/image_0.png)

While object recognition models can recognize thousands of categories of objects
such as jackals and anteaters, description models cannot compose sentences to
describe these objects correctly in context. Our novel object captioner model
overcomes this problem by building visual description systems
which can describe new objects without pairs of images and sentences about these
objects.

* [Refer to this blogpost to learn how NOC
works](http://bair.berkeley.edu/blog/2017/08/08/novel-object-captioning/)

* [Video of the Oral Talk at CVPR 2017](https://youtu.be/OQNoy4pgDr4)
* [Slides](https://drive.google.com/open?id=0Bxz2Bk18GoW9TzRrMEZ0VVdKbzA)
* [Project Page with additional resources](http://vsubhashini.github.io/noc.html)


### Getting Started.

To get started you need to compile from this branch of caffe:
```
git clone https://github.com/vsubhashini/noc.git
```

**Compile Caffe**

To compile Caffe, please refer to the [Installation page](http://caffe.berkeleyvision.org/installation.html).


### Caption images using our pre-trained models.

Pre-trained models corresponding to the results reported in the paper can be
dowloaded here: [Drive
link](https://drive.google.com/open?id=0B90_72zRQe88cVBNd2RQaEZEZGM), [Dropbox
link](https://www.dropbox.com/sh/0ydd6mv1yy4dyi4/AABFzUzLNO0vssIvxrmAeG9fa?dl=0)

**Change directory and download the pre-trained models.**
```
cd examples/noc
./download_models.sh
```

**Run the captioner.**
```
python noc_captioner.py -i images_list.txt
```

Output with the default options:

```
Captioning 10 images...
Text output will be written to:
./results/output.imgnetcoco_3loss_voc72klabel_inglove_prelm75k_sgd_lr4e5_iter_80000.caffemodel.h5_
CNN ...
Computing features for images 0-9 of 10
Generated caption (length 11, log_p = -8.323791, log_p_word = -0.756708):
A man is sitting at a table with a cake.
Generated caption (length 12, log_p = -9.886197, log_p_word = -0.823850):
A group of people standing on a beach with a kite.
Generated caption (length 12, log_p = -13.384445, log_p_word = -1.115370):
A street sign on a city street with cars and cars.
Generated caption (length 12, log_p = -9.699789, log_p_word = -0.808316):
A dog laying on top of a white and black dog.
Generated caption (length 10, log_p = -5.238667, log_p_word = -0.523867):
A man riding skis down a snow covered slope.
Generated caption (length 10, log_p = -12.567964, log_p_word = -1.256796):
A truck with a large truck on the back.
Generated caption (length 12, log_p = -9.764039, log_p_word = -0.813670):
A man is holding a glass of wine in his hand.
Generated caption (length 12, log_p = -10.339204, log_p_word = -0.861600):
A man is standing in the dirt with a baseball bat.
Generated caption (length 10, log_p = -8.151620, log_p_word = -0.815162):
A woodpecker sitting on a tree in a park.
Generated caption (length 50, log_p = -41.878472, log_p_word = -0.837569):
A woman holding a giant flounder in the background ...

```

###  NOTES

**NOTE1: The model is not trained on all COCO objects and is hence not
competitive with other models trained on all MSCOCO training/val data**

**NOTE2: The model is trained on imagenet labels for some objects refer
to the following section on training the model to know more.**

### Training the model.

To train the model you need to download the MSCOCO image captioning dataset 
(the splits for training and held-out images are in `data_utils/image_list/`.
We also use the ImageNet dataset (http://image-net.org/download). For the ImageNet
experiments, some classes are outside the 1,000 classes chosen for the ILSVRC
challenge. To see which images we used, refer to image ids in `data_utils/image_list/`
which includes imagenet image filename and label used for training.

Please refer to the [Deep Compositional Captioning link here](https://github.com/LisaAnne/DCC)
for help with downloading the data.

**Model Training scripts**

* Model prototext is specified in `3loss_coco_fc7_voc72klabel.shared_glove72k.prototxt`
* Solver prototext including hyperparameters are in `solver_3loss_coco_fc7_voc72klabel.shared_glove72k.prototxt`
* Script to launch the training job is in `train_3loss_coco_fc7_voc72klabel.sh`

**Code to prepare training hdf5 data**

The network has 3 components one which takes just images with labels, the next
takes input images and corresponding captions, and the third part takes just
text as input. The code in `data_utils` is _provided as a reference_ to generate
all 3 types of data.

* `data_utils/tripleloss_labels_coco_to_hdf5_data.py` creates hdf5 data from images with labels (like imagenet, or coco images with multiple labels).
* `data_utils/text_labels_coco_to_hdf5_data.py` creates hdf5 data from images with captions.
* `data_utils/tripleloss_text_coco_to_hdf5_data.py` creates hdf5 from plain text data.

### Reference

If you find this code helpful, please consider citing:

[Captioning Images with Diverse Objects](https://vsubhashini.github.io/noc.html)

    Captioning Images with Diverse Objects
    S. Venugopalan, L. A. Hendricks, M. Rohrbach, R. Mooney, T. Darrell, K. Saenko
    The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2017

    @inproceedings{venugopalan17cvpr,
          title = {Captioning Images with Diverse Objects},
          author={Venugopalan, Subhashini and Hendricks, Lisa Anne and Rohrbach,
          Marcus and Mooney, Raymond, and Darrell, Trevor and Saenko, Kate},
          booktitle = {Proceedings of the IEEE Conference on Computer Vision and
          Pattern Recognition (CVPR)},
          year = {2017}
    }


You might also want to refer to,

* [Deep Compositional Captioning: Describing Novel Object Categories without Paired Training Data](http://arxiv.org/abs/1511.05284)
* [Code for DCC can be found here.](https://github.com/LisaAnne/DCC)


