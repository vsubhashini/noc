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
To compile Caffe, please refer to the [Installation page](http://caffe.berkeleyvision.org/installation.html).

```

### Caption images using our pre-trained models. ###

Pre-trained models corresponding to the results reported in the paper can be
dowloaded here: [Drive
link](https://drive.google.com/open?id=0B90_72zRQe88cVBNd2RQaEZEZGM), [Dropbox
link](https://www.dropbox.com/sh/0ydd6mv1yy4dyi4/AABFzUzLNO0vssIvxrmAeG9fa?dl=0)

** Change directory and download the pre-trained models. **
```
    cd examples/noc
    ./download_models.sh
```

** Run the captioner. **
```
    python noc_captioner.py -i images_list.txt
```


