mkdir -p checkpoints

# VGG16
wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
tar -xvf vgg_16_2016_08_28.tar.gz
rm vgg_16_2016_08_28.tar.gz
mv vgg_16.ckpt checkpoints/

# VGG19
wget http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz
tar -xvf vgg_19_2016_08_28.tar.gz
rm vgg_19_2016_08_28.tar.gz
mv vgg_19.ckpt checkpoints/
