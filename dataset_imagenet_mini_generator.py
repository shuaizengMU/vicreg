import os
import random
import shutil
from threading import Thread
from time import *


class CopyImageNet(Thread):
  """Generate imagenet mini
  """

  def __init__(self, imagenet_data_dir, mini_labels_dict, mini_label_id):
    Thread.__init__(self)
    self.imagenet_data_dir = imagenet_data_dir
    self.mini_labels_dict = mini_labels_dict
    self.mini_label_id = mini_label_id
    self.mini_labels = self.mini_labels_dict[f'labels_{mini_label_id}']

  def run(self):
    sample_data_dir = f'./data/ILSVRC2012/imagenet-samemini-{self.mini_label_id}'
    for data_source_type in ['train', 'val']:
      for idx, label_name in enumerate(self.mini_labels):
        if idx % 20 == 0:
          print(data_source_type, idx)
          sleep(1)
        source_dir = os.path.join(self.imagenet_data_dir, data_source_type,
                                  label_name)
        target_dir = os.path.join(sample_data_dir, data_source_type, label_name)
        shutil.copytree(source_dir, target_dir)


def main():

  random.seed(0)
  imagenet_data_dir = './data/ILSVRC2012/imagenet'
  imagenet_labels_all = os.listdir(os.path.join(imagenet_data_dir, 'train'))

  # Imagenet-samemini-100/200/300
  mini_labels_dict = {}
  for num_mini_labels in [100, 200, 300]:
    if num_mini_labels == 100:
      fp = open('imagenet_meta/imagenet_mini-100.txt',
                mode='r',
                encoding='UTF-8')
      mini_labels_100 = fp.readlines()
      mini_labels_100 = [name.strip() for name in mini_labels_100]
      mini_labels = mini_labels_100

    else:
      remain_lables = [
          name for name in imagenet_labels_all if name not in mini_labels
      ]
      sample_data = random.sample(remain_lables, 100)
      mini_labels = mini_labels + sample_data

    mini_labels_dict[f'labels_{num_mini_labels}'] = mini_labels

  # copy Imagenet-samemini-100/200/300
  for num_mini_labels in [100, 200, 300]:
    copy_thread = CopyImageNet(imagenet_data_dir, mini_labels_dict,
                               num_mini_labels)
    copy_thread.start()


if __name__ == '__main__':
  main()