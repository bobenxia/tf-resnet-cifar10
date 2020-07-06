import tensorflow as tf
import os

from dataset import Cifar10DatasetBuilder
from dataset import read_data
from model import ResNetCifar10
from model_runners import ResNetCifar10Evaluator
from absl import flags
from absl import app

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

flags.DEFINE_string('data_path', '/home/micl/xia/dataset/cifar-10-batches-bin/', 'The path to the directory containing '
                    'Cifar10  binary files.')
flags.DEFINE_string('ckpt_path', '/home/micl/xia/resnet/', 'The path to the'
                    ' checkpoint file from which the model will be restored.')
flags.DEFINE_integer('num_layers', 20, 'Number of weighted layers. Valid '
                     'values: 20, 32, 44, 56, 110')
flags.DEFINE_boolean('shortcut_connection', True, 'Whether to add shortcut '
                     'connection. Defaults to True. False for Plain network.')

FLAGS = flags.FLAGS


def main(_):
  builder = Cifar10DatasetBuilder()

  labels, images = read_data(FLAGS.data_path, training=False) 
  dataset = builder.build_dataset(labels, images, batch_size=10000, training=False)

  model = ResNetCifar10(FLAGS.num_layers,
                        shortcut_connection=FLAGS.shortcut_connection)


  ckpt = tf.train.Checkpoint(model=model)

  evaluator = ResNetCifar10Evaluator(model)

  latest_ckpt = tf.train.latest_checkpoint(FLAGS.ckpt_path)
  if latest_ckpt:
    print('loading checkpoint %s ' % latest_ckpt)
    ckpt.restore(latest_ckpt).expect_partial()
    loss, acc = evaluator.evaluate(dataset, 10000)
    print('Eval loss: %s, eval accuracy: %s' % (loss, acc))

if __name__ == '__main__':
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  tf.config.experimental.set_memory_growth(device=physical_devices[0], enable=True)

  flags.mark_flag_as_required('data_path')
  flags.mark_flag_as_required('ckpt_path')

  app.run(main)

