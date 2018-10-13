# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Infers detections on a TFRecord of TFExamples given an inference graph.

Example usage:
  ./infer_detections \
    --input_tfrecord_paths=/path/to/input/tfrecord1,/path/to/input/tfrecord2 \
    --output_tfrecord_path_prefix=/path/to/output/detections.tfrecord \
    --inference_graph=/path/to/frozen_weights_inference_graph.pb

The output is a TFRecord of TFExamples. Each TFExample from the input is first
augmented with detections from the inference graph and then copied to the
output.

The input and output nodes of the inference graph are expected to have the same
types, shapes, and semantics, as the input and output nodes of graphs produced
by export_inference_graph.py, when run with --input_type=image_tensor.

The script can also discard the image pixels in the output. This greatly
reduces the output size and can potentially accelerate reading data in
subsequent processing steps that don't require the images (e.g. computing
metrics).
"""
import cv2
import numpy as np
import itertools
import tensorflow as tf
from object_detection.inference import detection_inference

tf.flags.DEFINE_string('input_tfrecord_paths', 'None',
                       help='A comma separated list of paths to input TFRecords.')
tf.flags.DEFINE_string('output_tfrecord_path', 'None',
                       help='Path to the output TFRecord.')
tf.flags.DEFINE_string('inference_graph', 'D:\\faster_rcnn_inception_v2_coco_2018_01_28\\frozen_inference_graph.pb',
                       help='Path to the inference graph with embedded weights.')
tf.flags.DEFINE_boolean('discard_image_pixels', False,
                        help='Discards the images in the output TFExamples. This'
                        ' significantly reduces the output size and is useful'
                        ' if the subsequent tools don\'t need access to the'
                        ' images (e.g. when computing evaluation measures).')

FLAGS = tf.flags.FLAGS


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  required_flags = ['input_tfrecord_paths', 'output_tfrecord_path',
                    'inference_graph']
  for flag_name in required_flags:
    if not getattr(FLAGS, flag_name):
      raise ValueError('Flag --{} is required'.format(flag_name))

  with tf.Session() as sess:
    # input_tfrecord_paths = [
    #     v for v in FLAGS.input_tfrecord_paths.split(',') if v]
    # tf.logging.info('Reading input from %d files', len(input_tfrecord_paths))
    image = tf.read_file('D:\\aaa.png')
    image = tf.image.decode_image(image, 3)
    image.set_shape([None, None, 3])
    image_tensor = tf.expand_dims(image, 0)

    # serialized_example_tensor, image_tensor = detection_inference.build_input(
    #     input_tfrecord_paths)
    tf.logging.info('Reading graph and building model...')
    (detected_boxes_tensor, detected_scores_tensor,
     detected_labels_tensor) = detection_inference.build_inference_graph(
         image_tensor, FLAGS.inference_graph)

    tf.logging.info('Running inference and writing output to {}'.format(
        FLAGS.output_tfrecord_path))
    sess.run(tf.local_variables_initializer())
    locations = sess.run(detected_boxes_tensor)
    print(locations)
    print(sess.run(detected_scores_tensor))
    print(sess.run(detected_labels_tensor))
    # tf.train.start_queue_runners()
    # with tf.python_io.TFRecordWriter(
    #     FLAGS.output_tfrecord_path) as tf_record_writer:
    #   try:
    #     for counter in itertools.count():
    #       tf.logging.log_every_n(tf.logging.INFO, 'Processed %d images...', 10,
    #                              counter)
    #       tf_example = detection_inference.infer_detections_and_add_to_example(
    #           serialized_example_tensor, detected_boxes_tensor,
    #           detected_scores_tensor, detected_labels_tensor,
    #           FLAGS.discard_image_pixels)
    #       tf_record_writer.write(tf_example.SerializeToString())
    #   except tf.errors.OutOfRangeError:
    #     tf.logging.info('Finished processing records')
    image = cv2.imread('D:\\aaa.png')

    image = cv2.resize(image, (1680, 1050))
    for location in locations:
        left_top = (int(image.shape[1]*location[1]), int(image.shape[0]*location[0]))
        rigt_bottom = (int(image.shape[1]*location[3]), int(image.shape[0]*location[2]))
        print(left_top, rigt_bottom)
        cv2.rectangle(image,left_top,rigt_bottom,(55,255,155),2)
    cv2.imshow('aaa', image)
    cv2.waitKey(0)
  cv2.destroyAllWindows()

if __name__ == '__main__':
  tf.app.run()
