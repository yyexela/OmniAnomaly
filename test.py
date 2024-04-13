import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    print('\n\n\n')
    print('Success!')
    print('\n\n\n')
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print('\n\n\n')
    print(e)
    print('\n\n\n')
