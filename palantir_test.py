import tensorflow as tf

def check_tensorflow_installation():
    try:
        # Check TensorFlow version
        print("TensorFlow Version:", tf.__version__)
        
        # Perform a simple computation using TensorFlow
        a = tf.constant(5)
        b = tf.constant(3)
        c = tf.add(a, b)
        print("TensorFlow is working! The result of 5 + 3 is:", c.numpy())
        
        # Check if GPU is available
        gpu_devices = tf.config.list_physical_devices('GPU')
        if gpu_devices:
            print("GPU is available! Devices:", gpu_devices)
        else:
            print("GPU is not available.")
            
    except Exception as e:
        print("An error occurred. TensorFlow may not be installed correctly.")
        print("Error details:", str(e))

# Run the check
check_tensorflow_installation()
