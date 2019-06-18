from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
import tensorflow as tf



# def get_gradient_T(model):
# 	"""
# 	Returns  
# 	"""
# 	weights = model.trainable_weights
# 	gradients = model.optimizer.get_gradients(model.total_loss, weights)
# 	input_tensors = model.inputs + model.sample_weights + model.targets + [K.learning_phase()]
# 	get_gradients = K.function(inputs=input_tensors, outputs=gradients)
# 	inputs = [x, x_off, np.ones(len(x)), y, 0]
# 	grads = get_gradients(inputs)	



#Keras Callback on Batch End!


class TestCallBack(Callback):
	"""
	Testing class to get items each call back!
	"""
	def on_batch_end(self, batch, logs=None):
		"""
		A backwards compatibility alias for `on_train_batch_end`.
		"""
		print(tf.keras.backend.eval(self.model.trainable_weights[0]))

