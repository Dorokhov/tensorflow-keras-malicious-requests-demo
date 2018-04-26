from tensorflow.python import pywrap_tensorflow
import os

model_dir = 'C:\Presentations\securityai-lstm\securitai-lstm-model\logs'
checkpoint_path = os.path.join(model_dir, "keras_embedding.ckpt-0.data-00000-of-00001")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
print (len(var_to_shape_map))
for key in var_to_shape_map:
    print("tensor_name: ", key)
    print(reader.get_tensor(key)) # Remove this is you want to print only variable names
