import math
import tensorflow as tf
class LoraLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        original_layer,
        rank = 8,
        alpha = 32,
        trainable = True,
        **kwargs,
    ):
        # Use the name of the original layer
        original_config = original_layer.get_config()
        name = original_config["name"]

        # if keyword of name, make it return None
        kwargs.pop("name", None)

        super().__init__(name =name, trainable = trainable, **kwargs)
        self.rank = rank
        self.alpha = alpha
        self._scale = alpha / rank

        self.original_layer = original_layer
        self.original_layer.trainable = False
        self.original_layer_unit =  original_layer.get_config()["units"]

        self.A = tf.keras.layers.Dense(
            units = self.rank,
            use_bias=False,
            # Note: the original paper mentions that normal distribution was
            # used for initialization. However, the official LoRA implementation
            # uses "Kaiming/He Initialization".
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=math.sqrt(5), mode="fan_in", distribution="uniform"
            ),
            trainable=trainable,
            name=f"lora_A",
        )
        self.B = tf.keras.layers.Dense(
            units = self.original_layer_unit,
            use_bias = False,
            kernel_initializer="zeros",
            trainable=trainable,
            name=f"lora_B",
        )

    def call(self, input_):
        # While trainig, the original output is first compute then added 
        # with the lora output, which mean, while training, the latency 
        # still exist. But after the model is merge, due to the reason
        # that lora's weight is added with original layer, it wouldn't have
        # the latency.
        if self.trainable:
            original_output = self.original_layer(input_)
            lora_output = self.B(self.A(input_)) * self._scale
            return original_output + lora_output
        else:
            return self.original_layer(input_)

    def merge(self):
        # merge the weight of lora and original layer 
        new_weights = tf.einsum("ab, bc -> ac", self.A.get_weights()[0], self.B.get_weights()[0])
        original_w, original_b = self.original_layer.get_weights()
        
        self.original_layer.set_weights([original_w + new_weights, original_b])
class GPUMemoryCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        target_batches,
        print_stats=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_batches = target_batches
        self.print_stats = print_stats

        self.memory_usage = []
        self.labels = []

    def _compute_memory_usage(self):
        memory_stats = tf.config.experimental.get_memory_info("GPU:0")
        # Convert bytes to GB and store in list.
        peak_usage = round(memory_stats["peak"] / (2**30), 3)
        self.memory_usage.append(peak_usage)

    def on_epoch_begin(self, epoch, logs=None):
        self._compute_memory_usage()
        self.labels.append(f"epoch {epoch} start")

    def on_train_batch_begin(self, batch, logs=None):
        if batch in self.target_batches:
            self._compute_memory_usage()
            self.labels.append(f"batch {batch}")

    def on_epoch_end(self, epoch, logs=None):
        self._compute_memory_usage()
        self.labels.append(f"epoch {epoch} end")
