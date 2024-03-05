import tensorflow as tf
import numpy as np
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer
from lora import *
from loader import *

if __name__ == "__main__":
    devices = tf.config.list_physical_devices('GPU')
    try:
        # Have to set this to full, cause tensorflow default use all vram
        tf.config.experimental.set_memory_growth(devices, False)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

    MODEL_NAME = "distilbert-base-uncased"
    tkzr = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    train_data, val_data = process_data("train-00000-of-00001.parquet", tkzr, train_ratio = 1.0)
    train_data = train_data.batch(32)
    val_data = val_data.batch(32)
    test_data, _  = process_data("test-00000-of-00001.parquet", tkzr, train_ratio = 1.0)
    # Use a smaller batch size for better observation between lora and lora after merge
    # Smaller batch size would lower the benefit from parrel computing.
    test_data = test_data.batch(4)

    # The batch number that we want to observe memory comsume

    batch_list = [1 * n for n in range(10)]

    # Monitor the consumption of vram
    vanila_tracker = GPUMemoryCallback(batch_list)
    lora_tracker = GPUMemoryCallback(batch_list)



    # vanila
    model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_NAME)
    model.summary()
    # Before the run, reset the momory stats
    tf.config.experimental.reset_memory_stats(
        "GPU:0"
    )
    model.compile(optimizer = "adam",loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics = ["accuracy"])
    model.fit(train_data, epochs = 5, callbacks = [vanila_tracker])
    loss, acc = model.evaluate(test_data)


    # Lora finetune
    model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_NAME)
    for block in model.distilbert.transformer.layer:
        query = block.attention.q_lin
        block.attention.q_lin = LoraLayer(query)

        value = block.attention.v_lin
        block.attention.v_lin = LoraLayer(value)
    # Tensorflow recommend calling call method to build all the weight
    id_, mask = tkzr("Hello World", padding = True , max_length = 250, truncation = True).values()
    id_ = np.array(id_)
    mask = np.array(mask)
    model(id_, mask)
    for layer in model._flatten_layers():
        list_of_sublayers = list(layer._flatten_layers())
        if len(list_of_sublayers) == 1:  # "leaves of the model"
            if layer.name in ["lora_A", "lora_B"]:
                layer.trainable = True
            else:
                layer.trainable = False
        else:
            # The reason that embeedings have to be specified here
            # is because that hugging face adopt the mixed usage of
            # adding weight to the layers and assigning other layers
            # like layernormalization and dropout inside the class.
            # Which means that it would be filtered out by previous
            # if statemnt but it does contain the weight(word
            # and position embeddings)
            if layer.name == "embeddings":
                layer.trainable = False
            
    model.summary()
    # Before the run, reset the momory stats
    tf.config.experimental.reset_memory_stats(
        "GPU:0"
    )
    model.compile(optimizer = "adam",loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics = ["accuracy"])
    model.fit(train_data, epochs = 5 , callbacks = [lora_tracker])
    loss, acc = model.evaluate(test_data)
    for block in model.distilbert.transformer.layer:
        block.attention.q_lin.merge()
        block.attention.v_lin.merge()
    loss, acc = model.evaluate(test_data)

    from matplotlib import pyplot as plt 
    plt.plot([n for n in range(len(vanila_tracker.memory_usage))], vanila_tracker.memory_usage)
    plt.plot([n for n in range(len(lora_tracker.memory_usage))], lora_tracker.memory_usage)
    plt.legend(["Vanila", "LoRA"])
    
    plt.show()


