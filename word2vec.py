import tensorflow_hub as hub

embed = hub.load("./embed/word2vec/")
# hub_layer = hub.KerasLayer("./embed/word2vec/",
#                            input_shape=[], dtype=tf.string)

embeddings = embed(["cat is on the mat", "dog is in the fog"])
print(embed)