# Hello

# The idea
The idea here is to develop an object detection model that can take advantage of
a visual embedding for the object labels that it can detect.
Usually, object detection is done in the classical ''one-hot way'' that naive
classification implies: the last layer has as many neurons as the number of possible labels for the classification.
What if, instead of a probability distribution over the labels, the output is a dense vector in a space reflecting the similarities in the appearance between different object labels?

We'll take a pre-trained object detection model, and tweak it by finetuning or feature extraction so that the output is a dense vector that can then be interpreted in the context of a visual embedding for object labels.

## Some ideas
- Can use any of available pre-trained obj. detectors from PyTorch
- In order to reshape the model's structure, let's add a fc layer at the end or just change the output size of the last existing fc
- For the learning process, allow a high learning rate for the last layer, but also allow some liberty to the last layers (note that there is a way of setting a different learning rate per each layer)
- A combination of feat extraction and finetuning: First you can freeze the feature extractor, and train the head; After that, you can unfreeze the feature extractor (or part of it), set the learning rate to something smaller, and continue training.

## Some questions
- Where do I get the visual embedding for object labels? It could be a visual embedding for word. Let's just start with *word2vec*, for example.
=======
# visual-embeddings
object classification experiment
