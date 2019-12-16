# face-similarity

## Network Architecture:
**Model: Siamese Facenet Network with inception-resnetV1 backbone and Pre-trained MTCNN for face detection**

I took pretrained facenet model as base model which is trained on 1M celebrity images. This network takes face image as input and outputs an embedding vector of 128 dimension. Then, I created a Siamese network which takes an image pair and tries to predict the distance between them.

This network requires a face image so I used mtcnn library to extract face from images to feed to Siamese network.

**Loss: Contrastive Loss**

Since this was a pair similarity identification problem, I used contrastive loss instead of binary crossentropy which minimises the distance between similar pairs and maximises the distance between dissimilar pairs.   

**Metric: Euclidian Distance**

## Training:

Initially I tried to finetune the whole facenet mmodel with 22M parameters. I also tried freezing some layers but, due to lack of computational resources I wasn't able to complete the training for this network.

In my second approach, I cropped the faces from the images, extracted the face embeddings using pretrained facenet network for all the images beforehand. I then trained the similar siamese network with addition of a few layers.

## Solution Workflow:
1.	face detection and training data preparation
2.	Face Embedding preparation
3.	Fine-Tuning Facenet
4.	Inference
