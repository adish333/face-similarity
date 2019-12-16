# face-similarity

## Network Architecture:
**Model: Siamese Facenet Network with inception-resnetV1 backbone and Pre-trained MLCNN for face detection**

I took pretrained facenet model as base model which is trained on 1M celebrity images. This network takes face image as input and outputs an embedding vector of 128 dimension. Then, I created a Siamese network which takes an image pair and tries to predict the distance between them.

**Loss: Contrastive Loss**

Since this was a pair similarity identification problem, I used contrastive loss instead of binary crossentropy which minimises the distance between similar pairs and maximises the distance between dissimilar pairs.   

**Metric: Euclidian Distance**

## Solution Workflow:
1.	face detection and training data preparation
2.	Face Embedding preparation
3.	Fine-Tuning Facenet
4.	Inference
