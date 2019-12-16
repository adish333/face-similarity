# face-similarity

## Network Architecture:
**Model**: Siamese Facenet Network with inception-resnetV1 backbone

I took pretrained facenet model which is trained on 1M celebrity images. I tried finetuning other backbones(resnet50, densenet201) on the provided data itself but it didn't perform that well.

**Loss**: Contrastive Loss

Since this was a pair similarity identification problem, I used contrastive loss instead of binary crossentropy which minimises the distance between similar pairs and maximises the distance between dissimilar pairs.   

**Metric**: Euclidian Distance

## Code Structure:
1.	face detection and training data preparation
2.	Face Embedding preparation
3.	Fine-Tuning Facenet
4.	Inference
