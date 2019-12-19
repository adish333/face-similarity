import keras.backend as K
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
import cv2
from numpy import expand_dims
import argparse

def get_face(filename, required_size=(160, 160)):
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    try:
        results = detector.detect_faces(image)
        x1, y1, width, height = results[0]['box']
    except:
        print("No face detected!")
        return None
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = image[y1:y2, x1:x2]
    face = cv2.resize(face, required_size)
    return face.astype('float32')

def get_embedding(model, face):
    try:
        mean, std = face.mean(), face.std()
        face = (face - mean) / std
        samples = expand_dims(face, axis=0)
        yhat = model.predict(samples)
    except:
        return None
    return yhat[0]

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def get_distance(img1_path, img2_path):
    face1 = get_face(img1_path)
    face1_embd = get_embedding(facenet_base, face1)

    face2 = get_face(img2_path)
    face2_embd = get_embedding(facenet_base, face2)
    try:
        distance = face_detector.predict([[face1_embd], [face2_embd]])[0][0]
    except:
        return -1
    return distance

def show_results(distance):
    print("The similarity between the images > ", max(0, 1-distance))
    if distance > 0.6:
        print("Not Same Person!")
    else:
        print("Same Person!")
        
facenet_base = load_model('facenet.h5')
detector = MTCNN()
face_detector = load_model("best_model.h5", custom_objects={'contrastive_loss':contrastive_loss})
print("all models successfully loaded!")

parser = argparse.ArgumentParser(description='Face Similarity Demo.')
parser.add_argument('-i1', "--image1", help = "path to first image")
parser.add_argument('-i2', "--image2", help = "path to second image")
args = parser.parse_args()

image1_path = args.image1
image2_path = args.image2
distance = get_distance(image1_path,image2_path)
show_results(distance)

          



