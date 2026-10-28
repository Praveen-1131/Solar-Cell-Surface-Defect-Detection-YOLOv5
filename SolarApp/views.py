from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
from django.conf import settings
import os
import io
import base64
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
import cv2
import pymysql
import pandas as pd
import numpy as np
import cv2
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
import pickle
import os
import matplotlib.pyplot as plt
import keras
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.applications import VGG16
from keras.applications import ResNet50
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Lambda, Activation, Flatten, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop, SGD
from Attention import attention #importing attention layer
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
import seaborn as sns

global username
global X_train, X_test, y_train, y_test, X, Y, labels
#define global variabels
X = [] #use to store image data
Y = [] #use to store label
P = [] #use to store bounding box or defect probability
labels = ['Mono', 'Poly']

#define global variables to calculate and store accuracy and other metrics
precision = []
recall = []
fscore = []
accuracy = []
mAP = []

X = np.load('model/X.txt.npy')#load all processed images
Y = np.load('model/Y.txt.npy')                    
P = np.load('model/P.txt.npy')

#function to calculate all metrics
def calculateMetrics(algorithm, predict, y_test):
    a = (accuracy_score(y_test,predict)*100)
    p = (precision_score(y_test, predict,average='macro') * 100)
    r = (recall_score(y_test, predict,average='macro') * 100)
    f = (f1_score(y_test, predict,average='macro') * 100)
    ma = average_precision_score(y_test, predict)*100
    a = round(a, 3)
    p = round(p, 3)
    r = round(r, 3)
    f = round(f, 3)
    ma = round(ma, 3)
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    mAP.append(ma)
    return algorithm

#preprocess image features and then split dataset into train and test
indices = np.arange(X.shape[0])
np.random.shuffle(indices)#shuffle image pixels
X = X[indices]
Y = Y[indices]
P = P[indices]
Y = to_categorical(Y)
#split dataset into train and test where 20% dataset size for testing and 80% for testing
split = train_test_split(X, Y, P, test_size=0.20, random_state=42)
(trainImages, testImages) = split[:2] #get train and test images
(trainLabels, testLabels) = split[2:4]#get train and test labels
(trainBBoxes, testBBoxes) = split[4:6]#get train bounding boxes as probability

#train existing FRCNN model
#using Resnet50 pretrained model as the base structure for the FasterRCNN
rcnn = ResNet50(weights="imagenet", include_top=False, input_tensor=Input(shape=(X.shape[1], X.shape[2], X.shape[3])))
rcnn.trainable = False
#get output
flatten = rcnn.output
#define FRCNN layers by converting multi dimension input features as single dimension
flatten = Flatten()(flatten)
#define FRCNN layer with 16 neurons for features filtration by using flatten
bboxHead = Dense(16, activation="relu")(flatten)
#define layer for bounding box probability
bboxHead = Dense(8, activation="relu")(bboxHead)
bboxHead = Dense(8, activation="relu")(bboxHead)
#define prediction for bounding boxes
bboxHead = Dense(1, activation="sigmoid", name="bounding_box")(bboxHead)
#define another layer for images training 
softmaxHead = Dense(16, activation="relu")(flatten)
softmaxHead = Dropout(0.2)(softmaxHead)
softmaxHead = Dense(8, activation="relu")(softmaxHead)
softmaxHead = Dropout(0.2)(softmaxHead)
#define classification layer as two classes Mono and Poly
softmaxHead = Dense(2, activation="softmax", name="class_label")(softmaxHead)
frcnn_model = Model(inputs=rcnn.input, outputs=(bboxHead, softmaxHead))
losses = {"class_label": "categorical_crossentropy", "bounding_box": "mean_squared_error"}
lossWeights = {"class_label": 1.0, "bounding_box": 1.0}
opt = Adam(lr=1e-4)
#compile and train the model
frcnn_model.compile(loss=losses, optimizer=opt, metrics=["accuracy"], loss_weights=lossWeights)
if os.path.exists("model/frcnn.hdf5") == False:
    trainTargets = {"class_label": trainLabels, "bounding_box": trainBBoxes}
    testTargets = {"class_label": testLabels, "bounding_box": testBBoxes}
    model_check_point = ModelCheckpoint(filepath='model/frcnn.hdf5', verbose = 1, save_best_only = True)
    hist = frcnn_model.fit(trainImages, trainTargets, validation_data=(testImages, testTargets), batch_size=32, epochs=10, verbose=1,callbacks=[model_check_point])
    f = open('model/frcnn.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()
else:
    frcnn_model = load_model('model/frcnn.hdf5')
predict = frcnn_model.predict(testImages)[1]#perform prediction on test data
predict = np.argmax(predict, axis=1)
testY = np.argmax(testLabels, axis=1)
calculateMetrics("Faster RCNN", predict, testY)#call this function to calculate accuracy and other metrics

#train propose YoloV5 with CA attention model
#define input shape
input_img = Input(shape=(32, 32, 3))
#create YoloV4 layers with 32, 64 and 512 neurons or data filteration size
x = Conv2D(32, (3, 3), padding = 'same', activation = 'relu')(input_img)
x = Conv2D(32, (3, 3), padding = 'same', activation = 'relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), padding = 'same', activation = 'relu')(x)
x = Conv2D(64, (3, 3), padding = 'same', activation = 'relu')(x)
x = MaxPooling2D((2, 2))(x)
#adding CA attention layer
x = attention(return_sequences=True,name='attention')(x)
x = Flatten()(x)
#define output layer with 4 bounding box coordinate and 1 weapan class
x = Dense(512, activation = 'relu')(x)
x = Dense(512, activation = 'relu')(x)
x_bb = Dense(1, name='bb')(x)
x_class = Dense(2, activation='softmax', name='class')(x)
#create yolo Model with above input details
yolo_model = Model([input_img], [x_bb, x_class])
#compile the model
yolo_model.compile(Adam(lr=0.001), loss=['mse', 'categorical_crossentropy'], metrics=['accuracy'])
if os.path.exists("model/yolo_weights.hdf5") == False:#if model not trained then train the model
    model_check_point = ModelCheckpoint(filepath='model/yolo_weights.hdf5', verbose = 1, save_best_only = True)
    hist = yolo_model.fit(trainImages, [trainBBoxes, trainLabels], batch_size=32, epochs=10, validation_split=0.2, callbacks=[model_check_point])
    f = open('model/yolo_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:#if model already trained then load it
    yolo_model = load_model("model/yolo_weights.hdf5", custom_objects={'attention': attention})
predict = yolo_model.predict(testImages)#perform prediction on test data
predict = np.argmax(predict[1], axis=1)
test = np.argmax(testLabels, axis=1)
calculateMetrics("Faster RCNN", predict, test)#call this function to calculate accuracy and other metrics

#train Yolov6 as the extension model as this model is the advance version fo Yolov5 and can have more classification
#accuracy compare to propose YOLOV5 with attention
#here we are modifying YOLOv6 structure by applying VGG16 as the base model and then defining Yolov6 architecture
yolov6 = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(X.shape[1], X.shape[2], X.shape[3])))
yolov6.trainable = False
#get output of VGG
flatten = yolov6.output
flatten = Flatten()(flatten)
#define new layers of YoloV6
bboxHead = Dense(16, activation="relu")(flatten)
bboxHead = Dense(8, activation="relu")(bboxHead)
bboxHead = Dense(8, activation="relu")(bboxHead)
bboxHead = Dense(1, activation="sigmoid", name="bounding_box")(bboxHead)
softmaxHead = Dense(16, activation="relu")(flatten)
softmaxHead = Dropout(0.2)(softmaxHead)
softmaxHead = Dense(8, activation="relu")(softmaxHead)
softmaxHead = Dropout(0.2)(softmaxHead)
softmaxHead = Dense(2, activation="softmax", name="class_label")(softmaxHead)
#crete yolov6 model
yolov6_model = Model(inputs=yolov6.input, outputs=(bboxHead, softmaxHead))
#define parameters
losses = {"class_label": "categorical_crossentropy", "bounding_box": "mean_squared_error"}
lossWeights = {"class_label": 1.0, "bounding_box": 1.0}
trainTargets = {"class_label": trainLabels, "bounding_box": trainBBoxes}
testTargets = {"class_label": testLabels, "bounding_box": testBBoxes}
opt = Adam(lr=1e-4)
#compile and train the model
yolov6_model.compile(loss=losses, optimizer=opt, metrics=["accuracy"], loss_weights=lossWeights)
if os.path.exists("model/yolov6.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/yolov6.hdf5', verbose = 1, save_best_only = True)
    hist = yolov6_model.fit(trainImages, trainTargets, validation_data=(testImages, testTargets), batch_size=32, epochs=10, verbose=1,callbacks=[model_check_point])
    f = open('model/yolov6.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()
else:
    yolov6_model = load_model('model/yolov6.hdf5')
predict = yolov6_model.predict(testImages)[1]#perform prediction on test data
predict = np.argmax(predict, axis=1)
testY = np.argmax(testLabels, axis=1)
calculateMetrics("Extension YoloV6", predict, testY)#call this function to calculate accuracy and other metrics
conf_matrix = confusion_matrix(predict, testY)

def Predict(request):
    if request.method == 'GET':
        return render(request, 'Predict.html', {})

def PredictAction(request):
    if request.method == 'POST':
        global labels
        yolov6_model = load_model('model/yolov6.hdf5')
        myfile = request.FILES['t1'].read()
        if os.path.exists('SolarApp/static/test.jpg'):
            os.remove('SolarApp/static/test.jpg')
        with open('SolarApp/static/test.jpg', "wb") as file:
            file.write(myfile)
        file.close()
        img = cv2.imread('SolarApp/static/test.jpg')#read test image
        img = cv2.resize(img, (32, 32), interpolation = cv2.INTER_CUBIC) #scale imaage
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#hsv transform
        img = cv2.flip(img, 1)#flip images
        img = img.reshape(1,32,32,3)#convert image as 4 dimension
        predict = yolov6_model.predict(img)#predict solar defect from test image
        predict_label = predict[1] #get classification defect labels
        defect_probability = predict[0][0][0]#get defect probability
        predict_label = np.argmax(predict_label)
        img = cv2.imread('SolarApp/static/test.jpg')
        img = cv2.resize(img, (600,400))#display image with predicted output
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.putText(img, 'Predicted As : '+labels[predict_label], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
        cv2.putText(img, 'Defect Probability : '+str(defect_probability), (10, 65),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
        plt.imshow(img)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.clf()
        plt.cla()
        context= {'data':'<font size="3" color="blue">Predicted As : '+labels[predict_label]+'<br/>Defect Probability % : '+str(defect_probability)+'</font>', 'img': img_b64}
        return render(request, 'UserScreen.html', context)

def TrainModel(request):
    if request.method == 'GET':
        global X_train, X_test, y_train, y_test, labels
        global accuracy, precision, recall, fscore, conf_matrix, mAP
        output='<table border=1 align=center width=100%><tr><th><font size="3" color="black">Algorithm Name</th><th><font size="3" color="black">Accuracy</th>'
        output += '<th><font size="3" color="black">Precision</th><th><font size="3" color="black">Recall</th><th><font size="3" color="black">FSCORE</th><th><font size="3" color="black">MAP</th></tr>'
        algorithms = ["Existing Faster-RCNN","Propose YoloV5 with CA Attention", "Extension YoloV6"]
        for i in range(len(algorithms)):
            output += '<td><font size="3" color="black">'+algorithms[i]+'</td><td><font size="3" color="black">'+str(accuracy[i])+'</td><td><font size="3" color="black">'+str(precision[i])+'</td>'
            output += '<td><font size="3" color="black">'+str(recall[i])+'</td><td><font size="3" color="black">'+str(fscore[i])+'</td><td><font size="3" color="black">'+str(mAP[i])+'</td></tr>'
        output+= "</table></br>"
        figure, axis = plt.subplots(nrows=1, ncols=2,figsize=(10, 3))#display original and predicted segmented image
        axis[0].set_title("Confusion Matrix Prediction Graph")
        axis[1].set_title("All Algorithms Performance Graph")
        ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g", ax=axis[0]);
        ax.set_ylim([0,len(labels)])    
        df = pd.DataFrame([['Existing Faster-RCNN','Precision',precision[0]],['Existing Faster-RCNN','Recall',recall[0]],['Existing Faster-RCNN','F1 Score',fscore[0]],['Existing Faster-RCNN','Accuracy',accuracy[0]],
                           ['Propose Yolov5 with CA Attention','Precision',precision[1]],['Propose Yolov5 with CA Attention','Recall',recall[1]],['Propose Yolov5 with CA Attention','F1 Score',fscore[1]],['Propose Yolov5 with CA Attention','Accuracy',accuracy[1]],
                           ['Extension YoloV6','Precision',precision[2]],['Extension YoloV6','Recall',recall[2]],['Extension YoloV6','F1 Score',fscore[2]],['Extension YoloV6','Accuracy',accuracy[2]],
                          ],columns=['Parameters','Algorithms','Value'])
        df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar', ax=axis[1])
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        #plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.clf()
        plt.cla()
        context= {'data':output, 'img': img_b64}
        return render(request, 'UserScreen.html', context)

def LoadDataset(request):    
    if request.method == 'GET':
        global X_train, X_test, y_train, y_test, X, Y, labels
        output = '<font size="3" color="black">ELPV Solar Surface Images Dataset Loaded</font><br/>'
        output += '<font size="3" color="blue">Total images found in Dataset = '+str(X.shape[0])+'</font><br/>'
        output += '<font size="3" color="blue">Different Class Labels found in Dataset = '+str(labels)+'</font><br/><br/>'
        output += '<font size="3" color="black">Dataset Train & Test Split details</font><br/>'
        output += '<font size="3" color="blue">80% dataset images used to train all Architectures = '+str(trainImages.shape[0])+'</font><br/>'
        output += '<font size="3" color="blue">20% dataset images used to test all Architectures = '+str(testImages.shape[0])+'</font><br/>'
        context= {'data':output}
        return render(request, 'UserScreen.html', context)

def RegisterAction(request):
    if request.method == 'POST':
        global username
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        email = request.POST.get('t4', False)
        address = request.POST.get('t5', False)
        
        output = "none"
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '2004', database = 'solar',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username:
                    output = username+" Username already exists"
                    break                
        if output == "none":
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '2004', database = 'solar',charset='utf8')
            db_cursor = db_connection.cursor()
            student_sql_query = "INSERT INTO register(username,password,contact,email,address) VALUES('"+username+"','"+password+"','"+contact+"','"+email+"','"+address+"')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            print(db_cursor.rowcount, "Record Inserted")
            if db_cursor.rowcount == 1:
                output = "Signup process completed. Login to perform Solar Surface Defect Detection"
        context= {'data':output}
        return render(request, 'Register.html', context)    

def UserLoginAction(request):
    global username
    if request.method == 'POST':
        global username
        status = "none"
        users = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '2004', database = 'solar',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username,password FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == users and row[1] == password:
                    username = users
                    status = "success"
                    break
        if status == 'success':
            context= {'data':'Welcome '+username}
            return render(request, "UserScreen.html", context)
        else:
            context= {'data':'Invalid username'}
            return render(request, 'UserLogin.html', context)

def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def Register(request):
    if request.method == 'GET':
       return render(request, 'Register.html', {})
