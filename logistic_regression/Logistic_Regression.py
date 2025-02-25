from PIL import Image
import os
from tqdm import tqdm
import numpy as np
from sklearn.utils import shuffle 
#import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

# Verimizde iki tür sınıf vardır kedi ve köpek
labels = ["dogs","cats"]

X_train = []
y_train = []

X_test = []
y_test = []

image_size = 28

# Bu adımda görüntü veri kümesini eğitim (training_set) ve test (test_set) olarak ikiye ayırarak yükler.
# Veriler, data/ klasörünün içindeki training_set/ ve test_set/ dizinlerinden okunur.
for label in labels:
    folderPath = os.path.join("data/training_set", label)
    for file in tqdm(os.listdir(folderPath)):
        
        img = Image.open(os.path.join(folderPath, file))
        img = img.resize((image_size, image_size))
      
        X_train.append(img)
        y_train.append(label)
        
for label in labels:
    folderPath = os.path.join("data/test_set", label)
    for file in tqdm(os.listdir(folderPath)):
        
        img = Image.open(os.path.join(folderPath, file))
        img = img.resize((image_size, image_size))
      
        X_test.append(img)
        y_test.append(label)
     
# Okunan görüntüler numpy tipine çevrilir
# Her işlem train ve test setleri için gerçekleştirilir
X_train = np.array(X_train).reshape(-1,image_size*image_size*3)
X_train = X_train/255.0
y_train = np.array(y_train)

y_train_new = []

for i in y_train:
    y_train_new.append(labels.index(i))
y_train = y_train_new
y_train = np.array(y_train)


X_test = np.array(X_test).reshape(-1,image_size*image_size*3)
X_test = X_test/255.0
y_test = np.array(y_test)

y_test_new = []

for i in y_test:
    y_test_new.append(labels.index(i))
y_test = y_test_new
y_test = np.array(y_test)

    

# Verimiz kedi ve köpek resimlerinin sıralı birşekilde olmasından dolayı rastgele bir şekilde karıştırılır
X_train, y_train = shuffle(X_train, y_train, random_state=42)
X_test, y_test = shuffle(X_test, y_test, random_state=42)

# Son olarak train ve test verilerini modelimize girdi olarak vermek için
# uygun boyutlara getirip veri ön işleme adımlarını tamamlıyoruz
X_train = X_train.T
X_test = X_test.T
y_train = y_train.reshape(1,-1)
y_test = y_test.reshape(1,-1)




def sigmoid(z):

    s = 1/(1+np.exp(-z))
    return s


def parametre_başlat(dim):

    # dim değişkeni eğitim verisindeki örneğin özellik sayısıdır 
    # w = np.zeros(dim,1) sütunu 1 yapmamızın sebebi logistic regressionu öğrenmek için başlangıçta 1 nöronla başlatıcaz
    w = np.zeros((dim,1))
    b = 0.0


    return w, b

def yayilim(w, b, X, Y):

    # m sayısı X deki örnek sayısını temsil eder
    m = X.shape[1]

    # ileri yayılım 
    z = np.dot(w.T, X) + b
    A = sigmoid(z)

    # cost hesaplama
    cost = -(1/m)*(np.sum(Y*np.log(A)+(1-Y)*np.log(1-A)))

    # geri yayalım
    dw = np.dot(X,(A-Y).T)/m
    db = (np.sum(A-Y))/m

    # costu array dizisinden skaler bir sayıya çeviriyoruz
    cost = np.squeeze(np.array(cost))

    grads = {"dw":dw,
             "db":db}

    return grads, cost



def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.5, print_cost=False):

    costs = []

    # Verilen iterasyon sayısı kadar döngüye girilerek w, b parametrelerini güncelliyoruz
    for i in range(num_iterations):
        grads, cost = yayilim(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate*dw
        b = b - learning_rate*db

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w":w,
              "b":b}
    grads = {"dw":dw,
             "db":db}
    
    return params, grads, costs

def predict(w, b, X):

    # Bu fonksiyonu eğittiğimiz modelin tahmin işlemini gerçekleştirmek için kullanıcaz 
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))

    z = np.dot(w.T, X) + b
    A = sigmoid(z)

    for i in range(X.shape[1]):

        # A matrisi modelimizin hesaplama sonucu elde ettiği değerlerdir
        # Bunları köpek ise 0, kedi ise 1 olarak sınıflandıracağız
        # 0.5 ve altı değerleri köpek, 0.5 üstü değerleri kedi olarak tanımlıyoruz 
        if A[0,i] > 0.5:
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0

    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):

    # Tüm yazdığımız fonksiyonların bir araya getirdiğimiz aşamasıdır bu fonksiyon
    w, b = parametre_başlat(X_train.shape[0])
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = params["w"]
    b = params["b"]
    
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)

    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    res = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return 

# 
history = model(X_train, y_train, X_test, y_test, num_iterations=2000, learning_rate=0.001, print_cost=True )




