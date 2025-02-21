# derin-ogrenme-baslangici
# Kedi ve Köpek Görüntü Sınıflandırma

Bu proje, lojistik regresyon kullanarak kedi ve köpek görüntülerini sınıflandıran bir makine öğrenmesi modelini içermektedir. Model, verilen görüntüleri alıp işleyerek bunları iki sınıfa (kedi veya köpek) ayırmayı amaçlamaktadır.

## İçindekiler
- [Gereksinimler](#gereksinimler)
- [Veri Kümesi](#veri-kümesi)
- [Ön İşleme Adımları](#ön-i%CC%87%C5%9Fleme-ad%C4%B1mlar%C4%B1)
- [Model Eğitimi](#model-e%C4%9Fitimi)
- [Kullanım](#kullan%C4%B1m)
- [Sonuçlar](#sonu%C3%A7lar)

## Gereksinimler
Bu projeyi çalıştırmak için aşağıdaki kütüphaneleri yüklemeniz gerekmektedir:
```bash
pip install numpy pillow tqdm scikit-learn tensorflow
```

## Veri Kümesi
Model, kedi ve köpek görüntülerinden oluşan bir veri kümesi üzerinde eğitilmektedir. Veri kümesi şu şekilde organize edilmiştir:
```
/data/
   training_set/
       dogs/
       cats/
   test_set/
       dogs/
       cats/
```
Her sınıf, ilgili klasörde bulunan görüntü dosyalarından oluşmaktadır.

## Ön İşleme Adımları
Kod, aşağıdaki veri ön işleme adımlarını uygular:
1. **Görüntüleri yükleme ve boyutlandırma:** Bütün görüntüler 28x28 boyutuna getirilir.
2. **Veriyi Numpy dizisine çevirme:** Görüntüler vektör haline getirilerek normalize edilir (0-1 aralığına çekilir).
3. **Etiketleri sayısallaştırma:** Kedi = 1, Köpek = 0 olacak şekilde sayısallaştırma yapılır.
4. **Veriyi karıştırma:** Eğitim ve test verileri rastgele karıştırılır.

## Model Eğitimi
Lojistik regresyon modeli aşağıdaki adımlarla eğitilmektedir:
- **İleri yayılım:** Sigmoid fonksiyonu kullanılarak tahminler yapılır.
- **Kayıp fonksiyonu hesaplama:** Binary cross-entropy kullanılarak kayıp değeri hesaplanır.
- **Geri yayılım:** Ağırlıklar ve bias değeri güncellenir.
- **Optimizasyon:** Belirtilen iterasyon sayısı boyunca model eğitilir.

## Kullanım
Aşağıdaki kod ile modeli çalıştırabilirsiniz:
```python
history = model(X_train, y_train, X_test, y_test, num_iterations=2000, learning_rate=0.001, print_cost=True)
```
Bu satır, eğitim ve test doğruluk oranlarını ekrana yazdıracaktır.

## Sonuçlar
Eğitim ve test doğruluk oranları şu şekilde hesaplanmaktadır:
```python
print("Train Accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
print("Test Accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
```
Başarımı artırmak için farklı öğrenme oranları, iterasyon sayıları veya daha karmaşık derin öğrenme modelleri kullanılabilir.
