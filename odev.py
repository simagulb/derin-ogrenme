# Girdi matrisi (X) ve AND kapısı için hedef çıktılar (y)
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 0, 0, 1]

# Elle rastgele seçilen ağırlıklar (w), sapma (b) ve öğrenme hızı (lr)
w = [0.50, -0.72]
b = 0.30
lr = 0.1

# Batch size = Epoch size (Hata 0 olana kadar eğit)
while True:
    dw, db, toplam_hata = [0, 0], 0, 0
    for x, t in zip(X, y):
        # Toplam fonksiyonu ve basamak (step) aktivasyon
        cikti = 1 if (w[0] * x[0] + w[1] * x[1] + b) > 0 else 0
        hata = t - cikti
        
        # Değişimleri biriktir
        dw[0] += lr * hata * x[0]
        dw[1] += lr * hata * x[1]
        db += lr * hata
        toplam_hata += abs(hata)

    # Epoch sonunda ağırlıkları tek seferde güncelle (Batch mantığı)
    w[0] += dw[0]
    w[1] += dw[1]
    b += db

    # Denklem sonucu 0 hataya ulaşınca döngü sonlanır
    if toplam_hata == 0:
        break

print("Eğitim Tamamlandı!")
print("Öğrenilen Ağırlıklar (w):", [round(v, 2) for v in w])
print("Öğrenilen Bias (b):", round(b, 2))

# Test çıktıları
for x, t in zip(X, y):
    tahmin = 1 if (w[0] * x[0] + w[1] * x[1] + b) > 0 else 0
    print(f"Girdi: {x} -> Beklenen: {t} | Çıktı: {tahmin}")
