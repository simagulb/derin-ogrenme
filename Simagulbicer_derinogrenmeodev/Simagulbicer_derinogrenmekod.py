import time, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from sklearn.metrics import mean_squared_error, r2_score

SEED = 17
EPOCHS = 300
BATCH = 64

p = np.linspace(-2, 2, 500).reshape(-1, 1).astype("float32")
y_true = (1 + np.sin(np.pi + p / 4)).astype("float32")


def egit(act="relu", lr=0.1, opt="SGD", init="normal", layers=1, neurons=2):
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)

    Init = tf.keras.initializers.GlorotNormal if init == "normal" else tf.keras.initializers.GlorotUniform
    activation = tf.nn.leaky_relu if act == "leaky_relu" else act

    katmanlar = [Input(shape=(1,))]
    for i in range(layers):
        katmanlar.append(Dense(neurons, activation=activation,
                               kernel_initializer=Init(seed=SEED + i),
                               bias_initializer=tf.keras.initializers.Constant(0.5)))
    katmanlar.append(Dense(1, activation="linear", kernel_initializer=Init(seed=SEED + 99)))
    model = Sequential(katmanlar)

    if opt == "SGD":
        o = tf.keras.optimizers.SGD(learning_rate=lr, clipnorm=1.0)
    elif opt == "RMSProp":
        o = tf.keras.optimizers.RMSprop(learning_rate=lr, clipnorm=1.0)
    else:
        o = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    model.compile(optimizer=o, loss="mse")

    t0 = time.time()
    h = model.fit(p, y_true, epochs=EPOCHS, batch_size=BATCH, verbose=0)
    sure = time.time() - t0

    tahmin = model.predict(p, verbose=0)
    mse = mean_squared_error(y_true, tahmin)
    r2 = r2_score(y_true, tahmin)
    return mse, r2, sure, h.history["loss"], tahmin


def tablo_olustur(baslik, kolon, secenekler):
    print(baslik)
    sonuclar = []
    for etiket, kw in secenekler:
        sonuclar.append((etiket,) + egit(**kw))
    en_iyi = int(np.argmin([s[1] for s in sonuclar]))

    df = pd.DataFrame({
        kolon: [s[0] for s in sonuclar],
        "MSE": [f"{s[1]:.6f}" for s in sonuclar],
        "R2": [f"{s[2]:.4f}" for s in sonuclar],
        "Süre (sn)": [f"{s[3]:.2f}" for s in sonuclar],
    })
    print(df.to_string(index=False))
    print("En iyi:", sonuclar[en_iyi][0])

    et, mse, r2, sure, hist, tahmin = sonuclar[en_iyi]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(hist)
    ax[0].set_yscale("log")
    ax[0].set_title(f"Epoch - Loss ({et})")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("MSE (log)")
    ax[0].grid(alpha=0.3)
    ax[1].plot(p, y_true, label="Hedef", lw=2)
    ax[1].plot(p, tahmin, "--", label="Ağ çıkışı", lw=2)
    ax[1].set_title("Hedef ve Ağ Çıkışı")
    ax[1].set_xlabel("p")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    ax[2].scatter(y_true, tahmin, s=8)
    ax[2].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
    ax[2].set_title(f"Gerçek - Tahmin (R² = {r2:.4f})")
    ax[2].set_xlabel("Gerçek y")
    ax[2].set_ylabel("Tahmin")
    ax[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    return df


aktivasyonlar = {
    "Lineer": "linear",
    "Sigmoid": "sigmoid",
    "Hiperbolik Tanjant": "tanh",
    "ReLU": "relu",
    "LeakyReLU": "leaky_relu",
    "ELU": "elu",
    "SiLU": "swish",
}
tablo1 = tablo_olustur(
    "Tablo 1: Aktivasyon Fonksiyonlarının Değişimi",
    "Aktivasyon",
    [(ad, dict(act=a)) for ad, a in aktivasyonlar.items()],
)


tablo2 = tablo_olustur(
    "Tablo 2: Öğrenme Oranının Etkisi",
    "Öğrenme Oranı",
    [(str(lr), dict(lr=lr)) for lr in (0.1, 0.01, 0.001)],
)


tablo3 = tablo_olustur(
    "Tablo 3: Öğrenme Oranı Ayar Yöntemi (Optimizatör) Etkisi",
    "Ayar Yöntemi",
    [(o, dict(opt=o)) for o in ("SGD", "RMSProp", "ADAM")],
)


tablo4 = tablo_olustur(
    "Tablo 4: Başlangıç Ağırlıklarının Olasılık Dağılımının Etkisi",
    "Dağılım",
    [("Normal", dict(init="normal")), ("Uniform", dict(init="uniform"))],
)


tablo5 = tablo_olustur(
    "Tablo 5: Gizli Katman Sayısının Etkisi",
    "Katman Sayısı",
    [(str(k), dict(layers=k)) for k in (1, 2, 5, 10, 20)],
)


tablo6 = tablo_olustur(
    "Tablo 6: Gizli Katmandaki Nöron Sayısının Etkisi",
    "Nöron Sayısı",
    [(str(n), dict(neurons=n)) for n in (2, 5, 10, 20, 50)],
)
