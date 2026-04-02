import os
import cv2 # Görüntü işleme (okuma, yazma, watermark ekleme)
import glob # Dosya listesi almak için 
import torch # Tensor işlemleri, ekran kartı seçimi
import random
import numpy as np
import torch.nn as nn # Neural network kütüphanesi
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader # Veriyi modelimize uygun şekilde hazırlamak için torch kütüphanesi

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256

#5 epoch = tüm dataset’in 5 kez görülmesi
EPOCHS = 5


# Geçici watermark dataset oluşturma kısmı

# Bu fonksiyon: görüntüye sahte watermark ekler + mask oluşturur
def add_watermark(image):
    h, w, _ = image.shape

    text = random.choice(["WATERMARK", "COPY", "SAMPLE"])
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = random.uniform(0.8, 2)
    thickness = random.randint(1, 3)

    size = cv2.getTextSize(text, font, scale, thickness)[0]
    x = random.randint(0, max(1, w - size[0]))
    y = random.randint(size[1], h)

    overlay = image.copy()
    cv2.putText(overlay, text, (x, y), font, scale, (255,255,255), thickness)

    alpha = random.uniform(0.2, 0.5)
    wm = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

     # Mask oluştur (sadece watermark bölgesi beyaz)
    mask = np.zeros((h, w), dtype=np.uint8)

    cv2.putText(mask, text, (x, y), font, scale, 255, thickness)

    return wm, mask

def create_dataset():
    os.makedirs("dataset/images", exist_ok=True)
    os.makedirs("dataset/masks", exist_ok=True)

    files = os.listdir("clean_images")

    for i, f in enumerate(files):
        img = cv2.imread(os.path.join("clean_images", f))
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        wm, mask = add_watermark(img)

        cv2.imwrite(f"dataset/images/{i}.png", wm)
        cv2.imwrite(f"dataset/masks/{i}.png", mask)


# dataset classı

class WMDataset(Dataset):
    def __init__(self):
        self.imgs = sorted(glob.glob("dataset/images/*.png"))
        self.masks = sorted(glob.glob("dataset/masks/*.png"))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        # Görüntüyü oku ve normalize et (0-1 arası)
        img = cv2.imread(self.imgs[i]) / 255.0
        mask = cv2.imread(self.masks[i], 0) / 255.0

        img = torch.tensor(img).permute(2,0,1).float()
        mask = torch.tensor(mask).unsqueeze(0).float()

        return img, mask


# U-NET modeli

# Küçük yapı: 2 tane convolution katmanı
class DoubleConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), # Convolution
            nn.ReLU(),                     # Aktivasyon
            nn.Conv2d(o, o, 3, padding=1),
            nn.ReLU()
        )
    def forward(self,x): return self.net(x)

# U-Net: segmentation (mask bulma) modeli
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = DoubleConv(3,64)
        self.p1 = nn.MaxPool2d(2)

        self.d2 = DoubleConv(64,128)
        self.p2 = nn.MaxPool2d(2)

        self.mid = DoubleConv(128,256)

        self.u1 = nn.ConvTranspose2d(256,128,2,2)
        self.c1 = DoubleConv(256,128)

        self.u2 = nn.ConvTranspose2d(128,64,2,2)
        self.c2 = DoubleConv(128,64)

        self.out = nn.Conv2d(64,1,1)

    def forward(self,x):
        d1 = self.d1(x)     # Feature extraction
        d2 = self.d2(self.p1(d1))   # Downsample

        m = self.mid(self.p2(d2))  # Ortadaki katman

        u1 = self.u1(m)   # Upsample
        u1 = self.c1(torch.cat([u1,d2],1))  # Skip connection

        u2 = self.u2(u1)
        u2 = self.c2(torch.cat([u2,d1],1))

        return torch.sigmoid(self.out(u2))   # 0-1 arası mask tahmini


# Eğitim kısmı

def train():
    ds = WMDataset()
    dl = DataLoader(ds, batch_size=8, shuffle=True)

    model = UNet().to(DEVICE)

    # Optimizer → model nasıl öğrenir
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()
    
    # Eğitim döngüsü
    for e in range(EPOCHS):
        loop = tqdm(dl)
        for x,y in loop:
            x,y = x.to(DEVICE), y.to(DEVICE)

            pred = model(x)   # Model tahmin yapar
            loss = loss_fn(pred,y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            loop.set_description(f"Epoch {e}")
            loop.set_postfix(loss=loss.item())

    torch.save(model.state_dict(), "model.pth")


# Test ve Sonuç kaydetme kısmı

def test(img_path):
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
    model.eval()

    img = cv2.imread(img_path)
    orig = img.copy()

    resized = cv2.resize(img,(IMG_SIZE,IMG_SIZE))/255.0
    t = torch.tensor(resized).permute(2,0,1).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        pred = model(t)[0][0].cpu().numpy()

    mask = (pred>0.5).astype("uint8")*255
    mask = cv2.resize(mask,(orig.shape[1],orig.shape[0]))

    result = cv2.inpaint(orig, mask, 3, cv2.INPAINT_TELEA)

    cv2.imwrite("result.png", result)
    print("Saved result.png")




if __name__ == "__main__":
    print("1: dataset oluştur")
    print("2: train")
    print("3: test")

    choice = input("> ")

    if choice == "1":
        create_dataset()
    elif choice == "2":
        train()
    elif choice == "3":
        path = input("image path: ")
        test(path)