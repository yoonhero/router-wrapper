import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import einops
from torchvision.models import resnet34, resnet18
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import glob
from sklearn.model_selection import train_test_split
from datetime import datetime
import tqdm

from essential import exp_logger


def crop_white_border(image):
    black = 100
    _image = image.copy()
    if image.ndim == 3 and image.shape[2] == 3:
        _image = np.mean(image, axis=2, dtype=int)

    non_white_mask = _image < black
    non_white_rows = np.any(non_white_mask, axis=1)
    non_white_cols = np.any(non_white_mask, axis=0)

    row_start = np.argmax(non_white_rows)
    row_end = len(non_white_rows) - np.argmax(non_white_rows[::-1])
    col_start = np.argmax(non_white_cols)
    col_end = len(non_white_cols) -  np.argmax(non_white_cols[::-1])
    cropped_image = image[row_start:row_end, col_start:col_end]
    print(row_start, row_end, col_start)
    return cropped_image

UNK = "-"
# img_width = 280
# img_height = 80
vocabs = "abcdefghijklmnopqrstuvwxyz"
atoi = {vocab: i+1 for i, vocab in enumerate(vocabs)}
atoi[UNK] = 0
itoa = {i+1: vocab for i, vocab in enumerate(vocabs)}
itoa[0] = UNK
total_vocabs = len(vocabs)

class ClassificationDataset(Dataset):
    def __init__(self, image_paths, targets, resolution=(50, 180)):
        self.image_paths = image_paths
        self.resize = resolution
        self.transform = T.Compose([T.ToTensor(), T.Normalize(mean=(0.482), std=(0.229))])
        self.targets = targets
        # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, i):
        image = np.array(Image.open(self.image_paths[i]))
        image = Image.fromarray(crop_white_border(image)).convert('LA').resize((self.resize[1], self.resize[0]), resample=Image.BILINEAR)
        image = np.mean(image, axis=2, keepdims=True, dtype=float) / 255

        targets = self.targets[i]

        image = self.transform(image)

        return {
            "images": image.to(torch.float32),
            "targets": torch.tensor(targets, dtype=torch.float32)
        }


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)

class Model(nn.Module):

    def __init__(self, resolution=(50, 180), dims=256):
        super().__init__()
        resnet = resnet18(weights='DEFAULT')
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.feature_extract = nn.Sequential(
            *list(resnet.children())[:-3],
            nn.Conv2d(256, 256, kernel_size=(3, 6), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.resolution = resolution
        linear_input_size = self._calc_linear_layer()
        self.linear = nn.Linear(linear_input_size, dims)
        self.drop = nn.Dropout(0.5)

        self.lstm = nn.GRU(dims, dims // 2, num_layers=3, bidirectional=True, dropout=0.5, batch_first=True)
        self.projection = nn.Linear(dims, total_vocabs+1)

    def _calc_linear_layer(self):
        height, width = self.resolution
        dummy_input = torch.zeros(1, 1, height, width)
        x = self.feature_extract(dummy_input)
        x = x.permute(0, 3, 1, 2)
        conv_output = x.view(x.size(0), x.size(1), -1)
        return conv_output.shape[-1]
    
    def encode(self, x):
        x = self.feature_extract(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(x.size(0), x.size(1), -1)
        features = self.drop(F.relu(self.linear(x)))

        return features

    def forward(self, images, targets=None):
        features = self.encode(images)
        hiddens, _ = self.lstm(features)

        x = self.projection(hiddens)
        
        if targets is not None:
            x = x.permute(1, 0, 2)
            loss = self.ctc_loss(x, targets)
            return x, loss
        
        return x, None

    @staticmethod
    def ctc_loss(x, targets):
        B = x.size(1)

        log_probs = F.log_softmax(x, 2)
        input_lengths = torch.full(size=(B,), fill_value=log_probs.size(0), dtype=torch.int32)
        target_lengths = torch.full(size=(B,), fill_value=targets.size(1), dtype=torch.int32)

        loss = nn.CTCLoss(blank=0)(log_probs, targets, input_lengths, target_lengths)
        return loss

def get_string_from_torch(arr, from_decoded=False):
    texts = []
    originals = []
    for i in range(arr.shape[0]):
        string = "" 
        e = arr[i]
        for j in range(len(e)): string += itoa[int(e[j])]
        originals.append(string)
        if from_decoded:
            string = string.split(UNK)
            texts.append("".join(string))
        else: texts.append(string)
    return texts, originals

def decode_pred(predictions):
    predictions = predictions.permute(1, 0, 2)
    predictions = torch.softmax(predictions, 2)
    predictions = torch.argmax(predictions, 2)
    predictions = predictions.detach().cpu().numpy()
    return get_string_from_torch(predictions, from_decoded=True)

if __name__ == "__main__":
    image_paths = glob.glob("./train/*.png")[:]
    original_targets = [image_path.split("/")[-1][:-4] for image_path in image_paths]
    targets = [[atoi[t] for t in word] for word in original_targets] 

    X_train, X_test, y_train, y_test = train_test_split(
        image_paths, targets, test_size=0.1, random_state=42)
    train_dataset = ClassificationDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = ClassificationDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"Train: {len(X_train)}    Test: {len(X_test)}")

    device = torch.device("cpu")
    model = Model().to(device)
    # model.apply(initialize_weights)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=5, verbose=True)

    best_acc = 0.0
    train_loss_data = []
    test_loss_data = []
    accuracy_data = []
    start = datetime.now()

    run = exp_logger.Logger(exp_logger.OFFLINE, run_name="Solving Captcha", configs={})

    for epoch in range(1000):
        # Train
        fin_loss = 0
        model.train()
        for data in tqdm.tqdm(train_loader):
            for key, value in data.items():
                data[key] = value.to(device)
            
            optimizer.zero_grad()
            _, loss = model(**data)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            fin_loss += loss.item()
            # break
        fin_loss /= len(train_loader)
        train_loss_data.append(fin_loss)

        fin_loss = 0
        fin_gt = []
        fin_preds = []
        with torch.no_grad():
            model.eval()
            for data in tqdm.tqdm(test_loader):
                for key, value in data.items():
                    data[key] = value.to(device)
                
                preds, loss = model(**data)
                fin_loss += loss.item()
                fin_gt.append(data["targets"].detach().cpu().numpy())
                fin_preds.append(preds)
                # break
            fin_loss /= len(test_loader)
        test_loss_data.append(fin_loss)

        valid_captcha_preds = []
        total = 0
        acc = 0
        data = {"gt":[], "pred": []}

        def similarity(word1, word2):
            # simple sim
            return sum([1 if word1[i] == word2[i] else 0 for i in range(min(len(word1), len(word2)))]) / len(word1)

        for b_gt, b_pred in zip(fin_gt, fin_preds):
            gt_texts, _ = get_string_from_torch(b_gt)
            pred_texts, original = decode_pred(b_pred)

            val = [1 if g == p else 0 for g, p in zip(gt_texts, pred_texts)]
            for g, p in zip(gt_texts, pred_texts):
                if g==p: print("SAME:", g, p)
                elif similarity(g, p) > 0.8:
                    print(f"Sim: {similarity(g, p)} over {g} and {p}")
                
            acc += sum(val)
            total += len(val)
            data["gt"].extend(gt_texts)
            data["pred"].extend(original)

        run.add_texts(data, iteration=epoch)
        print(gt_texts, pred_texts)

        acc /= total

        scheduler.step(test_loss_data[-1])
        # run.save_images("train", [], epoch, [""])
        if acc > best_acc:
            best_acc = acc
            run.log_ckp(model, optimizer, f"{start}-{epoch}-acc-{acc}.pkl")
            # torch.save(model.state_dict(), f"./tmp/")

        run.log({"train/loss": train_loss_data[-1], "test/loss": test_loss_data[-1], "test/acc": acc}, iteration=epoch)
        print(f"Epoch {epoch}: Train loss: {train_loss_data[-1]}    Test loss: {test_loss_data[-1]}   Accuracy: {acc}")
