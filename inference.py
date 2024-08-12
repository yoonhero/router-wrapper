import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

from train import Model, crop_white_border, decode_pred

class Inference():
    def __init__(self, ckpt_path: str, device: str="cpu"):
        self.model = Model().to(device)
        self.model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        self.model.eval()
        self.transform = T.Compose([T.ToTensor(), T.Normalize(mean=(0.482), std=(0.229))])
    
    def load_img(self, image):
        if not isinstance(image, type(Image)):
            image = Image.open(image)
        image = np.array(image)
        image = Image.fromarray(crop_white_border(image)).convert('LA').resize((180, 50), resample=Image.BILINEAR)
        image.show()
        image = np.mean(image, axis=2, keepdims=True, dtype=float) / 255
        image = self.transform(image).to(torch.float32)

        return image

    def pred(self, image):
        x = self.load_img(image)
        x = x.unsqueeze(0)
        predictions, _ = self.model(images=x)

        t, _ = decode_pred(predictions)
        return self.calc("".join(t))
    
    # Calculate fidelity
    def calc(self, pred):
        if len(pred) != 5:
            return pred, 0
        # c and e are so similar because the center line make them non-distinguishable even a human just for me...
        elif "e" in pred or "c" in pred:
            return pred, 0.5
        
        return pred, 1


if __name__ == "__main__":
    # model = Model()
    # torch.save(torch.load("model.pt")["model"], "model.pt")
    inference = Inference("./model.pt", "cpu")
    image = Image.open("puzzle.png")
    print(inference.pred(image))