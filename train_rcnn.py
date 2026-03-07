import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import glob
import numpy as np

# Dataset paths
DATASET_DIR = "Dataset/ts/ts"
CLASSES = ['prohibitory', 'danger', 'mandatory', 'other'] # 4 classes
NUM_CLASSES = len(CLASSES) + 1 # +1 for background

class TrafficSignDataset(Dataset):
    def __init__(self, imgs_dir):
        self.imgs_dir = imgs_dir
        # Find all images
        self.imgs = glob.glob(os.path.join(imgs_dir, "*.jpg"))
        
    def __getitem__(self, idx):
        # Load image
        img_path = self.imgs[idx]
        label_path = img_path.replace('.jpg', '.txt')
        
        # Read image
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = F.to_tensor(img_rgb)
        
        _, h, w = img_tensor.shape
        
        boxes = []
        labels = []
        
        # Parse YOLO format: class x_center y_center width height (Normalized)
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    class_id, x_c, y_c, w_box, h_box = map(float, line.strip().split())
                    
                    # Faster R-CNN expects pixel coordinates [xmin, ymin, xmax, ymax]
                    xmin = (x_c - (w_box / 2)) * w
                    ymin = (y_c - (h_box / 2)) * h
                    xmax = (x_c + (w_box / 2)) * w
                    ymax = (y_c + (h_box / 2)) * h
                    
                    boxes.append([xmin, ymin, xmax, ymax])
                    # Faster RCNN class 0 is always background, so we shift YOLO classes by +1
                    labels.append(int(class_id) + 1)
        
        if len(boxes) == 0:
            # If no boxes, provide empty dummy tensors to prevent crashing
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        iscrowd = torch.zeros((labels.shape[0],), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        return img_tensor, target

    def __len__(self):
        return len(self.imgs)

# PyTorch requires a custom collate function for object detection variable sized lists
def collate_fn(batch):
    return tuple(zip(*batch))

def create_model(num_classes):
    # Load pre-trained Faster R-CNN on MS COCO
    # We use v2 weights for better initial accuracy
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    
    # Get the number of input features for the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one (untrained) that accommodates our 5 custom classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def main():
    print(f"Loading dataset from {DATASET_DIR}...")
    dataset = TrafficSignDataset(DATASET_DIR)
    
    # Split dataset into training and validation (80 / 20)
    torch.manual_seed(42)
    indices = torch.randperm(len(dataset)).tolist()
    train_split = int(0.8 * len(dataset))
    
    dataset_train = torch.utils.data.Subset(dataset, indices[:train_split])
    dataset_valid = torch.utils.data.Subset(dataset, indices[train_split:])

    train_loader = DataLoader(dataset_train, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_fn)
    valid_loader = DataLoader(dataset_valid, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn)

    print(f"Training on {len(dataset_train)} images, validating on {len(dataset_valid)} images.")

    # Configure hardware
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Training on hardware: {device}")

    # Initialize model
    model = create_model(num_classes=NUM_CLASSES)
    model.to(device)

    # Optimizer (Stochastic Gradient Descent is highly stable for R-CNN)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    # Learning rate scheduler (decreases learning rate dynamically as we get closer to the perfect weights)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 30 # 30 Epochs for extended training to reduce false positives

    print("--- 🚀 Starting Custom Faster R-CNN Transfer Learning ---")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for i, (images, targets) in enumerate(train_loader):
            # Move to GPU/CPU
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass mathematically calculates how wrong the AI's guesses were
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass (Update the neural network weights)
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            
            if (i+1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {losses.item():.4f}")

        # Update learning rate tracker
        lr_scheduler.step()
        print(f"✅ Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {epoch_loss/len(train_loader):.4f}")

    print("--- 🎉 Training Complete! ---")
    
    output_path = "Model/weights/best_rcnn.pth"
    torch.save(model.state_dict(), output_path)
    print(f"Custom Model Weights successfully saved to: {output_path}")

if __name__ == "__main__":
    main()
