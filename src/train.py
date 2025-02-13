from src.utils import BCEDiceLoss, dice_coefficient, DiceLoss
import torch
import torch.optim as optim
import csv
import torch.nn as nn
import os


def train_UNet(model, train_loader, valid_loader, device, num_epochs=500, lr=1e-3,
               log_path='./logs/train_log2.csv', model_path='./models/model1/best_model2.pth',patience=50):
    # Use the combined BCE + Dice loss (ensure that BCEDiceLoss is defined/imported)
    criterion = BCEDiceLoss()
    
    # Adam with weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)
    
    best_loss = float('inf')
    patience_counter = 0
    model.to(device)

    # Open log file and write header (including Valid Dice)
    with open(log_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Loss", "Valid Loss", "Valid Dice"])

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        total_train_samples = 0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            batch_size_curr = images.size(0)
            train_loss += loss.item() * batch_size_curr
            total_train_samples += batch_size_curr
        
        train_loss = train_loss / total_train_samples

        model.eval()
        valid_loss = 0.0
        total_valid_samples = 0
        valid_dice = 0.0

        with torch.no_grad():
            for images, masks in valid_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                batch_size_curr = images.size(0)
                valid_loss += loss.item() * batch_size_curr
                total_valid_samples += batch_size_curr

                # Compute dice coefficient for the batch and accumulate
                batch_dice = dice_coefficient(outputs, masks)
                valid_dice += batch_dice * batch_size_curr

        valid_loss = valid_loss / total_valid_samples
        valid_dice = valid_dice / total_valid_samples
        
        # Step the cosine annealing scheduler
        scheduler.step()

        with open(log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, valid_loss, valid_dice])

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), model_path)
            patience_counter=0
        else:
            patience_counter+=1
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, "
              f"Valid Loss: {valid_loss:.4f}, Valid Dice: {valid_dice:.4f}, Patience: {patience_counter}")

        if patience_counter>=patience:
            print('Early Stopping Triggered!')
            break

def train_classifier(model, train_loader, valid_loader, device, num_epochs=500, lr=1e-3,
                     log_path='./logs/train_log.csv', model_path='./models/best_model.pth', patience=10):
    # Use CrossEntropyLoss for classification
    criterion = nn.CrossEntropyLoss()  #changed
    # Adam optimizer with weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  #changed
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)  #changed

    best_loss = float('inf')
    patience_counter = 0
    model.to(device)

    # Create directories for logs and model if they don't exist
    os.makedirs(os.path.dirname(log_path), exist_ok=True)  #changed
    os.makedirs(os.path.dirname(model_path), exist_ok=True)  #changed

    # Open log file and write header
    with open(log_path, mode='w', newline='') as f:  #changed
        writer = csv.writer(f)  #changed
        writer.writerow(["Epoch", "Train Loss", "Train Accuracy", "Valid Loss", "Valid Accuracy"])  #changed

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  #changed
            optimizer.zero_grad()  #changed
            outputs = model(images)  #changed
            loss = criterion(outputs, labels)  #changed
            loss.backward()  #changed
            optimizer.step()  #changed

            train_loss += loss.item() * images.size(0)  #changed
            _, predicted = torch.max(outputs, 1)  #changed
            correct_train += (predicted == labels).sum().item()  #changed
            total_train += labels.size(0)  #changed

        train_loss = train_loss / total_train  #changed
        train_accuracy = correct_train / total_train  #changed

        model.eval()
        valid_loss = 0.0
        correct_valid = 0
        total_valid = 0

        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)  #changed
                outputs = model(images)  #changed
                loss = criterion(outputs, labels)  #changed
                valid_loss += loss.item() * images.size(0)  #changed
                _, predicted = torch.max(outputs, 1)  #changed
                correct_valid += (predicted == labels).sum().item()  #changed
                total_valid += labels.size(0)  #changed

        valid_loss = valid_loss / total_valid  #changed
        valid_accuracy = correct_valid / total_valid  #changed

        scheduler.step()  #changed

        # Log epoch results
        with open(log_path, mode='a', newline='') as f:  #changed
            writer = csv.writer(f)  #changed
            writer.writerow([epoch + 1, train_loss, train_accuracy, valid_loss, valid_accuracy])  #changed

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_accuracy:.4f}, Patience: {patience_counter}")  #changed

        # Early stopping based on validation loss
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), model_path)  #changed
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print('Early Stopping Triggered!')
            break

