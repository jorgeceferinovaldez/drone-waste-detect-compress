

# Función para entrenar el modelo
def train_batch(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    #for original_image, _ in train_loader:
    for original_image in train_loader:
        original_image = original_image.to(device)
        
        optimizer.zero_grad()
        outputs = model(original_image)
        
        loss = criterion(outputs, original_image)
        
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    train_loss /= len(train_loader) # Dividir por el número de batches para obtener la media
    return train_loss