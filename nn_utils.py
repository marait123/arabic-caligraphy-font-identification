import torch
from torch import nn, optim

def nn_train(model, X_train, Y_train, X_valid, Y_valid, X_test, Y_test, epochs=100, lr = 0.1):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.parameters(), lr)
    
    model.to(device)

    train_losses, validation_losses, test_losses = [], [], []
    train_losses = []
    for epoch in range(epochs):
        inputs, labels = X_train.to(device), Y_train.to(device)
        
        optimizer.zero_grad()

        log_ps = model(inputs.float())
        
        loss = criterion(log_ps, labels.long())

        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            train_loss = loss.item() / len(labels)
            train_losses.append(train_loss)
            
            with torch.no_grad():
                model.eval()
                inputs, labels = X_valid.to(device), Y_valid.to(device)
                log_ps = model(inputs.float())
                loss = criterion(log_ps, labels.long())
                valid_loss = loss.item()/len(labels)
                validation_losses.append(valid_loss)

                inputs, labels = X_test.to(device), Y_test.to(device)
                log_ps = model(inputs.float())
                loss = criterion(log_ps, labels.long())
                test_loss = loss.item()/len(labels)
                test_losses.append(test_loss)
            
            model.train()

            print(f'Epoch: {epoch+1}/{epochs}',
                f"Training Loss: {train_loss}",
                f"validation Loss: {valid_loss}",
                f"Test Loss: {test_loss}")

    return train_losses, validation_losses, test_losses

def nn_predict(model, features):
    with torch.no_grad():
        model.eval()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        features = features.to(device)
        logps = model(features.float())
        ps = torch.exp(logps.float())
        predictions = ps.argmax(dim=1)
    
    model.train()
    return predictions.to('cpu')

def nn_accuracy(predictions, labels):
    correct = (predictions == labels).sum().item()
    return correct/len(labels)
    

    


