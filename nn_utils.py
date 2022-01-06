import torch
from torch import nn, optim
import json

def nn_train(model, X_train, Y_train, X_valid, Y_valid, epochs=100, lr = 0.1, validate_every = 100, debug=False):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.parameters(), lr, weight_decay=0.6e-4)
    # optimizer = optim.SGD(model.parameters(), lr, momentum=0.999)
    model.to(device)

    train_losses, validation_losses= [], []
    train_losses = []

    # with open('evaluation.json', 'r') as file:
    #     best_evaluation = json.load(file)

    best_evaluation = {'validation_loss':10, 'validation_accuracy':0}

    for epoch in range(epochs):
        inputs, labels = X_train.to(device), Y_train.to(device)
        
        optimizer.zero_grad()

        log_ps = model(inputs.float())
        
        loss = criterion(log_ps, labels.long())

        loss.backward()
        optimizer.step()
        

        if epoch % validate_every == 0:
            train_loss = loss.item() / len(labels)
            ps = torch.exp(log_ps.float())
            predictions = ps.argmax(dim=1).to('cpu')
            train_accuracy = nn_accuracy(predictions, Y_train)
            train_losses.append(train_loss)
            
            with torch.no_grad():
                model.eval()
                inputs, labels = X_valid.to(device), Y_valid.to(device)
                log_ps = model(inputs.float())
                loss = criterion(log_ps, labels.long())
                valid_loss = loss.item()/len(labels)
                validation_losses.append(valid_loss)
                ps = torch.exp(log_ps.float())
                predictions = ps.argmax(dim=1).to('cpu')
                valid_accuracy = nn_accuracy(predictions, Y_valid)

                if valid_loss < best_evaluation['validation_loss']:
                #     print(f'loss = {valid_loss}, accuracy = {valid_accuracy*100}')
                    best_evaluation['validation_loss'] = valid_loss
                    best_evaluation['validation_accuracy'] = valid_accuracy*100
                #     torch.save(model.state_dict(), 'model.pth')

            
            model.train()

            if debug:
                print(f'[INFO] Epoch: {epoch+1}/{epochs}\n',
                    f"Training Loss: {train_loss:0.3f}, training accuracy {train_accuracy*100:0.3f}%",
                    f"validation Loss: {valid_loss:0.3f}, validation accuracy {valid_accuracy*100:0.3f}%")
    
    # with open("evaluation.json", "w") as outfile:
    #     json.dump(best_evaluation, outfile)

    return train_losses, validation_losses

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
    

    


