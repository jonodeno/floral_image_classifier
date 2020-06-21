import torch
import argparse
import os
from model import Classifier
from torch import nn, optim
import utilities as util

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', metavar='data_directory', type=str)
parser.add_argument('--save_dir')
parser.add_argument('--arch')
parser.add_argument('--learning_rate', type=int)
parser.add_argument('--hidden_units', nargs='+')
parser.add_argument('--epochs', type=int)
parser.add_argument('--gpu', action='store_true')

if __name__ == '__main__':
    args = parser.parse_args()
    args_dict = {
        "hidden_layers": args.hidden_units if args.hidden_units else [2000, 1500, 750],
        "arch": args.arch if args.arch else "vgg16",
    }
    
    learnrate = args.learning_rate if args.learning_rate else 0.0005
    device = "cuda" if args.gpu else "cpu"
    
    model = Classifier(**args_dict).model
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learnrate)
    
    # Train the model 
    epochs = args.epochs if args.epochs else 5
    train_dir=args.data_dir+"/train"
    valid_dir=args.data_dir+"/valid"
    
    train_data_tuple = util.generate_loader(train_dir,False,True)
    train_data = train_data_tuple[0]
    train_dataset = train_data_tuple[1]
    valid_data = util.generate_loader(valid_dir)[0]
    
    model.to(device)
    for epoch in range(epochs):
        print(f"*******************STARTING EPOCH {epoch}********************************")
        for inputs, labels in train_data:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print("Training loss: ", loss.item())

            accuracy = 0
            with torch.no_grad():
                model.eval()
                valid_loss = 0
                for inputs, labels in valid_data:
                    inputs, labels = inputs.to(device), labels.to(device)
                    log_ps = model.forward(inputs)

                    loss = criterion(log_ps, labels)
                    valid_loss+=loss.item()

                    ps = torch.exp(log_ps)
                    top_p, top_k = ps.topk(1,dim=1)
                    equals = top_k == labels.view(*top_k.shape)
                    accuracy+=torch.mean(equals.type(torch.FloatTensor)).item()
                print(f"Validation loss: {valid_loss/len(valid_data)}")
                print(f"Accuracy: {accuracy/len(valid_data)}")
                model.train()
    print("Complete!!")
    model.class_to_idx = train_dataset.class_to_idx
    checkpoint = {
        'classifier': model.classifier,
        'state_dict': model.classifier.state_dict(),
        'no_epochs': epochs,
        'optimizer': optimizer.state_dict(),
        'class_to_idx':model.class_to_idx,
        'arch': model.arch
    }
    if args.save_dir:
        os.mkdir(args.save_dir)
        path = args.save_dir + '/checkpoint.pth'
        torch.save(checkpoint, path)
    else:
        torch.save(checkpoint, 'checkpoint.pth')