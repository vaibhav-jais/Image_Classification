from model import Network
import torch.optim as optim

net = Network()

# Loss function and optomizer
loss_function = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, nesterov = True)
optimizer = optim.Adam(net.parameters(), lr= 0.001)

# Model training
# number of epochs to train the model
num_epochs = 25


for epoch in range(1, num_epochs+1):
    time_starts = time.time()
    running_training_loss = []
    running_validation_loss = []

    # Training Phase
    for items in trainloader:
        training_images, training_labels = items

        # Forward pass
        outputs = net(training_images)                              # generating predictions on training images
        train_loss = loss_function(outputs, training_labels)                        # calculating training loss

        # Backward and optimize
        optimizer.zero_grad()
        train_loss.backward()     #  gradients are computed using backward() method.
        optimizer.step()          # step() method, that updates the parameters by performing single optimization step.

        running_training_loss.append(train_loss.item())
    training_loss = np.average(running_training_loss)

    # Validation Phase
    correct_count, total_count = 0, 0
    for items in loader:
        net.eval()
        validation_images, validation_labels = items

        validation_outputs = net(validation_images)                       # generating predictions on training images
        validation_loss = loss_function(validation_outputs, validation_labels)         # calculating validation loss
        running_validation_loss.append(validation_loss.item())

        ps = torch.exp(validation_outputs)
        probab = list(validation_outputs.cpu()[0])
        pred_label = probab.index(max(probab))
        #prediction.append(pred_label)
        true_label = validation_labels.cpu()[i]
        if(true_label == pred_label):
            correct_count += 1
        total_count += 1
        validation_accuracy = correct_count/ total_count
    validation_loss = np.average(running_validation_loss)

    #print("Epoch [{}], train_loss: {:.4f}".format(train_loss_value))
    print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, training_loss, validation_loss, validation_accuracy))
    time_ends = time.time()
    time_elapsed = time_ends - time_starts
    print("Time_elapsed for Epoch [{}] : [{}] s".format(epoch,time_elapsed))
