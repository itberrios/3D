TRAIN SETUP

EPOCHS = 10
LR = 0.01
REG_WEIGHT = 0.001 

optimizer = optim.Adam(classifier.parameters(), lr=LR)
criterion = PointNetLoss(reg_weight=REG_WEIGHT).to(DEVICE)

******************************************************

RESULTS

Loss: 
Cross Entropy with Transformation Matrix regularization

Test: decent accuracy, terrible MCC

