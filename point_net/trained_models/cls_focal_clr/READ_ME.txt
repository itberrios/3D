Train set up


EPOCHS = 25
LR = 0.0001
REG_WEIGHT = 0.001 

# manually downweight the high frequency classes
alpha = np.ones(NUM_CLASSES)
alpha[0] = 0.5  # airplane
alpha[4] = 0.5  # chair
alpha[-1] = 0.5 # table

gamma = 1

optimizer = optim.Adam(classifier.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.01, 
                                              step_size_up=2000, cycle_momentum=False)
criterion = PointNetLoss(alpha=alpha, gamma=gamma, reg_weight=REG_WEIGHT).to(DEVICE)

*********************************************************************************************

TEST RESULTS:
Model 38 -- Test Loss: 0.2801 - Test Accuracy: 0.7983 - Test MCC: 0.0793
Model 37 -- Test Loss: 0.2728 - Test Accuracy: 0.8382 - Test MCC: 0.0665
Model 35 -- Test Loss: 0.2316 - Test Accuracy: 0.8517 - Test MCC: 0.0819
Model 27 -- Test Loss: 0.3462 - Test Accuracy: 0.8135 - Test MCC: 0.0607
