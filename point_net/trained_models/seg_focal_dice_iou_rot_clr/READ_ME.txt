Trained Models obtained from the best IOU (rotations on training data with 0.25 probability)
Set up:

EPOCHS = 100
LR = 0.0001

# manually set alpha weights
alpha = np.ones(len(CATEGORIES))
alpha[0:3] *= 0.25 # balance background classes
alpha[-1] *= 0.75  # balance clutter class

gamma = 1

optimizer = optim.Adam(seg_model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-6, max_lr=1e-3, 
                                              step_size_up=1000, cycle_momentum=False)
criterion = PointNetSegLoss(alpha=alpha, gamma=gamma, dice=True).to(DEVICE)

**************************************************************************************************

Test Results:
NUM_TEST_POINTS = 15000
model 27: Test Loss: 3.0481 - Test Accuracy: 0.6944 - Test MCC: 0.6400 - Test IOU: 0.5420
model 29: Test Loss: 68.1054 - Test Accuracy: 0.7026 - Test MCC: 0.6493 - Test IOU: 0.5515
model 30: Test Loss: 35.8008 - Test Accuracy: 0.6999 - Test MCC: 0.6481 - Test IOU: 0.5474




