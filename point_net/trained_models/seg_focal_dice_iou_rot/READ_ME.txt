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
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.01, 
                                              step_size_up=2000, cycle_momentum=False)
criterion = PointNetSegLoss(alpha=alpha, gamma=gamma, dice=True).to(DEVICE)

**************************************************************************************************

Test Results:
NUM_TEST_POINTS = 15000
model 67: Test Loss: 0.6808 - Test Accuracy: 0.7337 - Test MCC: 0.6866 - Test IOU: 0.5886
model 68: Test Loss: 0.6084 - Test Accuracy: 0.7515 - Test MCC: 0.7077 - Test IOU: 0.6098 **best**
model 89: Test Loss: 0.8167 - Test Accuracy: 0.7436 - Test MCC: 0.6976 - Test IOU: 0.6004




