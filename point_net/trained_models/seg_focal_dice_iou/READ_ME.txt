Trained Models obtained from the best IOU
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
model 40: Test Loss: 0.7471 - Test Accuracy: 0.7227 - Test MCC: 0.6713 - Test IOU: 0.5743
model 44: Test Loss: 2.2411 - Test Accuracy: 0.7281 - Test MCC: 0.6801 - Test IOU: 0.5820
model 64: Test Loss: 0.6690 - Test Accuracy: 0.7461 - Test MCC: 0.6995 - Test IOU: 0.6042




