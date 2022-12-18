Train setup:

import torch.optim as optim
from point_net_loss import PointNetLoss

EPOCHS = 50
LR = 0.0001
REG_WEIGHT = 0.


# manually set alpha weights
alpha = np.ones(len(CATEGORIES))
alpha[0:3] *= 0.25 # balance background classes
alpha[-1] *= 0.75  # balance clutter class

gamma = 1

optimizer = optim.Adam(seg_model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.01, 
                                              step_size_up=2000, cycle_momentum=False)
criterion = PointNetLoss(alpha=alpha, gamma=gamma, reg_weight=REG_WEIGHT).to(DEVICE)

*****************************************************************************************************
Test Results
Model 60: Test Loss: 0.4479 - Test Accuracy: 0.7449 - Test MCC: 0.7001

# different test loss and NUM_TEST_POINTS = 15000
Model 60: Test Loss: 0.7072 - Test Accuracy: 0.7416 - Test MCC: 0.6964- Test IOU: 0.5984