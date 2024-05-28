# The Experiments For 《Generalization Bound and New Algorithm for Clean-Label Backdoor Attack》

### What we offer here is the code that find backdoor poison on CIFAR-10, use the network VGG16 and Resnet. 

#### 1 'Backdoor_attack_method.py' is code of the algorithm 1 in paper, and the effect of poisoning was also tested. 
##### 1.1 In its line 29, 'bud' is the budget of poison, you can change it from 8/255 to 32/255, or some others you like; in line 30, 'num' is the number of poisoned samples in training set, you can change it from 300 to 500, not more than 5000; in line 31, network=1 means use ResNet, network=2 means use VGG; in line 32, lp is target label, not more than 9.
##### 1.2 For the purpose of saving time, we load the trained network F_1 instead of training it from scratch, if you want to train F_1, use 'Small_vgg_9layers.py'.
#### 2 'Small_vgg_9layers.py' is code to train F_1.
##### 2.1 In its line 188, it is the path to save F_1, if you change it, you also need to modify line 124 of 'Backdoor_attack_method.py'. 
#### 3 See 'Requiresments' for experiment environment.
#### 4 Every 'py' file runs directly on GPU.
