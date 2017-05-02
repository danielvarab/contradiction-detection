# Novel approach
Our implementation of an NLI system, taking ques from Parikh et. al., IBM BiMBM and Steven Merity

## Changelog
* Added BatchNormalization for SUM type alignment and aggregation modules
* Added ReLu activations to the final three Dense layers
* Switched to rmsprop optimizer
* Added dropout all around with 0.2
* Added BatchNormalization on last 3 Dense
* Added L2 Reg on last 3 Dense with 4e-6
