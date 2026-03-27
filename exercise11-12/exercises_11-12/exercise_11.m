clear all;close all;clc
rng(1234)
%% STAGE I: DATA LOADING AND PRE-PROCESSING (IF NECESSARY)
% Point 1
mnist_path = './MNIST';
train_ratio_subset = 0.2; % random sample 20% of the total MNIST training set (0.2*60K examples)
valid_ratio = 0.2; % random 20% of the training set held out as validation set
[x_train, y_train, x_test, y_test, x_valid, y_valid] = ...
    load_mnist(mnist_path, train_ratio_subset, valid_ratio);

% Random display of training images
figure()
idx = randperm(size(x_train, 4),9);
for i = 1:numel(idx)
    target_idx = idx(i);
    subplot(3,3,i)    
    imagesc(x_train(:,:,:,target_idx)); colormap gray
    title(strcat("Digit: ", char(y_train(target_idx))))
    hold on;
end
sgtitle('Examples of input images')
% Histogram of labels for all sets (train, valid and test)
figure()
subplot(311)
histogram(y_train, 'BarWidth', 0.5)
ylim([0, 1200])
title('Training set')
ylabel('# examples')
subplot(312)
histogram(y_valid, 'BarWidth', 0.5)
ylim([0, 1200])
title('Validation set')
ylabel('# examples')
subplot(313)
histogram(y_test, 'BarWidth', 0.5)
ylim([0, 1200])
title('Test set')
ylabel('# examples')
%% STAGE II: MODEL DESIGN
% Points 3 and 4
% Definition of some architectural hyper-parameters
hidden_neurons_per_layer = [128, 64];
p_drop = 0.25; 

% INPUT LAYER
input = imageInputLayer([28 28 1],"Name","input",...
    "Normalization",'zscore');
% This way, that the input layer also keeps track of the mean and std on
% the training distribution. It uses them to standardize inputs (for each
% set).

% FIRST FC + RELU + DROPOUT BLOCK
fc_hidden0 = fullyConnectedLayer(hidden_neurons_per_layer(1), "Name","fc_hidden0");
% input_shape: (784,); output_shape: (128,)
act_hidden0 = reluLayer("Name","act_hidden0");
dropout_hidden0 = dropoutLayer(p_drop, "Name","dropout_hidden0");

% SECOND FC + RELU + DROPOUT BLOCK
fc_hidden1 = fullyConnectedLayer(hidden_neurons_per_layer(2), "Name","fc_hidden1");
% input_shape: (128,); output_shape: (64,)
act_hidden1 = reluLayer("Name","act_hidden1");
dropout_hidden1 = dropoutLayer(p_drop, "Name","dropout_hidden1");

% OUTPUT LAYER + SOFTMAX
fc_out = fullyConnectedLayer(10,"Name","fc_out");
% input_shape: (64,); output_shape: (10,)
act_out=softmaxLayer("Name","act_out");

% LOSS FUNCTION
loss = classificationLayer("Name","loss");

layers = [
    input

    fc_hidden0
    act_hidden0
    dropout_hidden0
    
    fc_hidden1
    act_hidden1
    dropout_hidden1

    fc_out
    act_out
    
    loss];

% NN visualization 
analyzeNetwork(layers)
%% STAGE III: MODEL OPTIMIZATION 
% Points 5 and 6
checkpoint_dir = './checkpoints_nn_mnist';
if ~isdir(checkpoint_dir)
    mkdir(checkpoint_dir)
end
% in this directory 'checkpoints' (i.e., NN trainable parameters)
% at each epoch will be saved, with unique filenames

% Definition of some optimization hyper-parameters
optimizer = 'sgdm'; % stochastic gradient descent with momentum 
lr = 0.001;
mini_bs = 64;
max_epochs = 50;
l2_reg_par = 0.0001;
momentum_par = 0.9;

n_batches_per_epoch = floor(length(y_train)/mini_bs);
options = trainingOptions(optimizer, ...
    'InitialLearnRate',lr, ...
    'L2Regularization', l2_reg_par,...
    'Momentum', momentum_par,...
    'MaxEpochs',max_epochs, ...
    'MiniBatchSize',mini_bs, ...
    'Shuffle','every-epoch',...
    'VerboseFrequency',n_batches_per_epoch, ...
    'ValidationData',{x_valid,y_valid}, ...
    'ValidationFrequency',n_batches_per_epoch, ...
    'CheckpointPath', checkpoint_dir,...
    'ExecutionEnvironment', 'cpu');

[net, info] = trainNetwork(x_train,y_train,layers,options);
% If you have MATLAB R2021b, you can use also 'OutputNetwork', 'best-validation-loss' 
% in trainingOptions to return the best model on the validation loss after training, without the need of
% saving the model after each epoch to perform early stopping,
% i.e., the "net" variable contains the best parameters on the validation set.

% Loss and evaluation metric visualization
train_loss = info.TrainingLoss;
valid_loss = info.ValidationLoss;

train_acc = info.TrainingAccuracy;
valid_acc = info.ValidationAccuracy;

save(fullfile(checkpoint_dir, 'tracked_metrics.mat'), ...
    'train_loss', 'valid_loss','train_acc','valid_acc')

clear info net train_loss valid_loss train_acc valid_acc
%%
checkpoint_dir = './checkpoints_nn_mnist';
load(fullfile(checkpoint_dir, 'tracked_metrics.mat'))
mini_bs = 64;
n_batches_per_epoch = floor(length(y_train)/mini_bs);

iterations = 1:length(train_acc);

figure('Units','normalized','Position',[0 0 .5 1])
subplot(211)
plot(iterations, train_loss,'b')
hold on
plot(iterations, valid_loss,'ro', 'MarkerFaceColor', 'r')
hold on
ylabel('loss')
xlabel('iterations')
legend({'train', 'valid'})

subplot(212)
plot(iterations, train_acc,'b')
hold on
plot(iterations, valid_acc,'ro', 'MarkerFaceColor', 'r')
ylabel('accuracy')
xlabel('iterations')
legend({'train', 'valid'})

[~, target_iteration_earlystop] = min(valid_loss);
fprintf('Minimum validation loss at epoch: %d;iteration: %d \n',...
    target_iteration_earlystop/n_batches_per_epoch,target_iteration_earlystop) 
%% STAGE IV: MODEL TESTING 
% Point 7
checkpoint_dir = './checkpoints_nn_mnist';
% Select and load the best model on the validation set (e.g., for the validation loss)
target_fname = 'net_checkpoint__7450__2021_12_17__18_04_43.mat'; % change it depending on your model filename
load(fullfile(checkpoint_dir,target_fname));
% Evaluate on the test set
[y_test_pred, probs_test] = classify(net,x_test);
fprintf('Test set accuracy: %1.4f\n',...
    mean(y_test_pred==y_test))
% Visualize 9 test examples that were misclassified
idx_wrong = find(y_test_pred~=y_test);
num_wrong_images = length(idx_wrong);
figure()
idx = randperm(num_wrong_images,9);
for i = 1:numel(idx)
    target_idx = idx_wrong(idx(i));
    subplot(3,3,i)    
    imagesc(x_test(:,:,:,target_idx)); colormap gray
    
    s = strcat("Digit: ", char(y_test(target_idx)), " (pred: ", char(y_test_pred(target_idx)), ")");
    title(s)
    hold on;
end
sgtitle('Examples of misclassified input images')