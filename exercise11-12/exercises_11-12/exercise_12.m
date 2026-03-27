clear all;close all;clc
rng('default')
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
conv_nkernels = [16, 32];
p_drop = 0.25;

% INPUT LAYER
input = imageInputLayer([28 28 1],"Name","input",...
    "Normalization",'zscore');
% This way, that the input layer also keeps track of the mean and std on
% the training distribution. It uses them to standardize inputs (for each
% set).

% FIRST CONV + RELU + DROPOUT BLOCK
conv_hidden0 = convolution2dLayer([5,5],conv_nkernels(1), "Name","conv_hidden0");
% input_shape: (28, 28, 1); output_shape: (24, 24, 8)
act_hidden0 = reluLayer("Name","act_hidden0");
pool_hidden0 = averagePooling2dLayer([2,2],"Stride",[2,2],"Name","pool_hidden0"); 
dropout_hidden0 = dropoutLayer(p_drop, "Name","dropout_hidden0");

% SECOND CONV+RELU+DROPOUT BLOCK
conv_hidden1 = convolution2dLayer([5,5],conv_nkernels(2), "Name","conv_hidden1");
% input_shape: (12, 12, 8); output_shape: (8, 8, 16)
act_hidden1 = reluLayer("Name","act_hidden1");
pool_hidden1 = averagePooling2dLayer([2,2],"Stride",[2,2],"Name","pool_hidden1");
% input_shape: (8, 8, 16); output_shape: (4, 4, 16)
dropout_hidden1 = dropoutLayer(p_drop, "Name","dropout_hidden1");

% OUTPUT LAYER + SOFTMAX
fc_out = fullyConnectedLayer(10,"Name","fc_out");
% input_shape: (256,); output_shape: (10,)
act_out=softmaxLayer("Name","act_out");

% CROSS-ENTROPY LOSS FUNCTION
loss = classificationLayer("Name","loss");

layers = [
    input

    conv_hidden0
    act_hidden0
    pool_hidden0
    dropout_hidden0
    
    conv_hidden1
    act_hidden1
    pool_hidden1
    dropout_hidden1
    
    fc_out
    act_out
    
    loss];

% NN visualization
analyzeNetwork(layers)

% 18378 parameters vs. 109376 of exercize 13
%% STAGE III: MODEL OPTIMIZATION
% Points 5 and 6
checkpoint_dir = './checkpoints_cnn_mnist';
if ~isdir(checkpoint_dir)
    mkdir(checkpoint_dir)
end
% in this directory 'checkpoints' (i.e., training and optimization
% parameters) at each epoch will be saved, with unique filenames

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

train_loss = info.TrainingLoss;
valid_loss = info.ValidationLoss;

train_acc = info.TrainingAccuracy;
valid_acc = info.ValidationAccuracy;

save(fullfile(checkpoint_dir, 'tracked_metrics.mat'), ...
    'train_loss', 'valid_loss','train_acc','valid_acc')

clear info net train_loss valid_loss train_acc valid_acc
%%
checkpoint_dir = './checkpoints_cnn_mnist';
load(fullfile(checkpoint_dir, 'tracked_metrics.mat'))
% Loss and evaluation metric visualization 
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
checkpoint_dir = './checkpoints_cnn_mnist';
% Select and load the best model on the validation set (e.g., for the validation loss)
target_fname = 'net_checkpoint__7301__2021_12_17__18_19_19.mat';
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
% 97.51% vs. 94.98% with NN with less parameters (more parsimonious NN)
%% OPTIONAL: MODEL TESTING ON YOUR OWN DIGITS
% Point 8
close all
fname = 'sample_digits_v1.jpg';
fpath = fullfile('./your_own_digits',fname);
[my_digit] = load_my_digit(fpath);
% Evaluate the CNN on the selected digit
[y_pred_my_digit, probs_my_digit] = classify(net,my_digit);
[probs_sorted, idx] = sort(probs_my_digit); % sort probabilities

g=figure('Units','normalized','Position',[0 0 0.5 0.25]);
subplot(121)
imagesc(my_digit); colormap gray
subplot(122)
barh(probs_sorted, 'FaceColor', 'magenta')
xlabel('probability')
ylabel('class')
xlim([0, 1])

classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];
yticklabels(classes(idx))
title('Predicted probabilities')
%% OPTIONAL: MODEL VISUALIZATION OF THE 1st CONV. LAYER
% Point 9
close all
Wconv0 = net.Layers(2).Weights;
Wconv0 = squeeze(Wconv0);

g=figure('Units','normalized','Position',[0 0 .5 .5]);
for i=1:k
    subplot(4,4,i)
    imagesc(Wconv0(:,:,i)); colormap gray
end
sgtitle("1st conv. layer learned kernels")

act_correct = activations(net,my_digit,'conv_hidden0');
[m,n,c,k] = size(act_correct);
fprintf('Size of 1st activations: (%d, %d, %d, %d) \n', [m,n,c,k])
act_correct = squeeze(act_correct);
[m,n,k] = size(act_correct);
g=figure('Units','normalized','Position',[0 0 .5 .5]);
for i=1:k
    subplot(4,4,i)
    imagesc(act_correct(:,:,i)); colormap gray
end

s = strcat("1st conv. layer neuron outputs");
sgtitle(s)
% It is worth noticing, that the neurons of this first conv. layer
% respond to different orientation in the input image (e.g., vertical or
% horizontal or inclined orientations). Filtering the input image with 
% filters enhancing particular orientations helped to discriminate between
% the different digits. 