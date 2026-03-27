function [train_images, train_labels, test_images, test_labels, valid_images, valid_labels] = ...
    load_mnist(mnist_path, train_ratio_subset, valid_ratio)
% train set
fid = fopen(fullfile(mnist_path, 'train-images-idx3-ubyte'),'r','ieee-be'); % big endian
A = fread(fid,4,'uint32');
num_images = A(2);
mdim = A(3);
ndim = A(4);

tmp_train_images = fread(fid,mdim*ndim*num_images,'uint8=>uint8');
tmp_train_images = reshape(tmp_train_images,[mdim, ndim,num_images]);
tmp_train_images = permute(tmp_train_images, [2 1 3]); 
train_images = zeros(mdim, ndim, 1, num_images);
train_images(:, :, 1, :) = tmp_train_images;
fclose(fid);


fid = fopen(fullfile(mnist_path, 'train-labels-idx1-ubyte'),'r','ieee-be');
A = fread(fid,2,'uint32');
num_images = A(2);

train_labels = fread(fid,num_images,'uint8=>uint8');
train_labels = categorical(train_labels);

if train_ratio_subset<1
    idx_train = [];
    unique_classes = unique(train_labels);
    % subsampling training set
    for i=1:length(unique_classes)
        c = unique_classes(i);
        idx_c = find(train_labels==c);
        tmp_idx_train = datasample(idx_c, ...
            round(train_ratio_subset*length(idx_c)),...
            'Replace', false);
        idx_train = cat(1, idx_train, tmp_idx_train);
    end
    train_images = train_images(:, :, :, idx_train);
    train_labels = train_labels(idx_train);
end

valid_images = [];
valid_labels = [];
if valid_ratio<1
    idx_valid = [];
    unique_classes = unique(train_labels);
    
    % keeping a fraction of the training set as validation set, equally
    % sampling each class (each digit)
    for i=1:length(unique_classes)
        c = unique_classes(i);
        idx_c = find(train_labels==c);
        tmp_idx_valid = datasample(idx_c, round(valid_ratio*length(idx_c)),...
            'Replace', false);
        idx_valid = cat(1, idx_valid, tmp_idx_valid);
    end
    valid_images = train_images(:, :, :, idx_valid);
    valid_labels = train_labels(idx_valid);
    
    idx_train = setdiff(1:length(train_images), idx_valid);
    train_images = train_images(:, :, :, idx_train);
    train_labels = train_labels(idx_train);
end
valid_images = valid_images/255;
train_images = train_images/255;
fclose(fid);
% test set
fid = fopen(fullfile(mnist_path, 't10k-images-idx3-ubyte'),'r','ieee-be');
A = fread(fid,4,'uint32');
num_images = A(2);
mdim = A(3);
ndim = A(4);

tmp_test_images = fread(fid,mdim*ndim*num_images,'uint8=>uint8');
tmp_test_images = reshape(tmp_test_images,[mdim, ndim,num_images]);
tmp_test_images = permute(tmp_test_images, [2 1 3]); 
test_images = zeros(mdim, ndim, 1, num_images);
test_images(:, :, 1, :) = tmp_test_images;
test_images = test_images/255;
fclose(fid);

fid = fopen(fullfile(mnist_path, 't10k-labels-idx1-ubyte'),'r','ieee-be');
A = fread(fid,2,'uint32');
num_images = A(2);

test_labels = fread(fid,num_images,'uint8=>uint8');
test_labels = categorical(test_labels);
fclose(fid);

end