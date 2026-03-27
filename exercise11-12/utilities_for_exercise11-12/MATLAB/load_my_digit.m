function [my_digit] = load_my_digit(fpath)
my_digits = imread(fpath);
% Image (containing many digits wrote on white papers) pre-processing
my_digits = rgb2gray(my_digits); % converting to gray-scale
g=figure('Units','normalized','Position',[0 0 .5 .5]);
imagesc(my_digits); colormap gray
title('Select two points: top-left and bottom-right corner of the area containing a digit of interest...')
% Select the top-left and bottom-right corner of the area containing a digit
[x,y] = ginput(2);
close(g)

my_digits = 255-my_digits; % inverting dynamic (higher values: foreground, low values: background)
my_digits(my_digits<128)=0; % roughly bringing to 0 background
my_digits = double(my_digits)/255; % normalization between 0-1

% Digit pre-processing
start_row = int16(y(1));
stop_row = int16(y(2));
start_col = int16(x(1));
stop_col = int16(x(2));
% cropping the selected digit
my_digit = my_digits(start_row:stop_row, start_col:stop_col);
% Resizing digits according to the adopted dataset
my_digit = imresize(my_digit, [28, 28]);
my_digit(my_digit<0)=0;
my_digit(my_digit>1)=1;
% Forcing one channel (gray-scale image)
my_digit = reshape(my_digit, [28, 28, 1]);
end

