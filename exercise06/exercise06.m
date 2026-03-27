clear all
close all
clc
rng(1234)
% inputs
X_no_noise = 2*round(rand(10,3)) - 1;
X_no_noise = X_no_noise/sqrt(10);

% inputs with noise
sigma = 0.1;
X_noise = X_no_noise + sigma*randn(10,3);
X_noise(:,1)= X_noise(:,1)/norm(X_noise(:,1));
X_noise(:,2)= X_noise(:,2)/norm(X_noise(:,2));
X_noise(:,3)= X_noise(:,3)/norm(X_noise(:,3));

Y = eye(3,3);
% network training (on inputs without noise)
W = Y*X_no_noise';

% network output (without noise)
Y1_no_noise = W*X_no_noise(:,1);
Y2_no_noise = W*X_no_noise(:,2);
Y3_no_noise = W*X_no_noise(:,3);

% network output (with noise)
Y1_noise = W*X_noise(:,1);
Y2_noise = W*X_noise(:,2);
Y3_noise = W*X_noise(:,3);

% sigmoid-activated network output - k = 10
k = 10;
Y1_sig_no_noise = 1./(1+exp(-k*(Y1_no_noise - 0.5)));
Y2_sig_no_noise = 1./(1+exp(-k*(Y2_no_noise - 0.5)));
Y3_sig_no_noise = 1./(1+exp(-k*(Y3_no_noise - 0.5)));

Y1_sig_noise = 1./(1+exp(-k*(Y1_noise - 0.5)));
Y2_sig_noise = 1./(1+exp(-k*(Y2_noise - 0.5)));
Y3_sig_noise = 1./(1+exp(-k*(Y3_noise - 0.5)));

% sigmoid-activated network output - k = 20
k = 20;
Y1_sig_no_noise_ = 1./(1+exp(-k*(Y1_no_noise - 0.5)));
Y2_sig_no_noise_ = 1./(1+exp(-k*(Y2_no_noise - 0.5)));
Y3_sig_no_noise_ = 1./(1+exp(-k*(Y3_no_noise - 0.5)));

Y1_sig_noise_ = 1./(1+exp(-k*(Y1_noise - 0.5)));
Y2_sig_noise_ = 1./(1+exp(-k*(Y2_noise - 0.5)));
Y3_sig_noise_ = 1./(1+exp(-k*(Y3_noise - 0.5)));

compare_neuron_values(Y1_no_noise, Y1_noise, ...
                      Y1_sig_no_noise, Y1_sig_noise, ...
                      Y1_sig_no_noise_, Y1_sig_noise_)
                  
compare_neuron_values(Y2_no_noise, Y2_noise, ...
                      Y2_sig_no_noise, Y2_sig_noise,...
                      Y2_sig_no_noise_, Y2_sig_noise_) 
                      
compare_neuron_values(Y3_no_noise, Y3_noise,... 
                      Y3_sig_no_noise, Y3_sig_noise,...
                      Y3_sig_no_noise_, Y3_sig_noise_) 
                  
function compare_neuron_values(Y_no_noise, Y_noise, Y_sig_no_noise, Y_sig_noise, Y_sig_no_noise_, Y_sig_noise_)
    % auxillary function to plot network output as bars
    figure()
    for i = 1:3
        subplot(1,3,i)
        bar([1:6], [Y_no_noise(i ,1), ...
                    Y_noise(i ,1), ...
                    Y_sig_no_noise(i ,1), ...
                    Y_sig_noise(i ,1),...
                    Y_sig_no_noise_(i ,1), ...
                    Y_sig_noise_(i ,1)],...
            0.25,'FaceColor', 'k')
        ylim([-1, 1])
        xticklabels({'no noise','noise','sigmoided (k=10) no noise',...
            'sigmoided (k=10) noise','sigmoided (k=20) no noise','sigmoided (k=20) noise',})
        xtickangle(30)
    end
    
end
