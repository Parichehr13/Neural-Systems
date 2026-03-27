clear
close all
clc

rng(1234)

I1 = -ones(4,4);  % creo una matrice 4*4 di -1
I1(2,:) = -I1(2,:);  % seconda riga positiva;
I2 = -ones(4,4);  % creo una matrice 4*4 di -1
I2(:,2) = -I2(:,2);  % seconda colonna positiva;
I3 = -ones(4,4)+2*eye(4);  % la diagonale pricipale di 1
I4([1:4],:) = I3([4:-1:1],:); % inverto le righe della figura 3

figure(1)
subplot(221)
imagesc(I1)
subplot(222)
imagesc(I2)
subplot(223)
imagesc(I3)
subplot(224)
imagesc(I4)

%pause

V1 = Da_matrice_a_vettore(I1)/4;   % normalizzo dividendo per 4
V2 = Da_matrice_a_vettore(I2)/4;
V3 = Da_matrice_a_vettore(I3)/4;
V4 = Da_matrice_a_vettore(I4)/4;

Y_corretto = eye(4);

W = Y_corretto*[V1 V2 V3 V4]';   

% prove senza rumore
Y1_senza = W*V1;
Y2_senza = W*V2;
Y3_senza = W*V3;
Y4_senza = W*V4;

% genero le immagini con rumore

sigma = 0.3;

I1_noise = I1 + sigma*randn(4,4);
I2_noise = I2 + sigma*randn(4,4);
I3_noise = I3 + sigma*randn(4,4);
I4_noise = I4 + sigma*randn(4,4);

figure(2)
subplot(221)
imagesc(I1_noise)
subplot(222)
imagesc(I2_noise)
subplot(223)
imagesc(I3_noise)
subplot(224)
imagesc(I4_noise)

pause

V1_noise = Da_matrice_a_vettore(I1_noise);
V2_noise = Da_matrice_a_vettore(I2_noise);
V3_noise = Da_matrice_a_vettore(I3_noise);
V4_noise = Da_matrice_a_vettore(I4_noise);

V1_noise = V1_noise/norm(V1_noise);
V2_noise = V2_noise/norm(V2_noise);
V3_noise = V3_noise/norm(V3_noise);
V4_noise = V4_noise/norm(V4_noise);

% prove con rumore
Y1_noise = W*V1_noise;
Y2_noise = W*V2_noise;
Y3_noise = W*V3_noise;
Y4_noise = W*V4_noise;

k = 20;

%prove con sigmoide
Y1_senza_sig = 1./(1+exp(-k*(Y1_senza - 0.5)));
Y2_senza_sig = 1./(1+exp(-k*(Y2_senza - 0.5)));
Y3_senza_sig = 1./(1+exp(-k*(Y3_senza - 0.5)));
Y4_senza_sig = 1./(1+exp(-k*(Y4_senza - 0.5)));

Y1_noise_sig = 1./(1+exp(-k*(Y1_noise - 0.5)));
Y2_noise_sig = 1./(1+exp(-k*(Y2_noise - 0.5)));
Y3_noise_sig = 1./(1+exp(-k*(Y3_noise - 0.5)));
Y4_noise_sig = 1./(1+exp(-k*(Y4_noise - 0.5)));

% plotto i risultati
% ingresso 1
Y1_senza
Y1_noise
Y1_senza_sig
Y1_noise_sig

I_output = Y1_noise_sig(1)*I1 + Y1_noise_sig(2)*I2 + Y1_noise_sig(3)*I3 + Y1_noise_sig(4)*I4; % immagine ricostruita
figure(3)
subplot(221)
imagesc(I1_noise)
subplot(222)
imagesc(I_output)

pause
clc

% ingresso 2
Y2_senza
Y2_noise
Y2_senza_sig
Y2_noise_sig

I_output = Y2_noise_sig(1)*I1 + Y2_noise_sig(2)*I2 + Y2_noise_sig(3)*I3 + Y2_noise_sig(4)*I4; % immagine ricostruita
figure(3)
subplot(221)
imagesc(I2_noise)
subplot(222)
imagesc(I_output)

pause
clc

% ingresso 3
Y3_senza
Y3_noise
Y3_senza_sig
Y3_noise_sig

I_output = Y3_noise_sig(1)*I1 + Y3_noise_sig(2)*I2 + Y3_noise_sig(3)*I3 + Y3_noise_sig(4)*I4; % immagine ricostruita
figure(3)
subplot(221)
imagesc(I3_noise)
subplot(222)
imagesc(I_output)

pause
clc

% ingresso 4
Y4_senza
Y4_noise
Y4_senza_sig
Y4_noise_sig

I_output = Y4_noise_sig(1)*I1 + Y4_noise_sig(2)*I2 + Y4_noise_sig(3)*I3 + Y4_noise_sig(4)*I4; % immagine ricostruita
figure(3)
subplot(221)
imagesc(I4_noise)
subplot(222)
imagesc(I_output)


function I = Da_vettore_a_matrice(V)
% funzioneche genera un vettore V scandendo la matrice A per righe
N = sqrt(size(V));  % numero di righe e colonne, matrice quadrata
I= [];
for j = 1:N,
    I(j,:) = V(1+(j-1)*N:j*N);
end
end

function V = Da_matrice_a_vettore(I)
% funzioneche genera un vettore V scandendo la matrice A per righe
N = size(I,1);  % numero di righe
V = [];
for j = 1:N,
    V = [V I(j,:)];
end
V = V';
end
