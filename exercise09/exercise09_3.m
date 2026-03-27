clc
close all

X1 = zeros(30,30);
X1(10:20,10)=1;
X1(20,10:18)=1;
X2 = zeros(30,30);
diag=[5:23];
for j = 1:length(diag);
    X2(diag(j),diag(j))=1;
end
X3 = zeros(30,30);
X3(14:16,15:20)=1;
figure(1)
subplot(221)
imagesc(X1)
subplot(222)
imagesc(X2)
subplot(223)
imagesc(X3)
pause
Y1= Da_matrice_a_vettore(X1);
Y2= Da_matrice_a_vettore(X2);
Y3= Da_matrice_a_vettore(X3);

a=0.02;
teta =8;
W = (Y1-a)*(Y1-a)'+(Y2-a)*(Y2-a)'+(Y3-a)*(Y3-a)';

perc=0.15;
N=length(Y1);
Y=Y1;
index = find(rand(N,1)<perc);  % ritorna gli indici dei neuroni da cambiare
Y(index) = 1 - Y(index);
%cerco la lista dei neuroni che possono commutare
Commutano=find((Y-0.5).*((W*Y)-teta)<0);
L=length(Commutano);
disp(L)
figure
X = Da_vettore_a_matrice(Y) ;
imagesc(X)
pause
while L > 0
    indice = ceil(rand(1,1)*L);
    Y(Commutano(indice))=1-Y(Commutano(indice));
    Commutano=find((Y-0.5).*((W*Y)-teta)<0);
    L=length(Commutano);  
    X = Da_vettore_a_matrice(Y) ;
    imagesc(X)
    pause(0.1)
end




