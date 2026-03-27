clc
close all
clear

N1 = 6;   % lato dell'immagine
I1 = -ones(N1,N1);  % creo una matrice 4*4 di -1
I1(1,2:6) = 1;  % righe 2-4 positive;
I1(2:5,4)= 1;  % creo una matrice 4*4 di -1
I2 = -ones(N1,N1);  % creo una matrice 4*4 di -1
I2(1:5,1) = 1;
I2(1:5,4)=1;
I2(3,1:4)=1;
I3 = -ones(N1,N1);
I3(1:5,2)=1; 
I3(5,2:5)=1;
I4 = -ones(N1,N1);  % creo una matrice 4*4 di -1
I4(1,3:6)=1;
I4(1:5,3)=1;
I4(5,3:6)=1;



figure(1)
subplot(221)
imagesc(I1)
subplot(222)
imagesc(I2)
subplot(223)
imagesc(I3)
subplot(224)
imagesc(I4)

pause
Y1= Da_matrice_a_vettore(I1);
Y2= Da_matrice_a_vettore(I2);
Y3= Da_matrice_a_vettore(I3);
Y4= Da_matrice_a_vettore(I4);

W = Y1*Y1'+Y2*Y2'+Y3*Y3'+Y4*Y4';

perc=0.3;
N=length(Y3);
Cambia = rand(N,1)>perc;  % ritorna 1 nella grande maggioranza dei casi
Cambia = (Cambia-0.5)*2;
Y=Y3.*Cambia;
%cerco la lista dei neuroni che possono commutare
Commutano=find(Y.*(W*Y)<0);
L=length(Commutano);
figure
X = Da_vettore_a_matrice(Y) ;
imagesc(X)
pause
while L > 0
    indice = ceil(rand(1,1)*L);
    Y(Commutano(indice))=-1*Y(Commutano(indice));
    Commutano=find(Y.*(W*Y)<0);
    L=length(Commutano);    
    X = Da_vettore_a_matrice(Y) ;
    imagesc(X)
    pause(0.05)
end




