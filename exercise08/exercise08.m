clc
close all
load imdemos
%   box              128x128            16384  logical              
%   circles          256x256            65536  logical              
%   circuit          128x128            16384  uint8                
%   circuit4         256x256            65536  uint8                
%   coins            128x128            16384  uint8                
%   coins2           256x256            65536  uint8                
%   dots             128x128            16384  logical              
%   eight            256x256            65536  uint8                
%   glass            128x128            16384  uint8                
%   glass2           256x256            65536  uint8                
%   liftbody128      128x128            16384  uint8                
%   liftbody256      256x256            65536  uint8                
%   moon             128x128            16384  uint8                
%   pepper           128x128            16384  uint8                
%   pout             128x128            16384  uint8                
%   quarter          128x128            16384  uint8                
%   rice             128x128            16384  uint8                
%   rice2            128x128            16384  uint8                
%   rice3            256x256            65536  uint8                
%   saturn           128x128            16384  uint8                
%   saturn2          256x256            65536  uint8                
%   tire             128x128            16384  uint8                
%   trees            128x128            16384  uint8                
%   vertigo          128x128            16384  uint8                
%   vertigo2         256x256            65536  uint8   
XX1 = im2bw(saturn,0.5);
XX2=im2bw(vertigo,0.5);
XX3=im2bw(coins,0.5);
XX1=(XX1-0.5)*2;
XX2=(XX2-0.5)*2;
XX3=(XX3-0.5)*2;

N=size(XX1);
X1=XX1(1:2:N,1:2:N);
X2=XX2(1:2:N,1:2:N);
X3=XX3(1:2:N,1:2:N);
save immagini X1 X2 X3
clear                      % pulisco per evitare troppa occupazione di memoria
load immagini

figure
imagesc(X1)
figure
imagesc(X2)
figure
imagesc(X3)

pause

Y1= Da_matrice_a_vettore(X1);
Y2= Da_matrice_a_vettore(X2);
Y3= Da_matrice_a_vettore(X3);
 

W = Y1*Y1'+Y2*Y2'+Y3*Y3';

Y = Y1;  % scelgo una immagine
perc=0.05;
N=length(Y1);
Cambia = find(rand(N,1)<perc);  % verificato solo in una bassa percentuale di casi
Y(Cambia)=-Y(Cambia);
%cerco la lista dei neuroni che possono commutare
Commutano=find(Y.*(W*Y)<0);
L=length(Commutano);
figure
X = Da_vettore_a_matrice(Y) ;
imagesc(X)
pause
while L > 0
    indice = ceil(rand(1,1)*L);    %% randi(L,1);
    Y(Commutano(indice))=-1*Y(Commutano(indice));
    Commutano=find(Y.*(W*Y)<0);
    L=length(Commutano);    
    X = Da_vettore_a_matrice(Y) ;
    imagesc(X)
    pause(0.05)
end




