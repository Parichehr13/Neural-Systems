
clear
clc
close all
N=180;
sigex=2;                   %       2.;                              
sigin=24;                   %       6;                                                
Lex0=5;          %2.4  %6                                                               
Lin0=2;        %1.4 %3
phi=12;      
pend=0.6;   %0.6
soglia = 6;
k=[1:1:N];
for i = 1:N,
    distanza=abs(k-i);
    index = find(distanza > 90);
    distanza(index) = 180 - distanza(index);
    D2=distanza.^2;% matrice delle distanze al quadrato FORMA CIRCOLARE
    L(i,:)=-Lin0*exp(-D2/2/sigin/sigin); 
    L(i,i)=Lex0;    % non riceve sinapsi da se stesso
end 

stimolo = 2 % 1 stimoli rettangolare, 2 due stimoli ravvicinati

switch stimolo
    case 1
Ix = 6*ones(1,180);
Ix(90:110)=14;
    case 2
        Ix = 5*ones(1,180);%6*ones(1,180);
        Ix= Ix + 10*exp(-([1:180]-100).^2/2/10/10)+10*exp(-([1:180]-120).^2/2/10/10);%Ix + 8*exp(-([1:180]-85).^2/2/10/10)+8*exp(-([1:180]-115).^2/2/10/10);
end
        
tau=3;
iter=1000;%600; %1000     % lunghezza simulazione     
dt=0.1;   %0.1        % passo campionamento
t=[0:iter]*dt;      % asse dei tempi
LL=length(t);
x=zeros(N,iter);
for k=1:LL-1,
     x(:,k+1)=x(:,k)+dt*((1/tau)*(-x(:,k)+1./(1+exp(-(Ix'+L*x(:,k)-soglia)*pend))));   
     plot([1:N],x(:,k+1),'r','linewidth',2)
     axis([0 200 0 1])
     title(['passo' num2str(k)])
     pause(0.01)
end
feedforward = 1./(1+exp(-(Ix'-soglia)*pend));  % contiene l'uscita dovuta al solo input
figure
plot([1:N],x(:,k+1),'r',[1:N],feedforward,'b','linewidth',2)
%--------------------------------------------------------------------------

