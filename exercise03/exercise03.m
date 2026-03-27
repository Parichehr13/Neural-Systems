clc,clear all,close all
E0=-65;%mV
Ea=-90; %mV
taum=30;%ms (30-50ms)
taut=10;%ms
taua=1000; %(300-1000ms)
r=10; %Mohm
C=taum/r;
dt=0.01;
tend = 800;
t=0:dt:tend;
L=length(t);
V=zeros(1,L);
Vt=zeros(1,L);
Pa=zeros(1,L);
dPa=0.1; %(0.03-0.2)
I=4; %nA
Vtl=-55;%mV
Vth=50;%mV
V(1)=-65; %mV
Pa(1)=0;
gamax=2/r; %gamax= (0.5-5)/r
Vt(1)=Vtl;
index=[];
g=1/r;

for k=1:L-1
    ga=gamax*Pa(k);
    geq=g+ga;
    E0tot=(g*E0+ga*Ea)/(geq);
    rtot=1/geq;
    Vinf=E0tot+rtot*I;
    tau=C*rtot;
    V(k+1)= (V(k)-Vinf)*exp(-dt/tau)+Vinf;
    Vt(k+1)= (Vt(k) - Vtl)*exp(-dt/taut) + Vtl;
    Pa(k+1)=Pa(k)*exp(-dt/taua);
    if V(k+1)> Vt(k+1)
        V(k+1)=E0; 
        Vt(k+1)=Vth;
        Pa(k+1)=Pa(k+1)+dPa*(1-Pa(k+1));
        index=[index k+1];
    end
end
T=t(index(end))-t(index(end-1));
f=1/T;

spikes = zeros(1,L);
spikes(index) = 1;

figure
plot(t,V);title('Adattamento');
hold on
plot(t,Vt,'r');
figure
plot(t,Pa);title('Pa');
figure
plot(t,spikes)
axis([0 t(end) 0 1.1])

