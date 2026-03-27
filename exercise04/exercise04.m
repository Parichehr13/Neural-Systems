clc,clear all,close all
E0=-65;%mV
taum=30;%ms (10-30ms)
taut=5;%ms
taus1=10; % ms
taus2=10;
r=10; %Mohm
C=taum/r;
dt=0.01;
t=0:dt:150;
L=length(t);
V1=zeros(1,L);
Vt1=zeros(1,L);
Ps1=zeros(1,L);
V2=zeros(1,L);
Vt2=zeros(1,L);
Ps2=zeros(1,L);

Vtl=-55;%mV
Vth=50;%mV
% parametri sinapsi
dPs1=0.6; %(0.03-0.6)
dPs2=0.2;
Es2=0; %mV
Es1=-70; %mV
gsmax1=5/r; %gsmax= (0.5-5)/r
gsmax2=5/r; %gsmax= (0.5-5)/r
%
I1=4; %nA
I2=4;
V1(1)=-65; %mV
V2(1)=-65; %mV
Vt1(1)=Vtl; %mV
Vt2(1)=Vtl; %mV
Ps1(1)=0;
Ps2(1)=0;


gs1=gsmax1*Ps1(1);
gs2=gsmax2*Ps2(1);

index1=[];
index2=[];
g=1/r;

%1 eccitatorio, 2  inibitorio
for k=1:L-1
    geq1=g+gs1;
    geq2=g+gs2;
    E0tot1=(g*E0+gs1*Es1)/(geq1);
    E0tot2=(g*E0+gs2*Es2)/(geq2);
    rtot1=1/geq1;
    rtot2=1/geq2;
    Vinf1=E0tot1+rtot1*I1;
    Vinf2=E0tot2+rtot2*I2;
    tau1=C*rtot1;
    tau2=C*rtot2;
    V1(k+1)= (V1(k)-Vinf1)*exp(-dt/tau1)+Vinf1;
    Vt1(k+1)= (Vt1(k) - Vtl)*exp(-dt/taut) + Vtl;
    Ps1(k+1)=Ps1(k)*exp(-dt/taus1);
    gs1=gsmax1*Ps1(k+1);
    V2(k+1)= (V2(k)-Vinf2)*exp(-dt/tau2)+Vinf2;
    Vt2(k+1)= (Vt2(k) - Vtl)*exp(-dt/taut) + Vtl;
    Ps2(k+1)=Ps2(k)*exp(-dt/taus2);
    gs2=gsmax2*Ps2(k+1);
    if V1(k+1)> Vt1(k+1)
        V1(k+1)=E0; 
        Vt1(k+1)=Vth;
        Ps2(k+1)=Ps2(k+1)+dPs2*(1-Ps2(k+1));
        index1=[index1 k+1];
    end
     if V2(k+1)> Vt2(k+1)
        V2(k+1)=E0; 
        Vt2(k+1)=Vth;
        Ps1(k+1)=Ps1(k+1)+dPs1*(1-Ps1(k+1));
        index2=[index2 k+1];
    end
end

if length(index1) > 0
T1 = t(index1(end)) - t(index1(end-1));
f1=1/T1*1000     % questo rappresenta solo l'ultima frequenza. In rltà la frequenza varia nel tempo
end
if length(index2) > 0
T2 = t(index2(end)) - t(index2(end-1));
f2=1/T2*1000
end

figure
subplot(2,1,1)
plot(t,V1);title('voltage and threshold');
hold on
plot(t,Vt1,'r');
subplot(2,1,2)
plot(t,V2);title('voltage and threshold');
hold on
plot(t,Vt2,'r');
figure
subplot(2,1,1)
plot(t,Ps1,'linewidth',1);title('Ps1');
subplot(2,1,2)
plot(t,Ps2,'linewidth',1);title('Ps2');
figure
plot(t,V1,'b','linewidth',1);
hold on
plot(t,V2,'r','linewidth',1);
plot(t,Vt1,'--g','linewidth',1)
axis([0 t(end) -66 -45])

spikes1 = zeros(1,L);
spikes1(index1) = 1;
spikes2=zeros(1,L);
spikes2(index2) = 1;

figure
subplot(211)
plot(t,spikes1)
axis([0 t(end) 0 1.1])
subplot(212)
plot(t,spikes2)
axis([0 t(end) 0 1.1])

