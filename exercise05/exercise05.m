clc
clear
close all
Wep = 135;  %68 alpha-beta;  %108; % 128; %135 alpha;  %270 theta;   %675  % 1350
Wpe=0.8*Wep;
Wip=0.25*Wep;
Wpi=0.25*Wep;
Ae=3.25;
Ai=22;   %22;  %17.6;
ae=100;   % 50 75 150 200
ai=50;
kr=0.56;
v0=6;
rmax=5;
dt=0.0001;
t=[0:dt:200];
N=length(t);
yp=zeros(N,1);
zp=zeros(N,1);
ye=zeros(N,1);
ze=zeros(N,1);
yi=zeros(N,1);
zi=zeros(N,1);

for k = 1:N-1
n=200*randn(1,1)+160;

vp=Wpe*ye(k)-Wpi*yi(k);
ve=Wep*yp(k);
vi=Wip*yp(k);
rp=rmax/(1+exp(-kr*(vp-v0)));
re=rmax/(1+exp(-kr*(ve-v0)));
ri=rmax/(1+exp(-kr*(vi-v0)));
dyp = zp(k);
dzp = Ae*ae*rp-2*ae*zp(k)-ae*ae*yp(k);
dye = ze(k);
dze = Ae*ae*(re+n/Wpe)-2*ae*ze(k)-ae*ae*ye(k);
dyi = zi(k);
dzi = Ai*ai*ri-2*ai*zi(k)-ai*ai*yi(k);
yp(k+1)=yp(k)+dyp*dt;
zp(k+1)=zp(k)+dzp*dt;
ye(k+1)=ye(k)+dye*dt;
ze(k+1)=ze(k)+dze*dt;
yi(k+1)=yi(k)+dyi*dt;
zi(k+1)=zi(k)+dzi*dt;
end

eeg = Wpe*ye-Wpi*yi;
inizio = round(1.01/dt);  % elimino il primo secondo di transitorio
fine= round(3/dt);
plot(t(inizio:fine),eeg(inizio:fine),'r')
figure
[Peeg,f] =  pwelch(eeg,20000,[],20000,10000);
plot(f(5:80),Peeg(5:80),'b','linewidth',2)
