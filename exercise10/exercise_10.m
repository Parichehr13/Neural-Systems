clear
close all
clc
load chemo
N = length(fac);
err=1000;
W1o=(rand(1)-0.5);
W1c=(rand(1)-0.5);
teta1=(rand(1)-0.5);
W2o=(rand(1)-0.5);
W2c=(rand(1)-0.5);
teta2=(rand(1)-0.5);
Wu1=(rand(1)-0.5);
Wu2=(rand(1)-0.5);
tetau=(rand(1)-0.5);
gamma=0.001
conta=0;
Err_vet=[];
while conta<80000
    for i = 1 : N
        u1=W1o*Pao2(i)+W1c*Paco2(i)-teta1;
        u2=W2o*Pao2(i)+W2c*Paco2(i)-teta2;
        y1=sigmoid(u1);
        y2=sigmoid(u2);
        uusc=Wu1*y1+Wu2*y2-tetau;
        usc=sigmoid(uusc);
        E(i)=fac(i)-usc;
        deltau=E(i)*sigder(uusc);
        DWu1=gamma*deltau*y1;
        DWu2=gamma*deltau*y2;
        Dtetau=gamma*deltau*(-1);
        %calcolo delta prima unità nascosta
        delta1=sigder(u1)*deltau*Wu1;
        DW1o=gamma*delta1*Pao2(i);
        DW1c=gamma*delta1*Paco2(i);
        Dteta1=gamma*delta1*(-1);
        %calcolo delta seconda unità nascosta
        delta2=sigder(u2)*deltau*Wu2;
        DW2o=gamma*delta2*Pao2(i);
        DW2c=gamma*delta2*Paco2(i);
        Dteta2=gamma*delta2*(-1);
        Wu1=Wu1+DWu1;
        Wu2=Wu2+DWu2;
        tetau=tetau+Dtetau;
        W1o=W1o+DW1o;
        W1c=W1c+DW1c;
        teta1=teta1+Dteta1;
        W2o=W2o+DW2o;
        W2c=W2c+DW2c;
        teta2=teta2+Dteta2;
    end
    conta= conta+1;
    err=sum(E.^2);
    if round(conta/100)==conta/100
        disp(conta)
         Err_vet= [Err_vet err];
        [conta/1000 err]
    end
end
L = length(Err_vet);
    plot((1:1:L)*100,Err_vet,'linewidth',2)
    xlabel('Epoche','fontsize',14)
    ylabel('Errore','fontsize',14)
    set(gca,'fontsize',12)
save sinapsi
        