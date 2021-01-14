clear all; close all; clc;

% training N, test H, and simulations
N = 100; H = 25; SIM = 100;
rng(123);

% x
x = -.5+sort(rand(N+H,1));
% f(x)
y0 = 1+.8*x+sin(6*x);

% Noise variance
s2 = .25;

% no. polynomial terms
kmax = 11;
xi = [x.^(1:2:kmax)];
K = size(xi,2);

%%


mse_is = zeros(SIM,K);
mse_oos = zeros(SIM,K);

for ss = 1:SIM % simulation loop
    % simulate training and test data: y = f(x) + u
    y = y0 + randn(N+H,1)*sqrt(s2);
    yis = y(1:N);
    yoos = y(N+1:N+H);
    
    for kk = 1:K % loop on X
        Xis = [ones(N,1) xi(1:N,1:kk)];
        Xoos = [ones(H,1) xi(N+1:N+H,1:kk)];
        % ols estimation
        bhat = pinv(Xis'*Xis)*(Xis'*yis);
        % in-sample mse
        mse_is(ss,kk) =mean((yis-Xis*bhat).^2);
        % forecast on test sample
        fhat(:,ss,kk) = Xoos*bhat;
        % bias^2 forecast: f-f(x)
        bias_fhat_2(:,ss,kk) = (y0(N+1:N+H)-fhat(:,ss,kk)).^2;
        % OOS MSE
        mse_oos(ss,kk) =mean((yoos-fhat(:,ss,kk)).^2);

    end
end
%% Plot Bias-var trade-off
fig2 = figure(2);
subplot(1,2,1)
plot((1:K),mean(mse_is,1),'-or'); hold on
plot((1:K),mean(mse_oos,1),'-*b'); hold on; 
plot([.5 K+.5],[s2 s2],':k'); hold off; 
xlim([.5 K+.5]); xticks((1:K)); grid on 
legend('In-Sample MSE','Out-of-sample MSE','var(error)')
xlabel('# Regressors')
        
varf = squeeze(mean(var(fhat,0,1),2));
bias2 = squeeze(mean(mean(bias_fhat_2,1),2));
subplot(1,2,2)
plot((1:K),mean(mse_oos,1),'-*b'); hold on; 
plot((1:K),varf,'-.r'); hold on; 
plot((1:K),bias2,'-dk'); hold on; 
plot([.5 K+.5],[s2 s2],':k'); hold off; 

xlim([.5 K+.5]); xticks((1:K)); grid on 
legend('Out-of-sample MSE','var(forecast)','bias(forecast)^2','var(error)')
xlabel('# Regressors')

dim = [9,6]; 
set(gcf,'paperpositionmode','manual','paperunits','inches');
set(gcf,'papersize',dim,'paperposition',[0,0,dim]);
box on
set(gca,'Layer','top');
%print(fig2, '-r1200' ,'-dpdf',['biasvartradeoff.pdf']);