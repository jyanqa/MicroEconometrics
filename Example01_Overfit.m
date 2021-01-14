clear all; close all
% seed
rng(123)
% sample size
n  = 100;

% X and error term
x = -.5+sort(rand(n,1));
u = randn(n,1);
% True f(X)
y0 = 1+0.8*x+sin(6*x);

% y = f(X) + u; add noise
y = y0 + u;


% Plot signal
fig1 = figure(1);
plot(x,y0,'--k'); hold on;
% Plot sample of data
plot(x,y,'.r'); hold off; legend('f(x) = 1 + 0.8x + sin(6x)','Data: y = f(x) + noise');

dim = [8,6]; 
set(gcf,'paperpositionmode','manual','paperunits','inches');
set(gcf,'papersize',dim,'paperposition',[0,0,dim]);
box on
set(gca,'Layer','top');
print(fig1, '-dpdf',['Overfit_1.pdf']);

%% Use polynomials to approximate unknown DGP
kmax = 15; % X = [1 x, x^2, ..., x.^kmax]

% regressor
X = ones(n,1);
% Plot data and fitted
fig2 = figure(2);
for jj = 1:kmax
    X = [X x.^jj];
    [B,~,uhat,~,stats] = regress(y,X);
    subplot(3,5,jj);
    plot(x,y0,'--k',x,y,'.r',x,X*B,'-b'); legend('f(x)','Data',['Fitted, x^{' num2str(jj) '}'],'Location','NorthWest');
    title(['R2 = ' num2str(stats(1),'%5.4f'), ', RMSE = ' num2str(mean(uhat.^2).^0.5,'%5.4f')])
end

dim = [12,8]; 
set(gcf,'paperpositionmode','manual','paperunits','inches');
set(gcf,'papersize',dim,'paperposition',[0,0,dim]);
box on
set(gca,'Layer','top');
print(fig2, '-dpdf',['Overfit_2.pdf']);

%% Focus on the fit for k = 1,2,3,15
XX{1} = [ones(n,1) x];
XX{2} = [ones(n,1) x.^(1:2)];
XX{3} = [ones(n,1) x.^(1:3)];
XX{4} = [ones(n,1) x.^(1:15)];
tit = {'x','x, x^2','x, x^2, x^3','x, x^2, ..., x^{15}'};
fig3 = figure(3);
for ii = 1:4
    X = XX{ii};
    P = (X'*X)\X';
    for ss = 1:100
        y = y0 + randn(n,1)*0.75;
        yhat(:,ss) = X*(P*y);
    end
    subplot(2,2,ii)
    plot(x,yhat,'-g'); title(tit{ii});
    hold on;
    plot(x,y0,'-k','linewidth',2); hold off
end
        
dim = [8,6]; 
set(gcf,'paperpositionmode','manual','paperunits','inches');
set(gcf,'papersize',dim,'paperposition',[0,0,dim]);
box on
set(gca,'Layer','top');
print(fig3, '-dpdf',['Overfit_3.pdf']);  
        
%% Compare IS and OOS RMSE
is_rmse = NaN(kmax,1); oos_rmse=NaN(kmax,1);

X = ones(n,1);

cvsize = 10;

for jj = 1:kmax
    X = [X x.^jj];
    kk=1;
    for ii = cvsize:cvsize:100 % k-fold CV (k=10)
        % test and training data
        oos = ii-(cvsize-1):ii;
        is = setdiff(1:100,oos);
        % OLS
        [B,~,uhat] = regress(y(is),X(is,:));
        % Training and test RMSE
        is_rmse(jj,kk) = sqrt(mean(uhat.^2));
        oos_rmse(jj,kk) = sqrt(mean((y(oos)-X(oos,:)*B).^2));
        
        kk=kk+1;
    end
end

fig4 = figure(4);
subplot(1,2,1);
plot(1:kmax,log(mean(is_rmse,2)),'-or'); 
title('In sample'); ylabel('log RMSE'); xlabel('No. regressors');
xlim([0.5 kmax+.5]); xticks(1:1:kmax); grid on;

subplot(1,2,2);
plot(1:kmax,log(mean(oos_rmse,2)),'-db'); hold on;
title('Out-of-sample'); ylabel('log RMSE'); xlabel('No. regressors'); 
plot([0 kmax+1],[min(log(mean(oos_rmse,2))) min(log(mean(oos_rmse,2)))],'-k'); hold on;
yl=ylim;
plot([find(log(mean(oos_rmse,2))==min(log(mean(oos_rmse,2))))],[min(log(mean(oos_rmse,2)))],'db','MarkerFaceColor','b'); hold on;
xlim([0.5 kmax+.5]); xticks(1:1:kmax); grid on;

dim = [8,6]; 
set(gcf,'paperpositionmode','manual','paperunits','inches');
set(gcf,'papersize',dim,'paperposition',[0,0,dim]);
box on
set(gca,'Layer','top');
print(fig4, '-dpdf',['Overfit_4.pdf']);    