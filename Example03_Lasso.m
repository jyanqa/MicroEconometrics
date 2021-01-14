clear all;

n = 100; k = 50;

covx = NaN(k+1,k+1);
for ii = 1:k+1
    for jj = 1:k+1
        covx(ii,jj) = 0.5^abs(ii-jj);
    end
end

rng(123)
x = mvnrnd(zeros(1,k+1),covx,n);

x0 = x(:,1);            % variable of interest
z = x(:,2:end);         % control variables/Variable for Lasso selection
X = [ones(n,1) x0 z];   % True set of regressors

a = 2;                  % constant
b0 = 1;                 % coef of interest
bk = 0.01+(0.5-0.01)*rand(k,1);      % coef controls
B0 = [a b0 bk']';        % true coeff
y0 = X*B0;
nsim = 1000;

b0hat = zeros(nsim,1);
tic
for ss = 1:nsim
    y = y0 + randn(n,1);
    [BL,fitinfo] = lasso(z,y,'CV',5,'NumLambda',10);
    zl=z(:,ne(BL(:,fitinfo.IndexMinMSE),0));
    Xlasso = [ones(n,1) x0 zl];
    Bhat = Xlasso\y;
    b0hat(ss,1) = Bhat(2);
end
toc

clear BL fitinfo ss ii jj zl Xlasso y


%%
clc
fig = figure(1);
histogram(b0hat,20,'FaceColor',[167, 227, 98]./255); hold on
yl = ylim;
p2=plot([mean(b0hat) mean(b0hat)],yl,'--b','linewidth',3); hold on
p3=plot([1 1],yl,'-r','linewidth',3); hold off
legend([p2,p3],{'$E(\hat{\beta}_0^{lasso})$' '$\beta_0$'},'interpreter','latex');
dim = [4,3]; 
set(gcf,'paperpositionmode','manual','paperunits','inches');
set(gcf,'papersize',dim,'paperposition',[0,0,dim]);
box on
set(gca,'Layer','top');
print(fig, '-dpdf',['LassoHist.pdf']);

%%
save LassoMC.mat