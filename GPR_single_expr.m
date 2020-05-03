% addpath(genpath('../manopt'))
clear all
rng(2020);

n = 20;
m = 10*n^2;
k = 10; % assumed rank
k0 = 1; % true rank

% Create rank-1 gaussian sensing matrices A_i = A(:,1,i)*A(:,2,i)'
% A = 2*rand(n,2,m)-1;
A = randn(n,2,m);
A_mats = randn(n,n,m);
for ii = 1:m
    A_mats(:,:,ii) = A(:,1,ii)*A(:,2,ii)';
end

X0 = randn(n,k0)*randn(n,k0)';
y = zeros(m,1);
for ii = 1:m
    y(ii) = (A(:,1,ii)'*X0*A(:,2,ii))^2;
end

problem = struct();
problem.M = fixedrankembeddedfactory(n, n, k);
problem.cost = @(X) cost(X, y, A);
problem.egrad = @(X) egrad(X, y, A, A_mats);
% checkgradient(problem);

options = struct();
options.tolcost = 1e-12;
options.tolgradnorm = 1e-8;
options.maxiter = 1e5;
options.minstepsize = 1e-15;
options.statsfun = @(problem, X, stats) matrix_completion_error(X, stats, X0);

% Initializations:
X_init = problem.M.rand();

[X, ~, info_r] = steepestdescent(problem, X_init, options);

num_iters_r = length(info_r) - 1;

% Now optimize over correct rank:
problem.M = fixedrankembeddedfactory(n, n, k0);

X_init_corr_rk = struct();
X_init_corr_rk.U = X_init.U(:,1:k0);
X_init_corr_rk.V = X_init.V(:,1:k0);
X_init_corr_rk.S = X_init.S(1:k0,1:k0);

[X_corr_rk, ~, info_rtr] = steepestdescent(problem, X_init_corr_rk, options);

num_iters_rtr = length(info_rtr) - 1;

% Plot error and grad norm for last instance:
[errs_r, gradnorms_r] = extract_errs_from_info(info_r);
[errs_rtr, gradnorms_rtr] = extract_errs_from_info(info_rtr);

figure, subplot(2,1,1), semilogy(errs_r), hold on, semilogy(errs_rtr)
legend('k = 10', 'k = 1')
ylabel('$||X - X_{true}||_F / ||X_{true}||_F$', 'interpreter', 'latex')
xlabel('iteration #')
title(['m = ' num2str(m)])
set(gca, 'fontsize', 14)

subplot(2,1,2), semilogy(gradnorms_r), hold on, semilogy(gradnorms_rtr)
legend('k = 10', 'k = 1')
ylabel('$||grad\ f(X)||_F$', 'interpreter', 'latex')
xlabel('iteration #')
set(gca, 'fontsize', 14)

function f = cost(X, y, A)
m = length(y);
IP = zeros(m, 1);
for ii = 1:m
    IP(ii) = A(:,1,ii)'*X.U*X.S*X.V'*A(:,2,ii);
end
f = norm(IP.^2-y)^2/4/norm(y)^2;

end

function g = egrad(X, y, A, A_mats)
m = length(y);
IP = zeros(m, 1);
for ii = 1:m
    IP(ii) = A(:,1,ii)'*X.U*X.S*X.V'*A(:,2,ii);
end

g = zeros(size(X,1),size(X,2));
for ii = 1:m
    g = g + (IP(ii)^2-y(ii))*IP(ii)*A_mats(:,:,ii);
end
g = g./norm(y)^2;
end

function stats = matrix_completion_error(X, stats, X_true)
X = X.U*X.S*X.V';
stats.error = min(norm(X-X_true,'fro'), norm(X+X_true,'fro'))/norm(X_true,'fro');
end

function f = cost_factor(X, y, A)
m = length(y);
IP = zeros(m, 1);
for ii = 1:m
    IP(ii) = (A(:,1,ii)'*X{1})*(X{2}'*A(:,2,ii));
end
f = norm(IP.^2-y)^2/4/norm(y)^2;

end

function g = grad_factor(X, y, A, A_mats)
m = length(y);
IP = zeros(m, 1);
for ii = 1:m
    IP(ii) = (A(:,1,ii)'*X{1})*(X{2}'*A(:,2,ii));
end

g = zeros(size(X{1},1),size(X{2},1));
for ii = 1:m
    g = g + (IP(ii)^2-y(ii))*IP(ii)*A_mats(:,:,ii);
end
g = g./norm(y)^2;

g = {g*X{2}, g'*X{1}};
end

function stats = matrix_completion_error_factor(X, stats, X_true)
X = X{1}*X{2}';
stats.error = min(norm(X-X_true,'fro'), norm(X+X_true,'fro'))/norm(X_true,'fro');
end

function [errs, gradnorms] = extract_errs_from_info(info)
errs = zeros(length(info), 1);
for ii = 1:length(info)
    errs(ii) = info(ii).error;
end

gradnorms = zeros(length(info),1);
for ii = 1:length(info)
    gradnorms(ii) = info(ii).gradnorm;
end
end
