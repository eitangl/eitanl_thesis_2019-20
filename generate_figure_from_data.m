t = randi(trials);
[errs_r, gradnorms_r] = extract_errs_from_info(info_r{t});
[errs_rtr, gradnorms_rtr] = extract_errs_from_info(info_rtr{t});

figure, subplot(3,1,1), semilogy(errs_r, 'linewidth', 2), hold on, semilogy(errs_rtr, 'linewidth', 2)
legend({'k = 10', 'k = 1'}, 'box', 'off')
ylabel('$||X - X_{true}||_F / ||X_{true}||_F$', 'interpreter', 'latex')
xlabel('iteration #')
title(['m = ' num2str(m)])
set(gca, 'fontsize', 14)

subplot(3,1,2), semilogy(gradnorms_r, 'linewidth', 2), hold on, semilogy(gradnorms_rtr, 'linewidth', 2)
legend({'k = 10', 'k = 1'}, 'box', 'off')
ylabel('$||grad\ f(X)||_F$', 'interpreter', 'latex')
xlabel('iteration #')
set(gca, 'fontsize', 14)

bins = logspace(0,4,50);
subplot(3,1,3), histogram(num_iters_r, bins), hold on, histogram(num_iters_rtr, bins)
xlabel('# of iterations')
legend({'k = 10', 'k = 1'}, 'box', 'off')
set(gca, 'fontsize', 14)
set(gca,'xscale','log')

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
