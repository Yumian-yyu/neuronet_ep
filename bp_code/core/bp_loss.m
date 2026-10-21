function e = bp_loss(net, X, T, lossType)
%BP_LOSS 只算误差不反传，用于训练过程中评估
if nargin < 4 || isempty(lossType)
    lossType = 'mse';
end
[A, ~] = bp_forward(net, X);
Y = A{end};
switch lower(lossType)
    case 'mse'
        e = mean(0.5 * sum((T - Y).^2, 1));
    case 'ce'
        e = -mean(sum(T .* log(max(Y, realmin)), 1));
    otherwise
        error('未知损失函数: %s', lossType);
end
end
