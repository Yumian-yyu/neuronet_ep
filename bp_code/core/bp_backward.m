function [gW, gb, loss] = bp_backward(net, X, T, lossType)
%BP_BACKWARD 误差反传（累积整个批次的梯度）
%
%   [gW, gb, loss] = bp_backward(net, X, T, 'mse')
%
%   误差函数（MSE）:  E = (1/N) * sum_i 0.5 * sum_j (t_j - y_j)^2
%   输出层误差:      delta_L = (y - t) .* f'(z_L)      ← ③
%   隐层误差:        delta_l = (W_{l+1}' * delta_{l+1}) .* f'(z_l)   ← ④
%   权值梯度:        dW_l = delta_l * a_{l-1}' / N      ← ⑤
%
%   lossType:
%     'mse' —— 平方误差 + 任意输出激活（教材做法，默认）
%     'ce'  —— 交叉熵 + softmax 输出（此时输出层 delta = y - t）

if nargin < 4 || isempty(lossType)
    lossType = 'mse';
end

[A, Z] = bp_forward(net, X);
Y  = A{end};   % 网络输出
N  = size(X, 2);
L  = net.L;

gW = cell(1, L);
gb = cell(1, L);

% ---- 输出层误差项 ----
switch lower(lossType)
    case 'mse'
        loss  = mean(0.5 * sum((T - Y).^2, 1));
        delta = (Y - T) .* bp_dact(Z{L}, net.outAct);   % ③ δL = (y-t)·f'(z)输出层误差
    case 'ce'
        Yc    = max(Y, realmin);
        loss  = -mean(sum(T .* log(Yc), 1));
        delta = (Y - T);                     % softmax + 交叉熵的简化形式
    otherwise
        error('未知损失函数: %s', lossType);
end

% ---- 逐层反传 ----
for l = L:-1:1
    gW{l} = delta * A{l}' / N;                       % ⑤ dW = δ·aᵀ/N 梯度
    gb{l} = mean(delta, 2);
    if l > 1
        delta = (net.W{l}' * delta) .* bp_dact(Z{l-1}, net.act{l-1});  % ④ δl = (Wᵀδ)·f'(z)隐层误差
    end
end
end
