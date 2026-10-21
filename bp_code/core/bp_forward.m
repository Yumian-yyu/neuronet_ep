function [A, Z] = bp_forward(net, X)
%BP_FORWARD 前向传播
%   X : nIn x N  输入（每列一个样本）
%   A : 1 x (L+1) 元胞，A{1}=X，A{l+1} 为第 l 层输出
%   Z : 1 x L 元胞，第 l 层净输入（激活前）
%
%   对应教材公式：  a_l = f( W_l * a_{l-1} + b_l )

N  = size(X, 2);
L  = net.L;
A  = cell(1, L+1);
Z  = cell(1, L);
A{1} = X;

for l = 1:L
    zl = net.W{l} * A{l} + repmat(net.b{l}, 1, N);   % ① 净输入 z = W·a+b
    Z{l} = zl;
    if l == L
        A{l+1} = bp_act(zl, net.outAct);             % ② 输出层 a = f(z)
    else
        A{l+1} = bp_act(zl, net.act{l});             % ② 隐层   a = f(z)
    end
end
end
