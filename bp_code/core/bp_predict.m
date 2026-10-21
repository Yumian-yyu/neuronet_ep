function [yhat, Y] = bp_predict(net, X)
%BP_PREDICT 前向计算并取最大输出作为类别
%   yhat : 1 x N 预测类别（1..C）
%   Y    : C x N 网络原始输出
[A, ~] = bp_forward(net, X);
Y = A{end};
[~, yhat] = max(Y, [], 1);
end
