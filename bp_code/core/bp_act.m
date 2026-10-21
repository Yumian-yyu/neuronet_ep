function a = bp_act(z, type)
%BP_ACT 激活函数（按列作用，支持批量）5种
%   a = bp_act(z, 'logsig') | 'tansig' | 'purelin' | 'relu' | 'softmax'
switch lower(type)
    case 'logsig'                 % 1/(1+e^-x)，输出 (0,1)
        a = 1 ./ (1 + exp(-z));
    case 'tansig'                 % tanh(x)，输出 (-1,1)
        a = tanh(z);
    case 'purelin'                % 线性
        a = z;
    case 'relu'
        a = max(0, z);
    case 'softmax'                % 逐列归一化
        z = z - max(z, [], 1);    % 防止 exp 溢出
        e = exp(z);
        a = e ./ sum(e, 1);
    otherwise
        error('未知激活函数: %s', type);
end
end
