function d = bp_dact(z, type)
%BP_DACT 激活函数的导数（对净输入 z 求导，按列作用）
%   d = bp_dact(z, 'logsig') | 'tansig' | 'purelin' | 'relu'
%   注意：softmax 的导数不单独给出，softmax+交叉熵时输出层误差直接是 (y-t)
switch lower(type)
    case 'logsig'
        f = 1 ./ (1 + exp(-z));
        d = f .* (1 - f);         % f(1-f)
    case 'tansig'
        f = tanh(z);
        d = 1 - f.^2;             % 1-f^2
    case 'purelin'
        d = ones(size(z));
    case 'relu'
        d = double(z > 0);
    case 'softmax'
        error('softmax 的导数由交叉熵损失直接给出，不应单独调用');
    otherwise
        error('未知激活函数: %s', type);
end
end
