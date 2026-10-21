function P = norm_fit(X, method)
%NORM_FIT 按列(每个特征)统计归一化参数，只应在训练集上调用，避免信息泄漏
%   method: 'none' | 'minmax'(映射到[0,1]) | 'minmax11'([-1,1]) | 'zscore'
if nargin < 2 || isempty(method)
    method = 'none';
end
P.method = lower(method);
switch P.method
    case 'none'
        P.mu = 0;  P.sd = 1;
    case {'minmax', 'minmax11'}
        P.lo = min(X, [], 2);
        P.hi = max(X, [], 2);
        P.span = P.hi - P.lo;
        P.span(P.span == 0) = 1;            % 防止常数特征除零
    case 'zscore'
        P.mu = mean(X, 2);
        P.sd = std(X, 0, 2);
        P.sd(P.sd == 0) = 1;
    otherwise
        error('未知归一化方式: %s', method);
end
end
