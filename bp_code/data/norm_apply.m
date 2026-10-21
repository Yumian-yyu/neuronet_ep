function Xn = norm_apply(X, P)
%NORM_APPLY 用训练集统计到的参数对数据做归一化
switch P.method
    case 'none'
        Xn = X;
    case 'minmax'
        Xn = (X - P.lo) ./ P.span;
    case 'minmax11'
        Xn = 2 * (X - P.lo) ./ P.span - 1;
    case 'zscore'
        Xn = (X - P.mu) ./ P.sd;
    otherwise
        error('未知归一化方式: %s', P.method);
end
end
