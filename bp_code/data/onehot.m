function T = onehot(y, C)
%ONEHOT 类别标签转 one-hot 编码（行 = 类别）
%   y : 1xN 或 Nx1 的类别标号（1..C）
%   T : CxN 的 0/1 矩阵
y = y(:)';
if nargin < 2
    C = max(y);
end
N = numel(y);
T = zeros(C, N);
T(sub2ind([C N], y, 1:N)) = 1;
end
