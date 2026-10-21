function S = load_iris(root)
%LOAD_IRIS 读取实验一参考数据（Iris 鸢尾花，制表符分隔）
%   每行 = 4 个特征 + 1 个类别(1/2/3)
%   S.train / S.test / S.total 各含字段：X(4xN), y(1xN), T(3xN one-hot)

if nargin < 1 || isempty(root)
    root = fullfile(project_root(), '实验一参考数据');
end

S.train = local_read(fullfile(root, 'trainData.txt'));
S.test  = local_read(fullfile(root, 'testData.txt'));
S.total = local_read(fullfile(root, 'totalData.txt'));

S.nIn  = 4;
S.nOut = 3;
S.className = {'山鸢尾 setosa', '变色鸢尾 versicolor', '维吉尼亚鸢尾 virginica'};
end

% ------------------------------------------------------------------
function D = local_read(fname)
if ~isfile(fname)
    error('找不到数据文件: %s', fname);
end
raw = readmatrix(fname, 'Delimiter', '\t');
raw = raw(~isnan(raw(:,1)), :);      % 去掉可能的空行
D.X = raw(:, 1:4)';
D.y = raw(:, 5)';
D.T = onehot(D.y, 3);
end
