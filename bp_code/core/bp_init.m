function net = bp_init(sizes, varargin)
%BP_INIT 初始化一个全连接 BP 网络（手写实现，不依赖工具箱）
%
%   net = bp_init([4 8 3])
%   net = bp_init([4 8 3], 'act','tansig', 'outAct','logsig', ...
%                 'init','nguyen', 'seed',1)
%
%   输入：
%     sizes   —— 各层神经元个数，例如 [4 8 3] 表示 4 输入、1 个隐层8节点、3 输出
%                双隐层写 [4 8 4 3]
%   选项：
%     'act'    隐层激活函数，字符或元胞（每个隐层一个），可选
%              'logsig' | 'tansig' | 'purelin' | 'relu'
%     'outAct' 输出层激活函数，默认 'logsig'（经典 BP 教材做法）
%              若用 'softmax'，训练时应把 loss 设为 'ce'
%     'init'   权值初始化：'nguyen'（推荐）| 'uniform' | 'randn' | 'zeros'
%     'wRange' 当 init='uniform' 时权值范围 [-wRange, wRange]，默认 0.5
%     'seed'   随机种子，给定则重置随机流，便于重复实验
%
%   输出 net 结构体字段：
%     sizes, L(层数), act(隐层激活), outAct, W(1xL元胞), b(1xL元胞)

p = inputParser;
p.addParameter('act',    'logsig');
p.addParameter('outAct', 'logsig');
p.addParameter('init',   'nguyen');
p.addParameter('wRange', 0.5);
p.addParameter('seed',   []);
p.parse(varargin{:});
o = p.Results;

if numel(sizes) < 2
    error('sizes 至少要包含输入层和输出层');
end
if ~isempty(o.seed)
    rng(o.seed);                 % 固定随机流，保证实验可重复
end

L = numel(sizes) - 1;

% 隐层激活函数整理成元胞，长度 = 隐层个数
if ischar(o.act)
    acts = repmat({o.act}, 1, L-1);
else
    acts = o.act;
    if numel(acts) ~= L-1
        error('隐层激活函数个数应为 %d，当前 %d', L-1, numel(acts));
    end
end

net        = struct();
net.sizes  = sizes(:)';
net.L      = L;
net.act    = acts;
net.outAct = lower(o.outAct);
net.W      = cell(1, L);
net.b      = cell(1, L);

for l = 1:L
    [net.W{l}, net.b{l}] = local_init_W(sizes(l+1), sizes(l), o.init, o.wRange);
end
end

% ------------------------------------------------------------------
function [W, b] = local_init_W(m, n, method, wRange)%权值初始化
% 第 l 层权值 W 为 m x n（m=本层节点数, n=上层节点数）
switch lower(method)
    case 'uniform'
        W = (2*rand(m,n) - 1) * wRange;
        b = (2*rand(m,1) - 1) * wRange;
    case 'randn'
        W = randn(m,n) / sqrt(n);
        b = zeros(m,1);
    case 'zeros'
        W = zeros(m,n);
        b = zeros(m,1);
    case 'nguyen'   % Nguyen-Widrow 初始化，收敛更快
        gamma = 0.7 * m^(1/n);
        W = randn(m,n);
        W = W ./ sqrt(sum(W.^2, 2));      % 每个节点权向量归一化
        W = gamma * W;
        b = gamma * (2*rand(m,1) - 1);
    otherwise
        error('未知初始化方法: %s', method);
end
end
