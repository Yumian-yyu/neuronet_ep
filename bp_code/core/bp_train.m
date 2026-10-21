function [net, tr] = bp_train(net, X, T, varargin)
%BP_TRAIN 手写 BP 网络训练器（标准梯度下降及其改进算法）
%
%   [net, tr] = bp_train(net, X, T, 'method','gdm', 'lr',0.1, 'mc',0.9, ...
%                        'maxEpoch',2000, 'goal',1e-4, 'loss','mse', ...
%                        'valX',[], 'valT',[], 'valFail',300, 'verbose',false)
%
%   训练算法 method：
%     'gd'  标准梯度下降:      W(k+1) = W(k) - lr * dW
%     'gdm' 附加动量项:        dW(k+1) = mc*dW(k) + lr*g(k)
%                             W(k+1) = W(k) - dW(k+1)
%     'gda' 自适应学习率:      误差上升(>4%)则回退并 lr*0.7，下降则 lr*1.05
%     'gdx' 自适应学习率 + 动量
%     'inc' 增量(在线)学习:    每输入一个样本就更新一次权值
%
%   返回 tr：epoch/perf/vperf/lr/grad 记录向量，stop 停止原因，time 训练耗时

p = inputParser;
p.addParameter('method',  'gdm');
p.addParameter('lr',      0.1);
p.addParameter('mc',      0.9);
p.addParameter('maxEpoch',2000);
p.addParameter('goal',    0);        % 为0表示不按目标误差早停
p.addParameter('loss',    'mse');
p.addParameter('valX',    []);
p.addParameter('valT',    []);
p.addParameter('valFail', inf);
p.addParameter('minGrad', 1e-6);
p.addParameter('returnBest', true);
p.addParameter('verbose', false);
p.parse(varargin{:});
o = p.Results;

method   = lower(o.method);
useMom   = any(strcmp(method, {'gdm','gdx'}));
adaptive = any(strcmp(method, {'gda','gdx'}));

L  = net.L;
dW = cell(1,L);  db = cell(1,L);
for l = 1:L
    dW{l} = zeros(size(net.W{l}));
    db{l} = zeros(size(net.b{l}));
end

M = o.maxEpoch;
tr.epoch = zeros(M,1);  tr.perf = zeros(M,1);  tr.vperf = zeros(M,1);
tr.lr    = zeros(M,1);  tr.grad = zeros(M,1);
tr.stop  = 'max_epoch'; tr.time = 0; tr.bestEpoch = 0;

hasVal = ~isempty(o.valX);
bestV  = inf;
bestNet = net;
vFail  = 0;
lr     = o.lr;
N      = size(X,2);

t0 = tic;
for ep = 1:M
    if strcmp(method, 'inc')
        % ---------- 增量学习：逐样本更新 ----------
        for i = 1:N
            [gW, gb] = bp_backward(net, X(:,i), T(:,i), o.loss);
            for l = 1:L
                net.W{l} = net.W{l} - lr * gW{l};
                net.b{l} = net.b{l} - lr * gb{l};
            end
        end
        perf = bp_loss(net, X, T, o.loss);
        gn   = 0;
    else
        % ---------- 批量学习：一次算完所有样本的平均梯度 ----------
        [gW, gb, perf] = bp_backward(net, X, T, o.loss);   % ①~⑤ 算梯度
        gn = sqrt(sum(cellfun(@(m) sum(m(:).^2), gW)) + ...
                  sum(cellfun(@(m) sum(m(:).^2), gb)));

        Wold = net.W;  bold = net.b;

        if useMom
            for l = 1:L
                dW{l} = o.mc * dW{l} + lr * gW{l};
                db{l} = o.mc * db{l} + lr * gb{l};
            end
        else
            for l = 1:L
                dW{l} = lr * gW{l};
                db{l} = lr * gb{l};
            end
        end

        % ⑥ 更新权值 W ← W − η·dW
        for l = 1:L
            net.W{l} = net.W{l} - dW{l};
            net.b{l} = net.b{l} - db{l};
        end

        if adaptive
            perfNew = bp_loss(net, X, T, o.loss);
            if perfNew > perf * 1.04          % 这一步让误差变大 → 回退、降学习率
                net.W = Wold;  net.b = bold;
                lr = lr * 0.7;
                if useMom
                    for l = 1:L, dW{l}(:) = 0; db{l}(:) = 0; end
                end
            elseif perfNew < perf             % 有效 → 提一点学习率
                lr = lr * 1.05;
            end
            lr = min(max(lr, 1e-6), 10);
        end
    end

    % ---------- 记录 ----------
    tr.epoch(ep) = ep;
    tr.perf(ep)  = perf;
    tr.lr(ep)    = lr;
    tr.grad(ep)  = gn;

    if hasVal
        vp = bp_loss(net, o.valX, o.valT, o.loss);
        tr.vperf(ep) = vp;
        if vp < bestV
            bestV = vp;  bestNet = net;  tr.bestEpoch = ep;  vFail = 0;
        else
            vFail = vFail + 1;
        end
    end

    if o.verbose && (mod(ep,200)==0 || ep==1)
        fprintf('  epoch %4d  MSE=%.6f  lr=%.4f\n', ep, perf, lr);
    end

    % ---------- ⑦ 判停 ----------
    if ~isfinite(perf)
        tr.stop = 'diverged';  break;
    end
    if o.goal > 0 && perf <= o.goal
        tr.stop = 'goal';  break;
    end
    if gn <= o.minGrad && gn > 0
        tr.stop = 'min_grad';  break;
    end
    if hasVal && vFail >= o.valFail
        tr.stop = 'val_fail';  break;
    end
end

tr.time = toc(t0);
tr.epoch = tr.epoch(1:ep);
tr.perf  = tr.perf(1:ep);
tr.lr    = tr.lr(1:ep);
tr.grad  = tr.grad(1:ep);
if hasVal
    tr.vperf = tr.vperf(1:ep);
else
    tr.vperf = [];
end

if hasVal && o.returnBest && isfinite(bestV)
    net = bestNet;
end
end
