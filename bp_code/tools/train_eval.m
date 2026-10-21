function m = train_eval(sizes, Xtr, Ttr, ytr, Xte, yte, o)
%TRAIN_EVAL 一次“初始化→训练→评估”的完整流程，供各实验脚本调用
%
%   o 为选项结构体，缺省字段自动补齐：
%     seed, act, outAct, init, wRange  -> 传给 bp_init
%     method, lr, mc, maxEpoch, goal, loss -> 传给 bp_train
%
%   返回 m：stop, mse, accTrain, accTest, epochs, time, nParam, perf, lr

o = set_default(o, 'seed',      1);
o = set_default(o, 'act',      'logsig');
o = set_default(o, 'outAct',   'logsig');
o = set_default(o, 'init',     'nguyen');
o = set_default(o, 'wRange',   0.5);
o = set_default(o, 'method',   'gda');
o = set_default(o, 'lr',       0.1);
o = set_default(o, 'mc',       0.9);
o = set_default(o, 'maxEpoch', 2000);
o = set_default(o, 'goal',     1e-3);
o = set_default(o, 'loss',     'mse');

net = bp_init(sizes, 'seed', o.seed, 'act', o.act, 'outAct', o.outAct, ...
              'init', o.init, 'wRange', o.wRange);

[net, tr] = bp_train(net, Xtr, Ttr, 'method', o.method, 'lr', o.lr, 'mc', o.mc, ...
                     'maxEpoch', o.maxEpoch, 'goal', o.goal, 'loss', o.loss);

m.stop     = tr.stop;
m.mse      = tr.perf(end);
m.accTrain = mean(bp_predict(net, Xtr) == ytr);
m.accTest  = mean(bp_predict(net, Xte) == yte);
m.epochs   = numel(tr.perf);
m.time     = tr.time;
m.perf     = tr.perf;
m.lr       = tr.lr;
m.nParam   = sum(cellfun(@numel, net.W)) + sum(cellfun(@numel, net.b));
m.net      = net;
end

% ------------------------------------------------------------------
function o = set_default(o, f, v)
if ~isfield(o, f) || isempty(o.(f))
    o.(f) = v;
end
end
