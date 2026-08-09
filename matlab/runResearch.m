% runResearch.m
% A-share statistical modeling study:
% GARCH(1,1) volatility model, HMM 3-state regime detection, cointegration test.

clear;
clc;
scriptDir = fileparts(mfilename('fullpath'));
projectDir = fileparts(scriptDir);
dataDir = fullfile(projectDir, 'data', 'matlab');
outputPath = fullfile(dataDir, 'results.json');

%% Load index daily returns
indexTable = readtable(fullfile(dataDir, 'index_close.csv'));
indexClose = indexTable.close;
indexReturns = 100 * diff(log(indexClose));

%% GARCH(1,1) on index returns
garchModel = garch(1, 1);
[garchFit, ~, logLikelihood] = estimate(garchModel, indexReturns, 'Display', 'off');
garchResult = struct( ...
    'constant', garchFit.Constant, ...
    'arch', garchFit.ARCH{1}, ...
    'garch', garchFit.GARCH{1}, ...
    'loglikelihood', logLikelihood, ...
    'numObs', numel(indexReturns));

%% HMM with 2 states (calm/turbulent) on absolute index returns
numStates = 2;
absReturns = abs(indexReturns);
bounds = quantile(absReturns, 1 / 2);
sequence = discretize(absReturns, [-Inf, bounds(1), Inf]);
transition0 = [0.95, 0.05; 0.05, 0.95];
emission0 = [0.7, 0.3; 0.3, 0.7];
[transition, emission] = hmmtrain(sequence, transition0, emission0, ...
    'MaxIterations', 500, 'Tolerance', 1e-6);
states = hmmviterbi(sequence(:), transition, emission);
stateMeanReturn = accumarray(states(:), indexReturns(:), [], @mean);
stateMeanAbs = accumarray(states(:), absReturns(:), [], @mean);
stateCount = accumarray(states(:), 1);
hmmResult = struct( ...
    'transition', transition, ...
    'emission', emission, ...
    'stateMeanReturn', stateMeanReturn, ...
    'stateMeanAbs', stateMeanAbs, ...
    'stateCount', stateCount);

%% Cointegration test between two bank stocks
p1 = readtable(fullfile(dataDir, '600000_close.csv'));
p2 = readtable(fullfile(dataDir, '601166_close.csv'));
logPrices = [log(p1.close), log(p2.close)];
[hCoint, pValue] = egcitest(logPrices);
cointResult = struct('isCointegrated', hCoint, 'pValue', pValue);

%% Save results
results = struct('garch', garchResult, 'hmm', hmmResult, 'cointegration', cointResult);
fileId = fopen(outputPath, 'w');
fprintf(fileId, '%s', jsonencode(results));
fclose(fileId);
fprintf('RESEARCH1_DONE\n');
