function q = inference_step(o, f, m0, S0,  maxIter, tol, doPlot)
% inference_step - Variational Laplace state inference given noisy observation
%
% Inputs:
%   o        - observation (e.g., 3x1 position vector)
%   f        - generative function: f(m) predicts expected observation
%   m0       - prior mean (6x1)
%   S0       - prior covariance (6x6)
%   R        - observation noise covariance (3x3)
%   maxIter  - max iterations for VL
%   tol      - convergence tolerance
%   doPlot   - true/false to plot convergence
%
% Output:
%   q - struct with:
%         .mu     - posterior mean
%         .Sigma  - posterior covariance
%         .logL   - final log likelihood
%         .allm   - history of MAP estimates

% Set observation (as cell array expected by VL)
y = o(:); % single observation as 1x1 cell array

% Wrap the generative function to work with 1x1 input cell
fwrap = @(m) f(m); % returns a 1x1 cell

% Call your VL routine
[m, V, D, logL, ~, ~, allm] = fitVariationalLaplaceThermo(y, fwrap, m0, S0, maxIter, tol, doPlot);

% Return result struct
q.mu = m;
q.Sigma = (V*V')+D;
q.logL = logL;
q.allm = allm;
end
