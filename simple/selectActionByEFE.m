% function a_opt = selectActionByEFE(s_mu, A, B, H, R, goal, action_space)
% % selectActionByEFE - Choose action that minimizes expected free energy
% %
% % Inputs:
% %   s_mu         - current inferred latent state (6x1)
% %   A, B         - dynamics and control matrices
% %   H, R         - observation model and noise
% %   goal         - desired position (3x1)
% %   action_space - matrix of candidate actions (3 x n_actions)
% %
% % Output:
% %   a_opt - selected action (3x1)
% 
% nA = size(action_space, 2);
% EFE = zeros(1, nA);
% 
% for i = 1:nA
%     a = action_space(:, i);
% 
%     % Predict next state
%     s_pred = A * s_mu + B * a;
% 
%     % Predict observation
%     o_pred = H * s_pred;
% 
%     % Goal cost (negative log-likelihood under desired state)
%     goal_cost = 0.5 * (o_pred - goal)' * inv(R) * (o_pred - goal);
% 
%     % Optional: epistemic term (approximate with observation entropy)
%     % entropy = 0.5 * log(det(2 * pi * exp(1) * R)); % constant if R fixed
%     % For now: ignore epistemic term (or add weighted penalty)
% 
%     EFE(i) = goal_cost; % + entropy (optional)
% end
% 
% % Select action that minimizes EFE
% [~, idx] = min(EFE);
% a_opt = action_space(:, idx);
% end
% function [a_opt, EFE, o_preds] = selectActionByEFE(s_mu, A, B, H, R, goal, action_space)
% % selectActionByEFE - Choose action that minimizes expected free energy
% %
% % Also returns:
% %   EFE    - expected free energy values
% %   o_preds - predicted observations for each action (3 x nA)
% 
% nA = size(action_space, 2);
% EFE = zeros(1, nA);
% o_preds = zeros(3, nA);
% 
% for i = 1:nA
%     a = action_space(:, i);
%     s_pred = A * s_mu + B * a;
%     o_pred = H * s_pred;
% 
%     o_preds(:, i) = o_pred;
%     EFE(i) = norm(o_pred - goal)^2;  % use Euclidean distance for now
% end
% 
% [~, idx] = min(EFE);
% a_opt = action_space(:, idx);
% end
function [a_opt, EFE, o_preds] = selectActionByEFE(s_mu, A, B, H, R, goal, action_space, obstacles)
% selectActionByEFE - Choose action that minimizes expected free energy
%
% Inputs:
%   s_mu         - current inferred state (6x1)
%   A, B         - dynamics and control matrices
%   H, R         - observation model and noise
%   goal         - target location (3x1)
%   action_space - candidate actions (3 x N)
%   obstacles    - 3 x n_obs matrix of obstacle positions
%
% Outputs:
%   a_opt   - optimal action (3x1)
%   EFE     - vector of expected free energy values
%   o_preds - predicted observations from each candidate action

nA = size(action_space, 2);
EFE = zeros(1, nA);
o_preds = zeros(3, nA);

for i = 1:nA
    a = action_space(:, i);
    s_pred = A * s_mu + B * a;
    o_pred = H * s_pred;
    o_preds(:, i) = o_pred;

    % Goal alignment cost (Euclidean distance)
    goal_cost = norm(o_pred - goal)^2;

    % Obstacle penalty: inverse of distance to nearest obstacle
    dists = vecnorm(obstacles - o_pred, 2, 1); % 1 x n_obs
    min_dist = min(dists);
    obstacle_cost = 1 / (min_dist + 1e-3);  % avoid divide-by-zero

    % Total expected free energy
    EFE(i) = goal_cost + 10 * obstacle_cost;  % 10x weight on obstacle penalty
end

[~, idx] = min(EFE);
a_opt = action_space(:, idx);
end
