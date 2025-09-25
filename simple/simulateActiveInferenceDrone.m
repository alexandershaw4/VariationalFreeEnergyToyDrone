%function simulateActiveInferenceDrone()
% simulateActiveInferenceDrone - Runs full drone simulation with active inference

% ======== PARAMETERS ========
T = 100;                       % number of timesteps
dt = 0.1;
maxIter = 32;
tol = 1e-4;
doPlot = false;

% Dynamics matrices
A = [1 0 0 dt 0  0;
     0 1 0 0  dt 0;
     0 0 1 0  0  dt;
     0 0 0 1  0  0;
     0 0 0 0  1  0;
     0 0 0 0  0  1];

B = [0.5*dt^2 * eye(3); dt * eye(3)];
H = [eye(3), zeros(3)];

Q = 1e-3 * eye(6);       % process noise
R = 1e-2 * eye(3);       % observation noise

% Action space (discrete)
vals = [-1, 0, 1];
[Ax, Ay, Az] = ndgrid(vals, vals, vals);
action_space = [Ax(:)'; Ay(:)'; Az(:)'];

% Initial state
s_true = [0; 0; 0; 0; 0; 0];
m0 = s_true + 0.1 * randn(6,1);  % initial belief
S0 = 0.1 * eye(6);               % initial uncertainty

% Target location
goal = [5; 5; 5];

% Storage
state_true = zeros(6, T);
state_est = zeros(6, T);
obs = zeros(3, T);

% ======== MAIN LOOP ========
for t = 1:T
    % Observation
    o = observeDrone(s_true, H, R);

    % Inference step
    f = @(s) H * s;  % linear generative model
    q = inference_step(o, f, m0, S0, R, maxIter, tol, doPlot);

    % Action selection
    [a, EFE_last, o_preds_last] = selectActionByEFE(q.mu, A, B, H, R, goal, action_space);

    if t == T
        EFE = EFE_last;
        o_preds = o_preds_last;
    end

    % Simulate true next state
    s_next = simulateDroneDynamics(s_true, a, A, B, Q);

    % Store data
    state_true(:, t) = s_true;
    state_est(:, t) = q.mu;
    obs(:, t) = o;

    % Update for next round
    s_true = s_next;
    m0 = q.mu;
    S0 = q.Sigma;
end

figure;
subplot(1,2,1);
plot3(state_true(1,:), state_true(2,:), state_true(3,:), 'k-', 'LineWidth', 2); hold on;
plot3(state_est(1,:), state_est(2,:), state_est(3,:), 'b--');
plot3(goal(1), goal(2), goal(3), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('X'); ylabel('Y'); zlabel('Z');
grid on; title('3D Drone Trajectory');
legend('True', 'Estimated', 'Goal');

% Plot EFE landscape (from final timestep)
subplot(1,2,2);
scatter3(o_preds(1,:), o_preds(2,:), o_preds(3,:), 60, EFE, 'filled');
xlabel('X_obs'); ylabel('Y_obs'); zlabel('Z_obs');
title('Expected Free Energy Landscape');
colorbar; axis equal;

% % ======== PLOT RESULTS ========
% plot3(state_true(1,:), state_true(2,:), state_true(3,:), 'k-', 'LineWidth', 2); hold on;
% plot3(state_est(1,:), state_est(2,:), state_est(3,:), 'b--');
% plot3(goal(1), goal(2), goal(3), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
% legend('True trajectory', 'Estimated trajectory', 'Goal');
% xlabel('X'); ylabel('Y'); zlabel('Z');
% grid on;
% title('3D Active Inference Drone Navigation');
% 
% %end
