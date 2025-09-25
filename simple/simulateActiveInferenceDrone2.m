function simulateActiveInferenceDrone2()
% Full drone simulation with Active Inference, moving goal, and obstacles


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
S0 = 1.0 * eye(6);               % initial uncertainty

% Obstacles
n_obs = 10;
obstacles = 10 * rand(3, n_obs);

% Moving goal trajectory
goal_traj = [5 + sin(linspace(0, 2*pi, T));
             5 + cos(linspace(0, 2*pi, T));
             5 + 0.5*sin(linspace(0, 4*pi, T))];

% Storage
state_true = zeros(6, T);
state_est = zeros(6, T);
obs = zeros(3, T);

% ======== MAIN LOOP ========
for t = 1:T
    goal = goal_traj(:, t);

    % Observation
    o = observeDrone(s_true, H, R);

    % Inference step
    f = @(s) H * s;  % linear generative model
    q = inference_step(o, f, m0, S0, maxIter, tol, doPlot);

    % Action selection with EFE and obstacle cost
    [a, EFE, o_preds] = selectActionByEFE(q.mu, A, B, H, R, goal, action_space, obstacles);

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

    % Save EFE plot data at final step
    if t == T
        o_preds_final = o_preds;
        EFE_final = EFE;
    end
end

% ======== PLOT RESULTS ========
figure;
subplot(1,2,1);
plot3(state_true(1,:), state_true(2,:), state_true(3,:), 'k-', 'LineWidth', 2); hold on;
plot3(state_est(1,:), state_est(2,:), state_est(3,:), 'b--');
plot3(goal_traj(1,:), goal_traj(2,:), goal_traj(3,:), 'r-', 'LineWidth', 1.5);
scatter3(obstacles(1,:), obstacles(2,:), obstacles(3,:), 100, 'x', 'MarkerEdgeColor', [0.4 0.2 0]);
xlabel('X'); ylabel('Y'); zlabel('Z'); grid on;
legend('True', 'Estimated', 'Moving Goal', 'Obstacles');
title('3D Drone Trajectory');

subplot(1,2,2);
scatter3(o_preds_final(1,:), o_preds_final(2,:), o_preds_final(3,:), 60, EFE_final, 'filled');
colorbar;
xlabel('X_{obs}'); ylabel('Y_{obs}'); zlabel('Z_{obs}');
title('Expected Free Energy Landscape');
grid on;

end
