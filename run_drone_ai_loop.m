% run_active_inference_loop.m
% Simulates agent–environment interaction for Active Inference drone

% Initialize agent
agent = ActiveInferenceDroneAgent();

% Define environment dynamics (can differ from agent’s model)
A_env = agent.A;
B_env = agent.B;
H_env = agent.H;
Q_env = 1e-3 * eye(6);
R_env = 1e-2 * eye(3);

% Set goal
agent.set_goal([5; 5; 5]);

% Initial true state
s_true = zeros(6,1);

% Simulation horizon
T = 100;

% Storage
state_true = zeros(6, T);
state_est = zeros(6, T);
actions = zeros(3, T);
obs = zeros(3, T);

% Initial dummy action
a = [0; 0; 0];

% Simulation loop
for t = 1:T
    % Step 1: environment transition and noisy observation
    [s_true, o] = agent.environment_step(s_true, a, A_env, B_env, H_env, Q_env, R_env);

    % Step 2: agent infers state from observation
    agent.infer_state(o);

    % Step 3: agent selects action
    a = agent.select_action();

    % Log
    state_true(:, t) = s_true;
    state_est(:, t) = agent.mu;
    actions(:, t) = a;
    obs(:, t) = o;
end

% Plot results
figure;
plot3(state_true(1,:), state_true(2,:), state_true(3,:), 'k-', 'LineWidth', 2); hold on;
plot3(state_est(1,:), state_est(2,:), state_est(3,:), 'b--');
plot3(agent.goal(1), agent.goal(2), agent.goal(3), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('X'); ylabel('Y'); zlabel('Z'); grid on;
legend('True Trajectory', 'Estimated Trajectory', 'Goal');
title('Active Inference Drone Navigation');
