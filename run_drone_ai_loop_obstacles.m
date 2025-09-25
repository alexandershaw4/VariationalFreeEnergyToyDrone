% run_active_inference_loop.m
% Simulates agent–environment interaction for Active Inference drone with obstacles

% Initialize agent
agent = ActiveInferenceDroneAgent();

% Define environment dynamics (can differ from agent’s model)
A_env = agent.A;
B_env = agent.B;
H_env = agent.H;
Q_env = 1e-3 * eye(6);
R_env = 1e-2 * eye(3);

% initi point - on the ground
start_pos = [0; 0; 0];

% Set goal
agent.set_goal([5; 5; 5]);

% Initial true state
s_true = zeros(6,1);

% Define obstacles - make sure at least one is directly in the way!
goal_pos = agent.goal(:);
n_close = 30;  % how many to place along the path
ratios = rand(1, n_close);   % sample fractional distances [0, 1]
path_points = start_pos * (1 - ratios) + goal_pos * ratios;
offsets = 0.5 * randn(3, n_close);  % noise around the path
obstacles_close = path_points + offsets;

n_total = 90;
n_far = n_total - n_close;
obstacles_far = 10 * rand(3, n_far);  % full 3D space

% all obstacles
obstacles = [obstacles_close, obstacles_far];


% Simulation horizon
T = 2000;

% Storage
state_true = zeros(6, T);
state_est = zeros(6, T);
actions = zeros(3, T);
obs = zeros(3, T);

% Initial dummy action
a = start_pos;

% Simulation loop
for t = 1:T
    % Step 1: environment transition and noisy observation
    [s_true, o] = agent.environment_step(s_true, a, A_env, B_env, H_env, Q_env, R_env);

    % Step 2: agent infers state from observation
    agent.infer_state(o);

    % check if we hit the target yet...
    arrival_threshold = 0.1;  % or 0.5 depending on tolerance
    has_arrived = norm(agent.mu(1:3) - agent.goal) < arrival_threshold;

    if has_arrived
        fprintf('Goal reached at timestep %d!\n', t);
        state_true(:,t:end)=[];
        state_est(:,t:end)=[];
        break;  
    end

    %if t > 1
    %    agent.update_dynamics(state_est(:, t-1), agent.mu, actions(:, t-1));
    %end

    % allow agent to know the time
    agent.current_time = t;
    agent.max_time = T;

    % Step 3: agent selects action (obstacle-aware)
    % EFE with obstacle penalty
    a = agent.select_action_rollout(10, obstacles);

    % Log
    state_true(:, t) = s_true;
    state_est(:, t) = agent.mu;
    actions(:, t) = a;
    obs(:, t) = o;
end

% Plot results
% figure;
% plot3(state_true(1,:), state_true(2,:), state_true(3,:), 'k-', 'LineWidth', 2); hold on;
% plot3(state_est(1,:), state_est(2,:), state_est(3,:), 'b--');
% plot3(agent.goal(1), agent.goal(2), agent.goal(3), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
% 
% [xg, yg, zg] = sphere(20);
% surf(arrival_threshold * xg + agent.goal(1), ...
%      arrival_threshold * yg + agent.goal(2), ...
%      arrival_threshold * zg + agent.goal(3), ...
%      'FaceColor', 'g', 'EdgeAlpha', 0.1, 'FaceAlpha', 0.2);
% 
% scatter3(obstacles(1,:), obstacles(2,:), obstacles(3,:), 80, 'x', 'MarkerEdgeColor', [0.5 0.1 0]);
% xlabel('X'); ylabel('Y'); zlabel('Z'); grid on;
% legend('True Trajectory', 'Estimated Trajectory', 'Goal', 'Obstacles');
% title('Active Inference Drone with Obstacles');
% 

% static
[xs, ys, zs] = sphere(20);
terrain_color = [0.6 0.8 0.6];
sky_color = [0.85 0.92 1.0];
ground_z = min([state_true(3,:), state_est(3,:), agent.goal(3)]) - 0.5;

% Precompute terrain
all_x = [state_true(1,:), state_est(1,:), obstacles(1,:), agent.goal(1)];
all_y = [state_true(2,:), state_est(2,:), obstacles(2,:), agent.goal(2)];
xlim_ = [min(all_x)-1, max(all_x)+1];
ylim_ = [min(all_y)-1, max(all_y)+1];
[xg, yg] = meshgrid(linspace(xlim_(1), xlim_(2), 50), linspace(ylim_(1), ylim_(2), 50));
zg = 0.1 * sin(0.2 * xg) .* cos(0.2 * yg) + ground_z;

% Final drone position
drone_pos = state_true(:, end);
drone_scale = 0.8;

figure('Color', sky_color, 'Position', [1440, 106, 1410, 1132]);
axis equal; grid on; view(3);
xlabel('X'); ylabel('Y'); zlabel('Z');
xlim(xlim_); ylim(ylim_); zlim([ground_z, max(state_true(3,:)) + 1]);
title('Active Inference Drone — Final Scene');

% Terrain
surf(xg, yg, zg, 'FaceAlpha', 0.6, 'EdgeColor', 'none', 'FaceColor', terrain_color); hold on;
fill3([xlim_(1), xlim_(2), xlim_(2), xlim_(1)], ...
      [ylim_(1), ylim_(1), ylim_(2), ylim_(2)], ...
      ground_z * ones(1,4), terrain_color, 'FaceAlpha', 0.2, 'EdgeColor', 'none');

% Obstacles
for i = 1:size(obstacles, 2)
    surf(0.3*xs + obstacles(1,i), 0.3*ys + obstacles(2,i), 0.3*zs + obstacles(3,i), ...
         'FaceColor', [0.5 0.3 0.1], 'EdgeColor', 'none', 'FaceAlpha', 0.9);
end

% Goal
surf(0.2*xs + agent.goal(1), 0.2*ys + agent.goal(2), 0.2*zs + agent.goal(3), ...
     'FaceColor', 'r', 'EdgeColor', 'none');

% Final drone body (mesh)
hCenter = surf(0.15*xs + drone_pos(1), 0.15*ys + drone_pos(2), 0.15*zs + drone_pos(3), ...
               'FaceColor', 'k', 'EdgeColor', 'none');
[xc, yc, zc] = cylinder(0.11, 12);
hx1 = surf(drone_scale*xc + drone_pos(1) - drone_scale/2, 0*yc + drone_pos(2), ...
           drone_scale*zc - drone_scale/2 + drone_pos(3), 'FaceColor', 'k', 'EdgeColor', 'none');
hx2 = surf(drone_scale*xc + drone_pos(1) + drone_scale/2, 0*yc + drone_pos(2), ...
           drone_scale*zc - drone_scale/2 + drone_pos(3), 'FaceColor', 'k', 'EdgeColor', 'none');
hy1 = surf(0*xc + drone_pos(1), drone_scale*yc + drone_pos(2) - drone_scale/2, ...
           drone_scale*zc - drone_scale/2 + drone_pos(3), 'FaceColor', 'k', 'EdgeColor', 'none');
hy2 = surf(0*xc + drone_pos(1), drone_scale*yc + drone_pos(2) + drone_scale/2, ...
           drone_scale*zc - drone_scale/2 + drone_pos(3), 'FaceColor', 'k', 'EdgeColor', 'none');

% Trajectories
plot3(state_true(1,:), state_true(2,:), state_true(3,:), 'k-', 'LineWidth', 2);
plot3(state_est(1,:), state_est(2,:), state_est(3,:), 'b--');
legend('True Trajectory', 'Estimated Trajectory', 'Goal');






% animation
%----------

% Animation settings
[xs, ys, zs] = sphere(20);
terrain_color = [0.6 0.8 0.6];
sky_color = [0.85 0.92 1.0];
ground_z = min([state_true(3,:), state_est(3,:), agent.goal(3)]) - 0.5;

% Precompute terrain
all_x = [state_true(1,:), state_est(1,:), obstacles(1,:), agent.goal(1)];
all_y = [state_true(2,:), state_est(2,:), obstacles(2,:), agent.goal(2)];
xlim_ = [min(all_x)-1, max(all_x)+1];
ylim_ = [min(all_y)-1, max(all_y)+1];
[xg, yg] = meshgrid(linspace(xlim_(1), xlim_(2), 50), linspace(ylim_(1), ylim_(2), 50));
zg = 0.1 * sin(0.2 * xg) .* cos(0.2 * yg) + ground_z;

% Set up figure
figure('Color', sky_color,'position',[1440         106        1410        1132]);
axis equal; grid on; view(3);
xlabel('X'); ylabel('Y'); zlabel('Z');
xlim(xlim_); ylim(ylim_); zlim([ground_z, max(state_true(3,:)) + 1]);
title('Active Inference Drone Animation');

% Terrain and ground
surf(xg, yg, zg, 'FaceAlpha', 0.6, 'EdgeColor', 'none', 'FaceColor', terrain_color); hold on;
fill3([xlim_(1), xlim_(2), xlim_(2), xlim_(1)], ...
      [ylim_(1), ylim_(1), ylim_(2), ylim_(2)], ...
      ground_z * ones(1,4), terrain_color, 'FaceAlpha', 0.2, 'EdgeColor', 'none');

% Obstacles
for i = 1:size(obstacles, 2)
    surf(0.3*xs + obstacles(1,i), 0.3*ys + obstacles(2,i), 0.3*zs + obstacles(3,i), ...
         'FaceColor', [0.5 0.3 0.1], 'EdgeColor', 'none', 'FaceAlpha', 0.9);
end

% Goal
surf(0.2*xs + agent.goal(1), 0.2*ys + agent.goal(2), 0.2*zs + agent.goal(3), ...
     'FaceColor', 'r', 'EdgeColor', 'none');

v = VideoWriter('drone_animation5.mp4', 'MPEG-4');
v.FrameRate = 30;
open(v);

az_smooth = 45; el_smooth = 30; 

% Animation loop
for t = 1:size(state_true, 2)
    % Plot current true position
    %hDrone = scatter3(state_true(1,t), state_true(2,t), state_true(3,t), ...
    %                  120, 'k', 'filled');
        % Create drone body as a 3D cross using small cylinders and sphere center
    drone_pos = state_true(:, t);
    drone_scale = 0.8;  % overall size of drone
    
    % Draw central body as small sphere
    hCenter = surf(0.15*xs + drone_pos(1), 0.15*ys + drone_pos(2), 0.15*zs + drone_pos(3), ...
                   'FaceColor', 'k', 'EdgeColor', 'none');
    
    % Draw arms in X and Y directions as rods
    [xc, yc, zc] = cylinder(0.11, 12);  % thin cylinder
    
    % X-arm
    hx1 = surf(drone_scale*xc + drone_pos(1) - drone_scale/2, ...
               0*yc + drone_pos(2), ...
               drone_scale*zc - drone_scale/2 + drone_pos(3), ...
               'FaceColor', 'k', 'EdgeColor', 'none');
    hx2 = surf(drone_scale*xc + drone_pos(1) + drone_scale/2, ...
               0*yc + drone_pos(2), ...
               drone_scale*zc - drone_scale/2 + drone_pos(3), ...
               'FaceColor', 'k', 'EdgeColor', 'none');
    
    % Y-arm
    hy1 = surf(0*xc + drone_pos(1), ...
               drone_scale*yc + drone_pos(2) - drone_scale/2, ...
               drone_scale*zc - drone_scale/2 + drone_pos(3), ...
               'FaceColor', 'k', 'EdgeColor', 'none');
    hy2 = surf(0*xc + drone_pos(1), ...
               drone_scale*yc + drone_pos(2) + drone_scale/2, ...
               drone_scale*zc - drone_scale/2 + drone_pos(3), ...
               'FaceColor', 'k', 'EdgeColor', 'none');

    % Optional: trace up to current time
    plot3(state_true(1,1:t), state_true(2,1:t), state_true(3,1:t), 'k-', 'LineWidth', 2);
    plot3(state_est(1,1:t), state_est(2,1:t), state_est(3,1:t), 'b--');

    drawnow;

    

    az = mod(t * 0.5, 360);  % slowly pan around
    el = 25 + 10 * sin(t * 0.05);  % gentle up-down oscillation
    view(az, el);

    % Optional: pause or record video
    %pause(0.02);
    frame = getframe(gcf);
    writeVideo(v, frame);

    %delete(hDrone);  % remove drone marker before next frame
    delete([hCenter, hx1, hx2, hy1, hy2]);
end

close(v);


