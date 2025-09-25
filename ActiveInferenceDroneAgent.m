% ActiveInferenceDroneAgent.m
% Agent-centric Active Inference framework (from scratch)
% The agent does not know the true state or true dynamics.

classdef ActiveInferenceDroneAgent < handle
    properties
        % Belief about current state
        mu       % mean (6x1)
        Sigma    % covariance (6x6)

        % Generative model parameters
        A        % state transition matrix (6x6)
        B        % control matrix (6x3)
        H        % observation matrix (3x6)
        R        % observation noise (3x3)
        Q        % process noise (6x6)

        % Prior preferences (goal)
        goal     % desired observation (3x1)

        % Inference settings
        maxIter = 32;
        tol = 1e-4;
        doPlot = false;

        % Action space
        action_space

        current_time = 1;
        max_time = 1000;  % or set dynamically
    end

    methods
        function obj = ActiveInferenceDroneAgent()
            % Initial belief
            obj.mu = zeros(6,1);
            obj.Sigma = 1.0 * eye(6);

            % Default generative model (linear)
            dt = 0.1;
            dr = .95;
            obj.A = [1 0 0 dt 0  0;
                     0 1 0 0  dt 0;
                     0 0 1 0  0  dt;
                     0 0 0 dr 0  0;
                     0 0 0 0  dr 0;
                     0 0 0 0  0  dr];

            obj.B = [0.5*dt^2 * eye(3); dt * eye(3)];
            obj.H = [eye(3), zeros(3)];
            obj.R = 1e-2 * eye(3);
            obj.Q = 1e-3 * eye(6);

            % Default goal
            obj.goal = [5; 5; 5];

            % Default action space
            %vals = [-1, 0, 1];
            vals = linspace(-1,1,5);
            [Ax, Ay, Az] = ndgrid(vals, vals, vals);
            obj.action_space = [Ax(:)'; Ay(:)'; Az(:)'];
        end

        function infer_state(obj, observation)
            % Inference via variational Laplace
            f = @(s) obj.H * s;
            [m, V, D, ~, ~, ~, ~] = fitVariationalLaplaceThermo(observation, f, obj.mu, obj.Sigma, obj.maxIter, obj.tol, obj.doPlot);
            obj.mu = m;
            obj.Sigma = (V*V')+D;  % reconstruction
        end

        function a = select_action(obj)
            % Evaluate EFE for each candidate action
            nA = size(obj.action_space, 2);
            EFE = zeros(1, nA);
            for i = 1:nA
                a_i = obj.action_space(:, i);
                s_pred = obj.A * obj.mu + obj.B * a_i;
                o_pred = obj.H * s_pred;
                EFE(i) = norm(o_pred - obj.goal)^2;  % simple goal-driven cost
            end
            [~, idx] = min(EFE);
            a = obj.action_space(:, idx);
        end

        % function a = select_action_rollout(obj, horizon, obstacles)
        %     % Multi-step EFE-based action selection with obstacle penalty and time bias
        %     nA = size(obj.action_space, 2);
        %     EFE_total = zeros(1, nA);
        % 
        %     % Require current time and max time to be set externally
        %     if ~isprop(obj, 'current_time') || ~isprop(obj, 'max_time')
        %         error('Agent must have current_time and max_time properties set.');
        %     end
        % 
        %     for i = 1:nA
        %         a1 = obj.action_space(:, i);
        %         s = obj.A * obj.mu + obj.B * a1;
        % 
        %         % Roll out future steps
        %         for h = 2:horizon
        %             a_h = obj.action_space(:, randi(nA));  % sample future actions
        %             s = obj.A * s + obj.B * a_h;
        %         end
        % 
        %         % Compute EFE terms at the end of the rollout
        %         o_pred = obj.H * s;
        %         goal_cost = norm(o_pred - obj.goal)^2;
        %         goal_weight = 1 + 5 * (obj.current_time / obj.max_time);
        % 
        %         % Obstacle penalty (soft, capped)
        %         dists = vecnorm(obstacles - o_pred, 2, 1);
        %         obstacle_cost = sum(1 ./ (dists + 1e-3));
        %         obstacle_cost = min(obstacle_cost, 5);
        % 
        %         % Arrival reward
        %         if goal_cost < 0.3^2
        %             goal_bonus = -5;
        %         else
        %             goal_bonus = 0;
        %         end
        % 
        %         % Final EFE
        %         EFE_total(i) = goal_weight * goal_cost + obstacle_cost + goal_bonus;
        %     end
        % 
        %     [~, idx] = min(EFE_total);
        %     a = obj.action_space(:, idx);
        % end
        % 

        function a = select_action_rollout(obj, horizon, obstacles)
            % Multi-step EFE-based action selection with obstacle penalty
            nA = size(obj.action_space, 2);
            EFE_total = zeros(1, nA);

            for i = 1:nA
                a1 = obj.action_space(:, i);
                s = obj.A * obj.mu + obj.B * a1;
                efe = norm(obj.H * s - obj.goal)^2;
                %relative_time = obj.current_time / obj.max_time;  % needs to be stored in agent
                %goal_cost = norm(obj.H * s - obj.goal)^2;
                %goal_weighted = goal_cost * (1 - relative_time);
                %efe = goal_weighted;

                % Roll out future steps
                for h = 2:horizon
                    a_h = obj.action_space(:, randi(nA));
                    s = obj.A * s + obj.B * a_h;
                    efe = efe + norm(obj.H * s - obj.goal)^2;
                    %relative_time = obj.current_time / obj.max_time;  % needs to be stored in agent
                    %goal_cost = norm(obj.H * s - obj.goal)^2;
                    %goal_weighted = goal_cost * (1 - relative_time);
                    %efe = efe + goal_weighted;
                end

                % Obstacle penalty at end of rollout
                dists = vecnorm(obstacles - obj.H * s, 2, 1);
                %sigma=1;
                %obstacle_cost = sum(exp(-vecnorm(o_pred - obstacles, 2, 1).^2 / (2 * sigma^2)));
                obstacle_cost = 1 / (min(dists) + 1e-3);
                EFE_total(i) = (efe/horizon) +  obstacle_cost;
            end

            [~, idx] = min(EFE_total);
            a = obj.action_space(:, idx);
        end

        function update_dynamics(obj, A_new, B_new)
            % Optional: update generative dynamics
            obj.A = A_new;
            obj.B = B_new;
        end

        function set_goal(obj, g)
            obj.goal = g(:);
        end

        function [s_next, o] = environment_step(~, s, a, A_env, B_env, H_env, Q_env, R_env)
            % Simulate true environment dynamics and observations
            w = mvnrnd(zeros(6,1), Q_env)';
            s_next = A_env * s + B_env * a + w;

            e = mvnrnd(zeros(3,1), R_env)';
            o = H_env * s_next + e;
        end
    end
end