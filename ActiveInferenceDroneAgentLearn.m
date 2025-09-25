% ActiveInferenceDroneAgent.m
% Agent-centric Active Inference framework (from scratch)
% The agent does not know the true state or true dynamics.

classdef ActiveInferenceDroneAgentLearn < handle
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

        % Learning rate for dynamics (used only if enabled)
        learning_rate = 0.01;
    end

    methods
        function obj = ActiveInferenceDroneAgentLearn()
            % Initial belief
            obj.mu = zeros(6,1);
            obj.Sigma = 1.0 * eye(6);

            % Default generative model (linear)
            dt = 0.1;
            obj.A = [1 0 0 dt 0  0;
                     0 1 0 0  dt 0;
                     0 0 1 0  0  dt;
                     0 0 0 1  0  0;
                     0 0 0 0  1  0;
                     0 0 0 0  0  1];

            obj.B = [0.5*dt^2 * eye(3); dt * eye(3)];
            obj.H = [eye(3), zeros(3)];
            obj.R = 1e-2 * eye(3);
            obj.Q = 1e-3 * eye(6);

            % Default goal
            obj.goal = [5; 5; 5];

            % Default action space
            vals = [-1, 0, 1];
            [Ax, Ay, Az] = ndgrid(vals, vals, vals);
            obj.action_space = [Ax(:)'; Ay(:)'; Az(:)'];
        end

        function infer_state(obj, observation)
            % Inference via variational Laplace
            f = @(s) obj.H * s;
            [m, V, ~, ~, ~, ~, ~] = fitVariationalLaplaceThermo(observation, f, obj.mu, obj.Sigma, obj.maxIter, obj.tol, obj.doPlot);
            obj.mu = m;
            obj.Sigma = V*V';  % reconstruction
        end

        function a = select_action_rollout(obj, horizon)
            % Multi-step EFE-based action selection
            nA = size(obj.action_space, 2);
            EFE_total = zeros(1, nA);

            for i = 1:nA
                a1 = obj.action_space(:, i);
                s = obj.A * obj.mu + obj.B * a1;
                efe = norm(obj.H * s - obj.goal)^2;

                % Roll out additional steps
                for h = 2:horizon
                    a_h = obj.action_space(:, randi(nA));  % random action for simplicity
                    s = obj.A * s + obj.B * a_h;
                    efe = efe + norm(obj.H * s - obj.goal)^2;
                end
                EFE_total(i) = efe;
            end

            [~, idx] = min(EFE_total);
            a = obj.action_space(:, idx);
        end

        function update_dynamics(obj, s_prev, s_post, a)
            % Optional: disabled learning by default
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
