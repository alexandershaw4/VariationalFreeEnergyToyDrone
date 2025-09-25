function s_next = simulateDroneDynamics(s, a, A, B, Q)
% simulateDroneDynamics - Simulate 3D drone motion with control input and process noise
%
% Inputs:
%   s - current state vector [x; y; z; vx; vy; vz]
%   a - control input [ax; ay; az]
%   A - dynamics matrix (6x6)
%   B - control matrix (6x3)
%   Q - process noise covariance (6x6)
%
% Output:
%   s_next - next state after applying action and noise

% Process noise
w = mvnrnd(zeros(6,1), Q)';

% State update
s_next = A * s + B * a + w;
end
