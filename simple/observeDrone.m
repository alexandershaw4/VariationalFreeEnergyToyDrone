function o = observeDrone(s, H, R)
% observeDrone - Generate noisy observation from state
%
% Inputs:
%   s - true latent state [x; y; z; vx; vy; vz]
%   H - observation matrix (e.g., [I3, 03]) to extract [x; y; z]
%   R - observation noise covariance (3x3)
%
% Output:
%   o - noisy observation (e.g., observed position)

% Observation noise
e = mvnrnd(zeros(size(R,1),1), R)';

% Observation generation
o = H * s + e;
end
