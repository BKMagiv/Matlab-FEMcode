% 2D Allen-Cahn equation
% ut + Au = lambda * f(u)
%          Space
% (0,1) ----------- (1,1)
%   |                 |
%   |                 |
%   |                 |
%   |                 |
%   |                 |
% (0,0) ----------- (1,0)
% time : [0,T]

clc;clear;
format long;

% space
X = 1.0; Y = 1.0;
% time
Tmax = 1.0;

% nonlinearity
lambda = 4.0;
f = @(u) lambda * (u - u.^3);
% initial condition
phi = @(x,y) sin(pi * x) .* sin(pi * y);

% spatial step
N = 36;
Nx = N; Ny = N;
% number of nodes
node = (Nx + 1) * (Ny + 1);

% degree of a polynomial: P1, P2 or P3
deg = "P1";
% node coordinates
P = mesh2D(Nx,Ny,[0 X],[0 Y]);
% Finite element cell
T = connect(Nx,Ny,deg);

% boundary node
B = boundary(Nx,Ny);
Lb = length(B);

% Laplace matrix & Mass matrix
A = Laplace(P,T); K = Mass(P,T);

% time step: fixed step size & IMEX Euler Method
M = 1024;
k = Tmax / M;

% numerical solution
U = zeros(node,M + 1);
U(:,1) = phi(P(:,1),P(:,2));

% solver
for j = 2 : M + 1
b = K * U(:,j - 1) + k * Nonlinearity(f,U(:,j - 1),P,T);
Stiff = K + k * A;

% boundary conditions
end
