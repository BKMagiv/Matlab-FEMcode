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
  for p = 1 : Lb
    Stiff(B(p),:) = 0;
    Stiff(B(p),:) = 0;
    Stiff(B(p),B(p)) = 1;
    b(B(p)) = 0;
  end
  
  U(:,j) = Stiff \ b;
end

% make gif
figure(1)
filename = 'AC.gif';
[meshX,meshY] = meshgrid(linspace(0,X,Nx + 1)',linspace(0,Y,Ny + 1)');
for j = 1 : M + 1
    uh = reshape(U(:,j),Ny + 1,Nx + 1);
    
    contourf(meshX,meshY,uh), axis([0 X 0 Y 0 1]), drawnow;

    im = frame2im(getframe(gcf));
    [a,map] = rgb2ind(im,256);
    if j == 1
        imwrite(a,map,filename,'gif','LoopCount',Inf,'DelayTime',0.1);
    else
        imwrite(a,map,filename,'gif','WriteMode','append','DelayTime',0.1);
    end
end

% function
function P = mesh2D(Nx,Ny,X,Y)
    node = (Nx + 1) * (Ny + 1);
    P = zeros(node,2);
    x = linspace(X(1),X(end),Nx + 1);
    y = linspace(Y(1),Y(end),Ny + 1);
    for j = 1 : Ny + 1
        for i = 1 : Nx + 1
            P(i + (j - 1) * (Nx + 1),1) = x(i);
            P(i + (j - 1) * (Nx + 1),2) = y(j);
        end
    end
end

function B = boundary(Nx,Ny)
    B = zeros(2 * (Nx + Ny),1);
    A = reshape(1 : (Nx + 1) * (Ny + 1),Nx + 1,Ny + 1)';
    l = 1;
    for j = 1 : Ny + 1
        for i = 1 : Nx + 1
            if j == 1 || i == 1 || j == Ny + 1 || i == Nx + 1 
                B(l) = A(j,i);
                l = l + 1;
            end
        end
    end
end

function T = connect(Nx,Ny,deg)
    node = (Nx + 1) * (Ny + 1);
    A = reshape(1 : node,Nx + 1,Ny + 1)';
    if deg == "P1"
        num = 2 * Nx * Ny;
        T = zeros(num,3);
        num = num / 2;
        for n = 1 : num
            xe = rem(n,Nx);
            if xe == 0
                xe = Nx;
            end
            ye = ceil(n / Nx);
            temp = A(ye : ye + 1,xe : xe + 1)';
            T(2 * n - 1,:) = temp([1 2 3]);
            T(2 * n,:) = temp([3 2 4]);
        end
    elseif deg == "P2"
        num = (Nx * Ny) / 2;
        T = zeros(num,6);
        num = num / 2;
        for n = 1 : num
            xe = rem(2 * n,Nx);
            if xe == 0
                xe = Nx;
            end
            ye =  2 * ceil(2 * n / Nx);
            temp = A(ye - 1 : ye + 1,xe - 1 : xe + 1)';
            T(2 * n - 1,:) = temp([1 3 7 2 5 4]);
            T(2 * n,:) = temp([7 3 9 5 6 8]);
        end
    elseif deg == "P3"
        num = (2 * Nx * Ny) / 9;
        T = zeros(num,10);
        num = num / 2;
        for n = 1 : num
            xe = rem(3 * n - 2,Nx);
            if xe == 0
                xe = Nx;
            end
            ye =  3 * ceil(3 * n / Nx) - 2;
            temp = A(ye : ye + 3,xe : xe + 3)';
            T(2 * n - 1,:) = temp([1 4 13 2 7 9 3 10 5 6]);
            T(2 * n,:) = temp([13 4 16 10 8 15 7 12 14 11]);
        end
    end
end

function phi = base_elements(xi,eta,Jacobi,dx,dy,deg)
    invJ = inv(Jacobi);
    if deg == "P1"
        phi = zeros(3,1);
        if dx == 0 && dy == 0
            phi(1) = 1 - xi - eta;
            phi(2) = xi;
            phi(3) = eta;
        elseif dx == 1 && dy == 0
            phi(1) = -invJ(1,1) - invJ(2,1);
            phi(2) =  invJ(1,1);
            phi(3) =  invJ(2,1);
        elseif dx == 0 && dy == 1
            phi(1) = -invJ(1,2) - invJ(2,2);
            phi(2) =  invJ(1,2);
            phi(3) =  invJ(2,2);
        end
    elseif deg == "P2"
        phi = zeros(6,1);
        if dx == 0 && dy == 0
            phi(1) = 2 * xi^2 + 2 * eta^2 + 4 * xi * eta - 3 * xi - 3 * eta + 1;
            phi(2) = 2 * xi^2 - xi;
            phi(3) = 2 * eta^2 - eta;
            phi(4) = -4 * xi^2 - 4 * xi * eta + 4 * xi;
            phi(5) = 4 * xi * eta;
            phi(6) = -4 * eta^2 - 4 * xi * eta + 4 * eta;
        elseif dx == 1 && dy == 0
            phi(1) = invJ(1,1) * (4 * xi + 4 * eta - 3) + invJ(2,1) * (4 * eta + 4 * xi - 3);
            phi(2) = invJ(1,1) * (4 * xi - 1);
            phi(3) = invJ(2,1) * (4 * eta - 1);
            phi(4) = invJ(1,1) * (4 - 4 * eta - 8 * xi) + invJ(2,1) * (-4 * xi);
            phi(5) = invJ(1,1) * (4 * eta) + invJ(2,1) * (4 * xi);
            phi(6) = invJ(1,1) * (-4 * eta) + invJ(2,1) * (4 - 4 * xi - 8 * eta);
        elseif dx == 0 && dy == 1
            phi(1) = invJ(1,2) * (4 * xi + 4 * eta - 3) + invJ(2,2) * (4 * eta + 4 * xi - 3);
            phi(2) = invJ(1,2) * (4 * xi - 1);
            phi(3) = invJ(2,2) * (4 * eta - 1);
            phi(4) = invJ(1,2) * (4 - 4 * eta - 8 * xi) + invJ(2,2) * (-4 * xi);
            phi(5) = invJ(1,2) * (4 * eta) + invJ(2,2) * (4 * xi);
            phi(6) = invJ(1,2) * (-4 * eta) + invJ(2,2) * (4 - 4 * xi - 8 * eta);
        end
    elseif deg == "P3"
        phi = zeros(10,1);
        if dx == 0 && dy == 0
            phi(1) = 1 - 5.5 * xi - 5.5 * eta + 9 * xi^2 + 18 * xi * eta + 9 * eta^2 - 4.5 * xi^3 - 13.5 * xi^2 * eta - 13.5 * xi * eta^2 - 4.5 * eta^3;
            phi(2) = xi - 4.5 * xi^2 + 4.5 * xi^3;
            phi(3) = eta - 4.5 * eta^2 + 4.5 * eta^3;
            phi(4) = 9 * xi - 22.5 * xi^2 - 22.5 * xi * eta + 13.5 * xi^3 + 27 * xi^2 * eta + 13.5 * xi * eta^2;
            phi(5) = -4.5 * xi * eta + 13.5 * xi^2 * eta;
            phi(6) = -4.5 * eta + 4.5 * xi * eta + 18 * eta^2 - 13.5 * xi * eta^2 - 13.5 * eta^3;
            phi(7) = -4.5 * xi + 18 * xi^2 + 4.5 * xi * eta - 13.5 * xi^3 - 13.5 * xi^2 * eta;
            phi(8) = -4.5 * xi * eta + 13.5 * xi * eta^2;
            phi(9) = 9 * eta - 22.5 * xi * eta - 22.5 * eta^2 + 13.5 * xi^2 * eta + 27 * xi * eta^2 + 13.5 * eta^3;
            phi(10) = 27 * xi * eta - 27 * xi^2 * eta - 27 * xi * eta^2;
        elseif dx == 1 && dy == 0
            phi(1) = invJ(1,1) * (-5.5 + 18 * xi + 18 * eta - 13.5 * xi^2 - 27 * xi * eta - 13.5 * eta^2) ...
                   + invJ(2,1) * (-5.5 + 18 * xi + 18 * eta - 13.5 * xi^2 - 27 * xi * eta - 13.5 * eta^2);
            phi(2) = invJ(1,1) * (1 - 9 * xi + 13.5 * xi^2);
            phi(3) = invJ(2,1) * (1 - 9 * eta + 13.5 * eta^2);
            phi(4) = invJ(1,1) * (9 - 45 * xi - 22.5 * eta + 40.5 * xi^2 + 54 * xi * eta + 13.5 * eta^2) ...
                   + invJ(2,1) * (-22.5 * xi + 27 * xi^2 + 27 * xi * eta);
            phi(5) = invJ(1,1) * (-4.5 * eta + 27 * xi * eta) ...
                   + invJ(2,1) * (-4.5 * xi + 13.5 * xi^2);
            phi(6) = invJ(1,1) * (4.5 * eta - 13.5 * eta^2) ...
                   + invJ(2,1) * (-4.5 + 4.5 * xi + 36 * eta - 27 * xi * eta - 40.5 * eta^2);
            phi(7) = invJ(1,1) * (-4.5 + 36 * xi + 4.5 * eta - 40.5 * xi^2 - 27 * xi * eta) ...
                   + invJ(2,1) * (4.5 * xi - 13.5 * xi^2);
            phi(8) = invJ(1,1) * (-4.5 * eta + 13.5 * eta^2) ...
                   + invJ(2,1) * (-4.5 * xi + 27 * xi * eta);
            phi(9) = invJ(1,1) * (-22.5 * eta + 27 * xi * eta + 27 * eta^2) ...
                   + invJ(2,1) * (9 - 22.5 * xi - 45 * eta + 13.5 * xi^2 + 54 * xi * eta + 40.5 * eta^2);
            phi(10) = invJ(1,1) * (27 * eta - 54 * xi * eta - 27 * eta^2) ...
                    + invJ(2,1) * (27 * xi - 27 * xi^2 - 54 * xi * eta);
        elseif dx == 0 && dy == 1
            phi(1) = invJ(1,2) * (-5.5 + 18 * xi + 18 * eta - 13.5 * xi^2 - 27 * xi * eta - 13.5 * eta^2) ...
                   + invJ(2,2) * (-5.5 + 18 * xi + 18 * eta - 13.5 * xi^2 - 27 * xi * eta - 13.5 * eta^2);
            phi(2) = invJ(1,2) * (1 - 9 * xi + 13.5 * xi^2);
            phi(3) = invJ(2,2) * (1 - 9 * eta + 13.5 * eta^2);
            phi(4) = invJ(1,2) * (9 - 45 * xi - 22.5 * eta + 40.5 * xi^2 + 54 * xi * eta + 13.5 * eta^2) ...
                   + invJ(2,2) * (-22.5 * xi + 27 * xi^2 + 27 * xi * eta);
            phi(5) = invJ(1,2) * (-4.5 * eta + 27 * xi * eta) ...
                   + invJ(2,2) * (-4.5 * xi + 13.5 * xi^2);
            phi(6) = invJ(1,2) * (4.5 * eta - 13.5 * eta^2) ...
                   + invJ(2,2) * (-4.5 + 4.5 * xi + 36 * eta - 27 * xi * eta - 40.5 * eta^2);
            phi(7) = invJ(1,2) * (-4.5 + 36 * xi + 4.5 * eta - 40.5 * xi^2 - 27 * xi * eta) ...
                   + invJ(2,2) * (4.5 * xi - 13.5 * xi^2);
            phi(8) = invJ(1,2) * (-4.5 * eta + 13.5 * eta^2) ...
                   + invJ(2,2) * (-4.5 * xi + 27 * xi * eta);
            phi(9) = invJ(1,2) * (-22.5 * eta + 27 * xi * eta + 27 * eta^2) ...
                   + invJ(2,2) * (9 - 22.5 * xi - 45 * eta + 13.5 * xi^2 + 54 * xi * eta + 40.5 * eta^2);
            phi(10) = invJ(1,2) * (27 * eta - 54 * xi * eta - 27 * eta^2) ...
                    + invJ(2,2) * (27 * xi - 27 * xi^2 - 54 * xi * eta);
        end
    end
end

function [Gauss_point,Gauss_weight] = Gauss()
    Gauss_point = [-sqrt(3/5) -sqrt(3/5);0 -sqrt(3/5);sqrt(3/5) -sqrt(3/5);
                   -sqrt(3/5)  0        ;0  0        ;sqrt(3/5)          0;
                   -sqrt(3/5)  sqrt(3/5);0  sqrt(3/5);sqrt(3/5)  sqrt(3/5)];
    Gauss_weight = [25;40;25;
                    40;64;40;
                    25;40;25] ./ 81;
end

function A = Laplace(P,T)
    node = length(P(:,1));
    num = length(T(:,1));
    if length(T(1,:)) == 3
        deg = "P1";
    elseif length(T(1,:)) == 6
        deg = "P2";
    elseif length(T(1,:)) == 10
        deg = "P3";
    end
    [point,weight] = Gauss();
    L = length(weight);

    A = zeros(node);
    for n = 1 : num
        Jacobi = [P(T(n,2),1) - P(T(n,1),1),P(T(n,3),1) - P(T(n,1),1);
                  P(T(n,2),2) - P(T(n,1),2),P(T(n,3),2) - P(T(n,1),2)];
        for l = 1 : L
            xi = 0.5 * (point(l,1) + 1); eta = 0.25 * (1 - point(l,1)) * (1 + point(l,2));
            A(T(n,:),T(n,:)) = A(T(n,:),T(n,:)) + ...
            0.125 * det(Jacobi) * weight(l) * (1 - point(l,1)) * ...
            (base_elements(xi,eta,Jacobi,1,0,deg) * base_elements(xi,eta,Jacobi,1,0,deg)' + ...
             base_elements(xi,eta,Jacobi,0,1,deg) * base_elements(xi,eta,Jacobi,0,1,deg)');
        end
    end
    A = sparse(A);
end

function K = Mass(P,T)
    node = length(P(:,1));
    num = length(T(:,1));
    if length(T(1,:)) == 3
        deg = "P1";
    elseif length(T(1,:)) == 6
        deg = "P2";
    elseif length(T(1,:)) == 10
        deg = "P3";
    end
    [point,weight] = Gauss();
    L = length(weight);

    K = zeros(node);
    for n = 1 : num
        Jacobi = [P(T(n,2),1) - P(T(n,1),1),P(T(n,3),1) - P(T(n,1),1);
                  P(T(n,2),2) - P(T(n,1),2),P(T(n,3),2) - P(T(n,1),2)];
        for l = 1 : L
            xi = 0.5 * (point(l,1) + 1); eta = 0.25 * (1 - point(l,1)) * (1 + point(l,2));
            K(T(n,:),T(n,:)) = K(T(n,:),T(n,:)) + ...
            0.125 * det(Jacobi) * weight(l) * (1 - point(l,1)) * ...
            (base_elements(xi,eta,Jacobi,0,0,deg) * base_elements(xi,eta,Jacobi,0,0,deg)');
        end
    end
    K = sparse(K);
end

function b = Nonlinearity(B,u,P,T)
    node = length(P(:,1));
    num = length(T(:,1));
    if length(T(1,:)) == 3
        deg = "P1";
    elseif length(T(1,:)) == 6
        deg = "P2";
    elseif length(T(1,:)) == 10
        deg = "P3";
    end
    [point,weight] = Gauss();
    L = length(weight);

    b = zeros(node,1);
    for n = 1 : num
        Jacobi = [P(T(n,2),1) - P(T(n,1),1),P(T(n,3),1) - P(T(n,1),1);
                  P(T(n,2),2) - P(T(n,1),2),P(T(n,3),2) - P(T(n,1),2)];
        for l = 1 : L
            xi = 0.5 * (point(l,1) + 1); eta = 0.25 * (1 - point(l,1)) * (1 + point(l,2));
            uh = base_elements(xi,eta,Jacobi,0,0,deg)' * u(T(n,:));
            b(T(n,:)) = b(T(n,:)) + 0.125 * det(Jacobi) * weight(l) * (1 - point(l,1)) * ...
                        B(uh) * base_elements(xi,eta,Jacobi,0,0,deg);
        end
    end
end

function b = Function(g,t,P,T)
    node = length(P(:,1));
    num = length(T(:,1));
    if length(T(1,:)) == 3
        deg = "P1";
    elseif length(T(1,:)) == 6
        deg = "P2";
    elseif length(T(1,:)) == 10
        deg = "P3";
    end
    [point,weight] = Gauss();
    L = length(weight);

    b = zeros(node,1);
    for n = 1 : num
        Jacobi = [P(T(n,2),1) - P(T(n,1),1),P(T(n,3),1) - P(T(n,1),1);
                  P(T(n,2),2) - P(T(n,1),2),P(T(n,3),2) - P(T(n,1),2)];
        for l = 1 : L
            xi = 0.5 * (point(l,1) + 1); eta = 0.25 * (1 - point(l,1)) * (1 + point(l,2));
            x = P(T(n,1),1) + Jacobi(1,1) * xi + Jacobi(1,2) * eta;
            y = P(T(n,1),2) + Jacobi(2,1) * xi + Jacobi(2,2) * eta;
            b(T(n,:)) = b(T(n,:)) + 0.125 * det(Jacobi) * weight(l) * (1 - point(l,1)) * ...
                        g(x,y,t) * base_elements(xi,eta,Jacobi,0,0,deg);
        end
    end
end

function I = NORM(u,P,T,Space)
    num = length(T(:,1));
    if length(T(1,:)) == 3
        deg = "P1";
    elseif length(T(1,:)) == 6
        deg = "P2";
    elseif length(T(1,:)) == 10
        deg = "P3";
    end
    [point,weight] = Gauss();
    L = length(weight);

    I = 0.0;
    if Space == "L2"
        for n = 1 : num
            Jacobi = [P(T(n,2),1) - P(T(n,1),1),P(T(n,3),1) - P(T(n,1),1);
                      P(T(n,2),2) - P(T(n,1),2),P(T(n,3),2) - P(T(n,1),2)];
            for l = 1 : L
                xi = 0.5 * (1 + point(l,1)); eta = 0.25 * (1 - point(l,1)) * (1 + point(l,2));
                uh = base_elements(xi,eta,Jacobi,0,0,deg)' * u(T(n,:));
                I = I + 0.125 * det(Jacobi) * weight(l) * (1 - point(l,1)) * uh^2;
            end
        end
    elseif Space == "H1"
        for n = 1 : num
            Jacobi = [P(T(n,2),1) - P(T(n,1),1),P(T(n,3),1) - P(T(n,1),1);
                      P(T(n,2),2) - P(T(n,1),2),P(T(n,3),2) - P(T(n,1),2)];
            for l = 1 : L
                xi = 0.5 * (1 + point(l,1)); eta = 0.25 * (1 - point(l,1)) * (1 + point(l,2));
                uhx = base_elements(xi,eta,Jacobi,1,0,deg)' * u(T(n,:));
                uhy = base_elements(xi,eta,Jacobi,0,1,deg)' * u(T(n,:));
                I = I + 0.125 * det(Jacobi) * weight(l) * (1 - point(l,1)) * (uhx^2 + uhy^2);
            end
        end
    end
    I = sqrt(I);
end
