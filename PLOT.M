% plot_mixed_init.m
% Plots the results of the mixed-initialized run, now including NN outputs

clear; clc;

% Load data
robot_data  = load('x_robot');   % columns: [t, e1, e2, ep1, ep2, nn1, nn2, r1, r2, qd1, qd2, x1, x2, tau1, tau2, ...]
wei_v_data  = load('x_wei_v');
wei_w_data  = load('x_wei_w');

% --- Extract time and signals ---
t    = robot_data(:,1);   % time [s]
e1   = robot_data(:,2);   % tracking error joint 1 [rad]
e2   = robot_data(:,3);   % tracking error joint 2 [rad]
nn1  = robot_data(:,6);   % neural network output 1
nn2  = robot_data(:,7);   % neural network output 2
tau1 = robot_data(:,14);  % control torque joint 1 [Nm]
tau2 = robot_data(:,15);  % control torque joint 2 [Nm]

% weights are logged every 10 steps: dt=0.001s => 0.01s interval
t_w  = (0:size(wei_v_data,1)-1) * 0.01;  % Adjust t_w length to match wei_v_data

% --- 1. Tracking Error ---
figure(1); clf;
plot(t, e1, 'b-', t, e2, 'r-', 'LineWidth', 1.2);
grid on;
xlabel('Time (s)');
ylabel('Tracking Error (rad)');
legend('e_1', 'e_2', 'Location', 'Best');
title('Joint Tracking Error (Mixed Initialization)');
set(gcf, 'Position', [100, 100, 800, 400]);

% --- 2. Control Torques ---
figure(2); clf;
plot(t, tau1, 'b-', t, tau2, 'r-', 'LineWidth', 1.2);
grid on;
xlabel('Time (s)');
ylabel('Control Torque (Nm)');
legend('\tau_1', '\tau_2', 'Location', 'Best');
title('Control Torques (Mixed Initialization)');
set(gcf, 'Position', [100, 100, 800, 400]);

% --- 3. Input→Hidden Weights ---
figure(3); clf;
plot(t_w, wei_v_data(:,1), '-o', ...
     t_w, wei_v_data(:,2), '-x', ...
     t_w, wei_v_data(:,3), '-s', ...
     t_w, wei_v_data(:,4), '-d', 'LineWidth', 1.2);
grid on;
xlabel('Time (s)');
ylabel('Weight value');
legend('w_{v21}', 'w_{v60}', 'w_{v22}', 'w_{v61}', 'Location', 'Best');
title('Input→Hidden Weights (Mixed Initialization)');
set(gcf, 'Position', [100, 100, 800, 400]);

% --- 4. Hidden→Output Weights ---
figure(4); clf;
plot(t_w, wei_w_data(:,1), '-o', ...
     t_w, wei_w_data(:,2), '-x', ...
     t_w, wei_w_data(:,3), '-s', ...
     t_w, wei_w_data(:,4), '-d', ...
     t_w, wei_w_data(:,5), '-^', ...
     t_w, wei_w_data(:,6), '-v', 'LineWidth', 1.2);
grid on;
xlabel('Time (s)');
ylabel('Weight value');
legend('w_{w1}', 'w_{w17}', 'w_{w5}', 'w_{w21}', 'w_{w10}', 'w_{w26}', 'Location', 'Best');
title('Hidden→Output Weights (Mixed Initialization)');
set(gcf, 'Position', [100, 100, 800, 400]);

% --- 5. Neural Network Outputs ---
figure(5); clf;
plot(t, nn1, 'b-', t, nn2, 'r-', 'LineWidth', 1.2);
grid on;
xlabel('Time (s)');
ylabel('NN Output');
legend('y_1', 'y_2', 'Location', 'Best');
title('Neural Network Outputs (Mixed Initialization)');
set(gcf, 'Position', [100, 100, 800, 400]);
