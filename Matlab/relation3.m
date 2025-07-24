clc;
close all;
clear;

%% === Plot Settings and Color Schemes ===
boxFillColor = [255, 173, 173] / 255;        % Box fill color (light red)
borderColor = [146, 0, 0] / 255;             % Border color (dark red)
outlierColor = [255, 173, 173] / 255;        % Outlier color (solid red)

%% === Load Data: Accelerometer ===
data = readtable('C:\Users\LWJ\Desktop\XiAn1.csv');  % Replace with actual path
accel_vibration = data{:, 24};   % Column 24: Accelerometer vibration level
power = data{:, 20};             % Column 20: Power

% Define sliding window steps
M1 = [5, 10, 20, 40, 80];

figure;
% Layout settings
xx = [0.035, 0.235, 0.435, 0.635, 0.835];  % X positions for subplots
yy = [0.74];                               % Y position for accelerometer row
width = 0.15;
height = 0.23;

% === Accelerometer Boxplots ===
for kk = 1:length(M1)
    M = M1(kk);
    numBlocks = floor(length(accel_vibration) / M);
    groupedPower = cell(1, 10);  % 10 bins for vibration level [0~9]

    for i = 1:numBlocks
        startIdx = (i - 1) * M + 1;
        endIdx = i * M;
        avgVibration = mean(accel_vibration(startIdx:endIdx));
        binIdx = min(10, max(1, floor(avgVibration + 0.5) + 1));  % bin to [1~10]
        groupedPower{binIdx} = [groupedPower{binIdx}, mean(power(startIdx:endIdx))];
    end

    % Pad groups with NaN for consistent boxplot dimensions
    maxLen = max(cellfun(@length, groupedPower));
    for k = 1:10
        groupedPower{k}(end+1:maxLen) = NaN;
    end

    subplot('Position', [xx(kk), yy(1), width, height]);
    boxplot(cell2mat(groupedPower'), ...
        'Labels', {'0','1','2','3','4','5','6','7','8','9'}, ...
        'Symbol', 'o', 'OutlierSize', 3, 'Colors', borderColor);

    % Customize outliers
    outliers = findobj(gca, 'Tag', 'Outliers');
    set(outliers, 'MarkerEdgeColor', outlierColor, 'MarkerFaceColor', outlierColor);

    % Customize box fill
    boxes = findobj(gca, 'Tag', 'Box');
    for i = 1:length(boxes)
        patch(boxes(i).XData, boxes(i).YData, boxFillColor, ...
              'FaceAlpha', 0.5, 'EdgeColor', borderColor, 'LineWidth', 1.2);
    end

    % Axis styling
    ax = gca;
    ax.LineWidth = 1.1;
    ax.FontSize = 20;
    ax.FontName = 'Times New Roman';
    ax.XTickLabel = {'0','1','2','3','4','5','6','7','8','9'};
    ax.Title.String = ['Step=' num2str(M)];
    ax.Title.FontSize = 20;
    ax.YLabel.String = 'Power';
    ax.YLabel.FontSize = 26;
    ax.Box = 'off';
    ax.XColor = 'k';
    ax.YColor = 'k';
end

% Title annotation
annotation('textbox', [0.1, 0.66, 0.8, 0.05], 'String', '(d) Accelerometer Vibration Level', ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 30);

%% === Load Data: Gyroscope ===
data = readtable('C:\Users\LWJ\Desktop\XiAn1.csv');  % Same file reused
gyro_x = data{:, 25};  
gyro_y = data{:, 25};  
gyro_z = data{:, 25};  
power = data{:, 20};  

% Color scheme for gyroscope plot
boxFillColor = [255, 223, 165] / 255;   % Light yellow
borderColor  = [127, 81, 0] / 255;      % Brown
outlierColor = [255, 223, 165] / 255;

yy = [0.41];  % Y position for gyroscope row

% === Gyroscope Boxplots ===
for kk = 1:length(M1)
    M = M1(kk);
    numBlocks = floor(length(gyro_x) / M);
    groupedPower = cell(1, 10);  % 10 bins for vibration level

    for i = 1:numBlocks
        startIdx = (i - 1) * M + 1;
        endIdx = i * M;
        avgGyro = mean([gyro_x(startIdx:endIdx), ...
                        gyro_y(startIdx:endIdx), ...
                        gyro_z(startIdx:endIdx)], 2);
        avgGyro = mean(avgGyro);
        binIdx = min(10, max(1, floor(avgGyro * 10) + 1));  % scaled binning
        groupedPower{binIdx} = [groupedPower{binIdx}, mean(power(startIdx:endIdx))];
    end

    maxLen = max(cellfun(@length, groupedPower));
    for k = 1:10
        groupedPower{k}(end+1:maxLen) = NaN;
    end

    subplot('Position', [xx(kk), yy(1), width, height]);
    boxplot(cell2mat(groupedPower'), ...
        'Labels', {'0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9'}, ...
        'Symbol', 'o', 'OutlierSize', 3, 'Colors', borderColor);

    outliers = findobj(gca, 'Tag', 'Outliers');
    set(outliers, 'MarkerEdgeColor', outlierColor, 'MarkerFaceColor', outlierColor);

    boxes = findobj(gca, 'Tag', 'Box');
    for i = 1:length(boxes)
        patch(boxes(i).XData, boxes(i).YData, boxFillColor, ...
              'FaceAlpha', 0.5, 'EdgeColor', borderColor, 'LineWidth', 1.2);
    end

    ax = gca;
    ax.LineWidth = 1.1;
    ax.FontSize = 20;
    ax.FontName = 'Times New Roman';
    ax.XTickLabel = {'0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9'};
    ax.Title.String = ['Step=' num2str(M)];
    ax.Title.FontSize = 20;
    ax.YLabel.String = 'Power';
    ax.YLabel.FontSize = 26;
    ax.Box = 'off';
    ax.XColor = 'k';
    ax.YColor = 'k';
end

% Title annotation
annotation('textbox', [0.1, 0.32, 0.8, 0.05], 'String', '(e) Gyroscope Vibration Level', ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 30);

