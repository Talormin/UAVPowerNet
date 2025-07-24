clc; close all; clear;

% Custom color settings for different features
boxFillColorEuler = [169, 252, 255] / 255;
borderColorEuler   = [0, 134, 138] / 255;

boxFillColorSpeed = [103, 136, 247] / 255;
borderColorSpeed   = [0, 32, 140] / 255;

boxFillColorDisp = [238, 155, 247] / 255;
borderColorDisp   = [125, 0, 138] / 255;

% Step sizes for sliding window
M1 = [5, 10, 20, 40, 80];
cols = 5; % Number of subplot columns
width = 0.15;
height = 0.23;
xx = [0.035, 0.235, 0.435, 0.635, 0.835];

%% --------- (a) Euler Angles Analysis ---------
data = readtable('your_path/ChengDu.csv');  % Replace with actual path
euler_x = data{:, 16}; 
euler_y = data{:, 16}; 
euler_z = data{:, 16}; 
power = data{:, 23};
yy = [0.74]; % Y-position for subplot row

figure;

for kk = 1:length(M1)
    M = M1(kk);
    numBlocks = floor(length(power) / M);
    groupedPower = cell(1, 5);

    for i = 1:numBlocks
        idx = (i-1)*M+1 : i*M;
        avgEuler = mean([euler_x(idx), euler_y(idx), euler_z(idx)], 2);
        avgEuler = mean(avgEuler);

        % Bin into categories based on average Euler angle
        if avgEuler <= -1
            groupIdx = 1;
        elseif avgEuler < 0
            groupIdx = 2;
        elseif avgEuler < 1
            groupIdx = 3;
        elseif avgEuler < 2
            groupIdx = 4;
        else
            groupIdx = 5;
        end

        groupedPower{groupIdx} = [groupedPower{groupIdx}, mean(power(idx))];
    end

    % Pad each group with NaNs for alignment
    maxLen = max(cellfun(@length, groupedPower));
    for k = 1:5
        groupedPower{k}(end+1:maxLen) = NaN;
    end

    % Draw subplot
    subplot('Position', [xx(kk), yy(1), width, height]);
    boxplot(cell2mat(groupedPower'), ...
        'Labels', {'-1', '0', '1', '2', '3'}, ...
        'Symbol', 'o', 'OutlierSize', 3, 'Colors', borderColorEuler);

    % Customize outliers and boxes
    outliers = findobj(gca, 'Tag', 'Outliers');
    set(outliers, 'MarkerEdgeColor', boxFillColorEuler, 'MarkerFaceColor', boxFillColorEuler);

    boxes = findobj(gca, 'Tag', 'Box');
    for b = 1:length(boxes)
        patch(boxes(b).XData, boxes(b).YData, boxFillColorEuler, ...
              'FaceAlpha', 0.5, 'EdgeColor', borderColorEuler, 'LineWidth', 1.2);
    end

    % Axis styling
    ax = gca;
    ax.LineWidth = 1.1;
    ax.FontSize = 20;
    ax.FontName = 'Times New Roman';
    ax.Box = 'off';
    ax.XColor = 'k'; ax.YColor = 'k';
    ax.Title.String = ['Step=' num2str(M)];
    ax.Title.FontSize = 20;
    ax.YLabel.String = 'Power';
    ax.YLabel.FontSize = 24;
end

% Euler angle annotation
annotation('textbox', [0.1, 0.66, 0.8, 0.05], 'String', '(a) Euler Angles (rad)', ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 30);

%% --------- (b) Velocity Analysis ---------
data = readtable('your_path/Korea.csv');
vel_x = data{:, 13}; 
vel_y = data{:, 14}; 
vel_z = data{:, 15}; 
power = data{:, 23};
yy = [0.41];

for kk = 1:length(M1)
    M = M1(kk);
    numBlocks = floor(length(power) / M);
    groupedPower = cell(1, 5);

    for i = 1:numBlocks
        idx = (i-1)*M+1 : i*M;
        avgVel = mean([vel_x(idx), vel_y(idx), vel_z(idx)], 2);
        avgVel = mean(avgVel);

        if avgVel <= -1
            groupIdx = 1;
        elseif avgVel < -0.25
            groupIdx = 2;
        elseif avgVel < 0.25
            groupIdx = 3;
        elseif avgVel < 0.75
            groupIdx = 4;
        else
            groupIdx = 5;
        end

        groupedPower{groupIdx} = [groupedPower{groupIdx}, mean(power(idx))];
    end

    maxLen = max(cellfun(@length, groupedPower));
    for k = 1:5
        groupedPower{k}(end+1:maxLen) = NaN;
    end

    subplot('Position', [xx(kk), yy(1), width, height]);
    boxplot(cell2mat(groupedPower'), ...
        'Labels', {'-1', '-0.5', '0', '0.5', '1'}, ...
        'Symbol', 'o', 'OutlierSize', 3, 'Colors', borderColorSpeed);

    outliers = findobj(gca, 'Tag', 'Outliers');
    set(outliers, 'MarkerEdgeColor', boxFillColorSpeed, 'MarkerFaceColor', boxFillColorSpeed);

    boxes = findobj(gca, 'Tag', 'Box');
    for b = 1:length(boxes)
        patch(boxes(b).XData, boxes(b).YData, boxFillColorSpeed, ...
              'FaceAlpha', 0.5, 'EdgeColor', borderColorSpeed, 'LineWidth', 1.2);
    end

    ax = gca;
    ax.LineWidth = 1.1;
    ax.FontSize = 20;
    ax.FontName = 'Times New Roman';
    ax.Box = 'off';
    ax.XColor = 'k'; ax.YColor = 'k';
    ax.Title.String = ['Step=' num2str(M)];
    ax.Title.FontSize = 20;
    ax.YLabel.String = 'Power';
    ax.YLabel.FontSize = 24;
end

annotation('textbox', [0.1, 0.34, 0.8, 0.05], 'String', '(b) Linear Velocity (m/s)', ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 30);

%% --------- (c) Displacement Analysis ---------
disp_x = data{:, 10}; 
disp_y = data{:, 11}; 
disp_z = data{:, 12}; 
power = data{:, 20};  % Note: using column 20

yy = [0.08];

for kk = 1:length(M1)
    M = M1(kk);
    numBlocks = floor(length(power) / M);
    groupedPower = cell(1, 5);

    for i = 1:numBlocks
        idx = (i-1)*M+1 : i*M;
        avgDisp = mean([disp_x(idx), disp_y(idx), disp_z(idx)], 2);
        avgDisp = mean(avgDisp);

        if avgDisp <= 3
            groupIdx = 1;
        elseif avgDisp < 4.5
            groupIdx = 2;
        elseif avgDisp < 5.5
            groupIdx = 3;
        elseif avgDisp < 6.5
            groupIdx = 4;
        else
            groupIdx = 5;
        end

        groupedPower{groupIdx} = [groupedPower{groupIdx}, mean(power(idx))];
    end

    maxLen = max(cellfun(@length, groupedPower));
    for k = 1:5
        groupedPower{k}(end+1:maxLen) = NaN;
    end

    subplot('Position', [xx(kk), yy(1), width, height]);
    boxplot(cell2mat(groupedPower'), ...
        'Labels', {'3', '4', '5', '6', '7'}, ...
        'Symbol', 'o', 'OutlierSize', 3, 'Colors', borderColorDisp);

    outliers = findobj(gca, 'Tag', 'Outliers');
    set(outliers, 'MarkerEdgeColor', boxFillColorDisp, 'MarkerFaceColor', boxFillColorDisp);

    boxes = findobj(gca, 'Tag', 'Box');
    for b = 1:length(boxes)
        patch(boxes(b).XData, boxes(b).YData, boxFillColorDisp, ...
              'FaceAlpha', 0.5, 'EdgeColor', borderColorDisp, 'LineWidth', 1.2);
    end

    ax = gca;
    ax.LineWidth = 1.1;
    ax.FontSize = 20;
    ax.FontName = 'Times New Roman';
    ax.Box = 'off';
    ax.XColor = 'k'; ax.YColor = 'k';
    ax.Title.String = ['Step=' num2str(M)];
    ax.Title.FontSize = 20;
    ax.YLabel.String = 'Power';
    ax.YLabel.FontSize = 24;
end

annotation('textbox', [0.1, 0, 0.8, 0.05], 'String', '(c) Displacement (m)', ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 30);
