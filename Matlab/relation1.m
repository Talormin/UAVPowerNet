clc; close all; clear;

% Color palette
C1=[59 125 183;244 146 121;242 166 31;180 68 108;220 211 30]./255;
% ... (Other color definitions are kept for custom styling)

% =============================
% (a) Temperature - Boxplot
% =============================

% Read temperature dataset
data = readtable('HeNan1.csv');  % <-- Replace with relative or secured path

angular_velocity_x = data{:, 6}; 
angular_velocity_y = data{:, 6}; 
angular_velocity_z = data{:, 6}; 
power = data{:, 23};  

M1 = [5,10,20,40,80];

figure;
rows = 2; cols = 5; width = 0.15; height = 0.23;
xx = [0.035, 0.235, 0.435, 0.635, 0.835];
yy = [0.74];

for kk = 1:length(M1)
    M = M1(kk);
    numBlocks = floor(length(angular_velocity_x) / M);
    groupedPower = cell(1,10);  
    for i = 1:numBlocks
        idx = (i - 1) * M + 1 : i * M;
        avgAV = mean(mean([angular_velocity_x(idx), angular_velocity_y(idx), angular_velocity_z(idx)], 2));
        powerVal = mean(power(idx));

        % Grouping based on angular velocity
        groupIdx = min(floor(avgAV) - 26 + 1, 10);
        groupIdx = max(groupIdx, 1);
        groupedPower{groupIdx} = [groupedPower{groupIdx}, powerVal];
    end

    maxLength = max(cellfun(@length, groupedPower));
    for k = 1:10
        groupedPower{k}(end+1:maxLength) = NaN;
    end

    subplot('Position', [xx(kk), yy(1), width, height]);
    hold on;
    for jj = 1:4
        boxplot(groupedPower{jj}(:), 'Positions', jj, 'Widths', 0.5, ...
            'Colors', [22 173 0]/255, 'Symbol', 'o', 'OutlierSize', 3, 'Whisker', 5);
    end
    set(gca, 'XTick', 1:4, 'XTickLabel', {'26','27','28','29'});
    ax = gca;
    ax.Box = 'off'; ax.XColor = 'k'; ax.YColor = 'k';
    ax.LineWidth = 1.1; ax.FontSize = 20; ax.FontName = 'Times New Roman';
    title(['Step=' num2str(M)], 'FontSize', 26);
    ylabel('Power', 'FontSize', 26);
end
annotation('textbox', [0.1, 0.66, 0.8, 0.05], 'String', '(a) Temperature (°C)', ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 30);


% =============================
% (b) Altitude - Boxplot
% =============================

data = readtable('combined_output.csv');  % <-- Replace with relative or secured path
angular_velocity_x = data{:, 5};
angular_velocity_y = data{:, 5};
angular_velocity_z = data{:, 5};
power = data{:, 20};
yy = [0.41];

for kk = 1:length(M1)
    M = M1(kk);
    numBlocks = floor(length(angular_velocity_x) / M);
    groupedPower = cell(1,4);  
    for i = 1:numBlocks
        idx = (i - 1) * M + 1 : i * M;
        avgAV = mean(mean([angular_velocity_x(idx), angular_velocity_y(idx), angular_velocity_z(idx)], 2));
        powerVal = mean(power(idx));

        if avgAV <= 150
            groupIdx = 1;
        elseif avgAV < 250
            groupIdx = 2;
        elseif avgAV < 350
            groupIdx = 3;
        else
            groupIdx = 4;
        end
        groupedPower{groupIdx} = [groupedPower{groupIdx}, powerVal];
    end

    maxLength = max(cellfun(@length, groupedPower));
    for k = 1:4
        groupedPower{k}(end+1:maxLength) = NaN;
    end

    subplot('Position', [xx(kk), yy(1), width, height]);
    hold on;
    for jj = 1:4
        boxplot(groupedPower{jj}(:), 'Positions', jj, 'Widths', 0.5, ...
            'Colors', [146 158 0]/255, 'Symbol', 'o', 'OutlierSize', 3, 'Whisker', 5);
    end
    set(gca, 'XTick', 1:4, 'XTickLabel', {'100','200','300','400'});
    ax = gca;
    ax.Box = 'off'; ax.XColor = 'k'; ax.YColor = 'k';
    ax.LineWidth = 1.1; ax.FontSize = 20; ax.FontName = 'Times New Roman';
    title(['Step=' num2str(M)], 'FontSize', 26);
    ylabel('Power', 'FontSize', 26);
end
annotation('textbox', [0.1, 0.34, 0.8, 0.05], 'String', '(b) Altitude (m)', ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 30);


% =============================
% (c) Air Pressure - Boxplot
% =============================

angular_velocity_x = data{:, 6};
angular_velocity_y = data{:, 6};
angular_velocity_z = data{:, 6};
power = data{:, 20};
yy = [0.08];

for kk = 1:length(M1)
    M = M1(kk);
    numBlocks = floor(length(angular_velocity_x) / M);
    groupedPower = cell(1,4);  
    for i = 1:numBlocks
        idx = (i - 1) * M + 1 : i * M;
        avgAV = mean(mean([angular_velocity_x(idx), angular_velocity_y(idx), angular_velocity_z(idx)], 2));
        powerVal = mean(power(idx));

        if avgAV <= 96500
            groupIdx = 1;
        elseif avgAV < 97500
            groupIdx = 2;
        elseif avgAV < 98500
            groupIdx = 3;
        else
            groupIdx = 4;
        end
        groupedPower{groupIdx} = [groupedPower{groupIdx}, powerVal];
    end

    maxLength = max(cellfun(@length, groupedPower));
    for k = 1:4
        groupedPower{k}(end+1:maxLength) = NaN;
    end

    subplot('Position', [xx(kk), yy(1), width, height]);
    hold on;
    boxplot([groupedPower{1}(:), groupedPower{2}(:), groupedPower{3}(:), groupedPower{4}(:)], ...
        'Labels', {'96', '97', '98', '99'}, 'Symbol', 'o', ...
        'OutlierSize', 3, 'Colors', [166 51 3]/255, 'Whisker', 50);

    ax = gca;
    ax.Box = 'off'; ax.XColor = 'k'; ax.YColor = 'k';
    ax.LineWidth = 1.1; ax.FontSize = 20; ax.FontName = 'Times New Roman';
    title(['Step=' num2str(M)], 'FontSize', 26);
    ylabel('Power', 'FontSize', 26);
end
annotation('textbox', [0.1, 0, 0.8, 0.05], 'String', '(c) Pressure (Pa)', ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 32);
