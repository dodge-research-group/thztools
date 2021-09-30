classdef DataPulse
    %DataPulse is a class which can import scan data from an appropriately
    %formatted file and puts both the numerical and any supporting data
    %into class properties
    
    properties %any property that can appear in the file should be here
        AcquisitionTime = ...   % Time that the data pulse was taken
            datetime(1,'ConvertFrom','datenum');
        TimeConstant = NaN;     % Time constant on the lock-in, in ms
        WaitTime = NaN;         % Time waited at each point, in ms
        Description = '';       % Description of the type of scan or sample
        SetPoint = NaN;
        ScanOffset = NaN;
        Temperature = NaN;      % Temperature, in K, of the sample
        Time = NaN;             % Data for time axis, in ps
        Amplitude = NaN;        % Voltage data, in mV
        ChannelAMaximum = NaN;  % Maximum temperature, Channel A (K)
        ChannelAMinimum = NaN;  % Minimum temperature, Channel A (K)
        ChannelAVariance = NaN; % Temperature variance, Channel A (K^2)
        ChannelASlope = NaN;    % dT/dt, Channel A (K/min)
        ChannelAOffset = NaN;   % Average temperature, Channel A (K)
        ChannelBMaximum = NaN;
        ChannelBMinimum = NaN;
        ChannelBVariance = NaN;
        ChannelBSlope = NaN;
        ChannelBOffset = NaN;
        ChannelCMaximum = NaN;
        ChannelCMinimum = NaN;
        ChannelCVariance = NaN;
        ChannelCSlope = NaN;
        ChannelCOffset = NaN;
        ChannelDMaximum = NaN;
        ChannelDMinimum = NaN;
        ChannelDVariance = NaN;
        ChannelDSlope = NaN;
        ChannelDOffset = NaN;
        DirName = '';
        File = struct('name','', ...
            'data','', ...
            'bytes',0, ...
            'isdir',false, ...
            'datenum',0);
    end
    
    properties (Dependent)      % Properties calculated from data
        Frequency               % Frequency axis data for spectrum
        FourierAmplitude        % Complex FFT amplitude of the signal
    end
    
    methods
        function obj = DataPulse(filename) %initialize from a datafile or as an empty pulse
            if nargin==1
                obj.DirName = fileparts(filename);
                obj.File = dir(filename);
                fid = fopen(filename);
                %first get the time the scan was taken, which appears on
                %the first line of the file
                obj.AcquisitionTime = datetime(fscanf(fid,'%*s%s\n',1),...
                    'InputFormat','yyyy-MM-dd''T''HH:mm:ss');
                %now keep grabbing each line of the file and matching it to
                %the property of the same name
                str = fgetl(fid);
                while ~strcmp(str,'')
                    info = textscan(str,'%s%s',...
                        'Whitespace','','Delimiter','\t');
                    if ~isempty(info{2}{1})
                        if ischar(obj.(info{1}{1}))
                            obj.(info{1}{1}) = info{2}{1};
                        elseif isnumeric(obj.(info{1}{1}))
                            obj.(info{1}{1}) = str2double(info{2}{1});
                        end
                    end
                    str = fgetl(fid);
                end
                data = fscanf(fid,'%f',[2,Inf])'; %now imports the numerical scan data
                fclose(fid);
                obj.Time = data(:,1);
                obj.Amplitude = data(:,2);
            end
        end
        function frequency = get.Frequency(obj) %calculate frequency range
            frequency = (0:floor(length(obj.Time))/2-1)'/(obj.Time(end)-obj.Time(1));
        end
        function famp = get.FourierAmplitude(obj) %calculate FFT
            famp = (fft(obj.Amplitude));
            famp = famp(1:floor(length(famp)/2));
        end
        function plot(obj,varargin) %plots the time trace and power spectrum
            figure()
            hold on
            for o = obj
                plot(o.Time,o.Amplitude,varargin{:})
            end
            xlabel('Time (ps)')
            ylabel('Voltage (V)')
            figure()
            hold on
            for o = obj
                plot(o.Frequency,10*log10(abs(o.FourierAmplitude).^2),varargin{:})
            end
            xlabel('Frequency (THz)')
            ylabel('Power (dB)')
            hold off
        end
    end
end