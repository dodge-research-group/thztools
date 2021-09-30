classdef SI
    %SI Fundamental constants in SI units
    
    properties (Constant)
        %% Fundamental constants (2019 SI)
        nuCs    = 9192631770;           % Caesium 133 frequency
        c       = 299792458;            % speed of light [m/s]
        h       = 6.62607015e-34;       % Planck's constant [J s]
        qe      = 1.602176634e-19;      % electron charge [C]
        kB      = 1.380649e-23;         % Boltzmann's constant [J/K]
        NA      = 6.02214076e23;        % Avagadro's constant [mol^-1]
        Kcd     = 683;                  % luminous efficacy of
                                        % monochromatic radiation of
                                        % frequency 540e12 Hz [lm/W]
       
        %% Electromagnetic constants
        alpha   = 1/137.035999139       % fine structure constant
        mu0     = 2*SI.h*SI.alpha/...
                    (SI.c*SI.qe^2);     % vacuum permeability [H/m]
        eps0    = 1/(SI.mu0*SI.c^2);    % vacuum permittivity [F/m]
        Z0      = sqrt(SI.mu0/SI.eps0); % vacuum impedance [Ohm]
        
        %% Electron properties
        me      = 9.10938291e-31;       % electron mass [kg]
        
        %% Proton properties
        mp      = 1.672621898e-27;      % proton mass [kg]
        
        %% Neutron properties
        mn      = 1.674927471e-27;      % neutron mass [kg]
        
        %% Quantum constants
        hbar    = SI.h/2/pi;            % hbar [J s]
        Ry      = SI.me*SI.qe^4/...		% Rydberg [J]
            (8*SI.eps0^2*SI.h^2);
        a0 = 4*pi*SI.eps0*SI.hbar^2/... % Bohr [m]
            (SI.me*SI.qe^2);
        
    end
        
end

